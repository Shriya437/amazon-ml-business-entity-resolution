"""
BLOCKING V4 — MEMORY-SAFE V3-PRESERVING + ADDITIVE
Amazon ML Challenge 2026 — Business Entity Resolution

Validation-first optimized version.

Key design:
- Preserves all empirically validated V3 channels.
- Adds true IDF-weighted name overlap, phonetic matching and sorted-token exact.
- MinHash and ANN are OFF for the first validation to keep RAM safe.
- Candidate-side records are NEVER materialized as millions of Python objects.
- Posting lists store compact integer row IDs using array('I').
- Candidate-side data is read in chunks.
- Only one country is indexed at a time.
- Validation sample is random/stratified and reusable.
- Country partition integrity is checked before blocking.
- candidate_pairs_v4.tsv is the exact candidate set produced by the settings.
"""

from __future__ import annotations

import json
import math
import re
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Optional: only phonetic matching needs this package.
try:
    from metaphone import doublemetaphone
except ImportError:
    doublemetaphone = None


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("processed_data")
STAGE1_TRAIN_DIR = OUTPUT_DIR / "stage1" / "train"
GROUND_TRUTH = OUTPUT_DIR / "stage0" / "train_ground_truth.tsv"

VALIDATION_DIR = OUTPUT_DIR / "validation"
VALIDATION_IDS = VALIDATION_DIR / "validation_s1_ids.txt"

OUTPUT_FILE = VALIDATION_DIR / "candidate_pairs_v4.tsv"
METRICS_FILE = VALIDATION_DIR / "blocking_metrics_v4.json"
CHANNEL_METRICS_FILE = VALIDATION_DIR / "blocking_channel_metrics_v4.json"

# First run: validation only.
FULL_RUN = False
VALIDATION_S1_LIMIT = 10_000
RANDOM_SEED = 20260927

# DO NOT cap during the first validation.
FINAL_CAP = None

# V4 lexical channels.
# MinHash is deliberately OFF for the first memory-safe validation.
ENABLE_MINHASH = False
ENABLE_PHONETIC = True

# ANN deliberately OFF.
ENABLE_EMBEDDINGS = False
ENABLE_BIDIRECTIONAL_ANN = False

# Only NEW V4 inverted blocks are bounded.
V4_MAX_BLOCK_SIZE = 5_000

# IDF settings.
MIN_TOKEN_DOCS = 2
MAX_TOKEN_DF_RATIO = 0.02
MIN_TOKEN_IDF = 2.0

# Operational settings.
CHUNK_SIZE = 50_000
PROGRESS_EVERY_CHUNKS = 5

# Set True only if you specifically want a large detail file.
WRITE_DETAIL_FILE = False
DETAIL_FILE = VALIDATION_DIR / "candidate_details_v4.tsv"

# Compiled regexes for speed.
TOKEN_RE = re.compile(r"[a-z0-9]+")
WS_RE = re.compile(r"\s+")
NON_ALNUM_WS_RE = re.compile(r"[^a-z0-9\s]+")


# ============================================================
# V3 CHANNELS — PRESERVED EXACTLY
# ============================================================

V3_RULES = (
    "name_prefix6",
    "name_first",
    "name_first_last",
    "address_number",
    "address_number_token",
    "address_prefix6",
    "addr_number_two_tokens",
    "addr_two_numbers",
    "addr_number_last_token",
    "addr_first_two_tokens",
    "addr_first_three_tokens",
)


# ============================================================
# BASIC HELPERS
# ============================================================

def text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def row_value(row, column: str) -> str:
    return text(getattr(row, column, ""))


def is_true(value) -> bool:
    return text(value).lower() in {"1", "true", "yes", "y"}


def tokenize(value) -> list[str]:
    value = text(value).lower()
    if not value:
        return []
    return TOKEN_RE.findall(value)


def token_set(value) -> set[str]:
    return set(tokenize(value))


def parse_locality_tokens(value) -> list[str]:
    value = text(value)
    if not value:
        return []

    if value.startswith("[") and value.endswith("]"):
        values = re.findall(r"['\"]([^'\"]+)['\"]", value)
        if values:
            out = []
            for item in values:
                out.extend(tokenize(item))
            return out

    return tokenize(value)


def country_slug(country: str) -> str:
    value = text(country)
    if not value:
        return "__unknown__"
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "__unknown__"


def validate_input_columns(path: Path) -> None:
    columns = set(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            nrows=0,
        ).columns
    )

    required = {
        "entity_id",
        "country",
        "name_full",
        "name_no_suffix",
        "name_sorted",
        "name_first2",
        "name_ascii",
        "name_transliterated",
        "name_transliterated_no_suffix",
        "has_address",
        "pin_code",
        "zip_code",
        "street_num",
        "locality_tokens",
        "state_token",
    }

    missing = required - columns
    if missing:
        raise RuntimeError(
            f"{path} is missing required normalized columns: {sorted(missing)}"
        )


def discover_country_files():
    files = {}

    for path in STAGE1_TRAIN_DIR.glob("*_s1.tsv"):
        slug = path.name[:-len("_s1.tsv")]
        files.setdefault(slug, {})["s1"] = path

    for path in STAGE1_TRAIN_DIR.glob("*_s2.tsv"):
        slug = path.name[:-len("_s2.tsv")]
        files.setdefault(slug, {})["s2"] = path

    for path in STAGE1_TRAIN_DIR.glob("*_s3.tsv"):
        slug = path.name[:-len("_s3.tsv")]
        files.setdefault(slug, {})["s3"] = path

    result = []
    for slug in sorted(files):
        item = files[slug]
        if "s1" not in item:
            continue

        result.append(
            (
                slug,
                item["s1"],
                item.get("s2", STAGE1_TRAIN_DIR / f"{slug}_s2.tsv"),
                item.get("s3", STAGE1_TRAIN_DIR / f"{slug}_s3.tsv"),
            )
        )

    return result


# ============================================================
# REPRESENTATION / FEATURE EXTRACTION
# ============================================================

NAME_COLUMNS = (
    "name_no_suffix",
    "name_transliterated_no_suffix",
    "name_full",
    "name_transliterated",
    "name_ascii",
)


def get_name_variants(row) -> list[str]:
    out = []
    seen = set()

    for column in NAME_COLUMNS:
        value = row_value(row, column)
        if value and value not in seen:
            out.append(value)
            seen.add(value)

    return out


def get_primary_name(row) -> str:
    for column in NAME_COLUMNS:
        value = row_value(row, column)
        if value:
            return value
    return ""


def get_name_tokens(row) -> set[str]:
    result = set()
    for value in get_name_variants(row):
        result.update(tokenize(value))
    return result


def get_sorted_name(row) -> str:
    return " ".join(sorted(token_set(row_value(row, "name_sorted"))))


def get_address_tokens(row) -> list[str]:
    locality = parse_locality_tokens(row_value(row, "locality_tokens"))
    state = row_value(row, "state_token").lower()
    street_num = row_value(row, "street_num")
    pin = row_value(row, "pin_code")
    zip_code = row_value(row, "zip_code")

    values = locality + ([state] if state else [])

    if street_num:
        values.append(street_num)
    if pin:
        values.append(pin)
    if zip_code:
        values.append(zip_code)

    return [x for x in values if x]


def get_address_text(row) -> str:
    return " ".join(get_address_tokens(row))


def get_first_name_token(row) -> str:
    for value in get_name_variants(row):
        tokens = tokenize(value)
        if tokens:
            return tokens[0]
    return ""


def get_last_name_token(row) -> str:
    for value in get_name_variants(row):
        tokens = tokenize(value)
        if tokens:
            return tokens[-1]
    return ""


def get_name_prefix6(row) -> str:
    return WS_RE.sub("", get_primary_name(row).lower())[:6]


def get_name_first_last(row) -> str:
    first = get_first_name_token(row)
    last = get_last_name_token(row)
    return f"{first}|{last}" if first and last else ""


def get_address_prefix6(row) -> str:
    return WS_RE.sub("", get_address_text(row).lower())[:6]


def get_address_number(row) -> str:
    return row_value(row, "street_num")


def get_address_numbers(row) -> list[str]:
    result = []

    street_num = get_address_number(row)
    if street_num:
        result.append(street_num)

    for token in get_address_tokens(row):
        if re.fullmatch(r"\d+[a-z]?", token.lower()) and token not in result:
            result.append(token)

    return result


def get_address_non_numeric_tokens(row) -> list[str]:
    return [
        token.lower()
        for token in get_address_tokens(row)
        if not re.fullmatch(r"\d+[a-z]?", token.lower())
    ]


def get_address_number_token(row) -> str:
    number = get_address_number(row)
    tokens = get_address_non_numeric_tokens(row)
    return f"{number}|{tokens[0]}" if number and tokens else ""


def get_addr_number_two_tokens(row) -> str:
    number = get_address_number(row)
    tokens = get_address_non_numeric_tokens(row)
    return (
        f"{number}|{tokens[0]}|{tokens[1]}"
        if number and len(tokens) >= 2
        else ""
    )


def get_addr_two_numbers(row) -> str:
    numbers = get_address_numbers(row)
    return f"{numbers[0]}|{numbers[1]}" if len(numbers) >= 2 else ""


def get_addr_number_last_token(row) -> str:
    number = get_address_number(row)
    tokens = get_address_non_numeric_tokens(row)
    return f"{number}|{tokens[-1]}" if number and tokens else ""


def get_addr_first_two_tokens(row) -> str:
    tokens = get_address_non_numeric_tokens(row)
    return f"{tokens[0]}|{tokens[1]}" if len(tokens) >= 2 else ""


def get_addr_first_three_tokens(row) -> str:
    tokens = get_address_non_numeric_tokens(row)
    return (
        f"{tokens[0]}|{tokens[1]}|{tokens[2]}"
        if len(tokens) >= 3
        else ""
    )


def extract_features(row):
    """
    Extract the same V3 keys as the previous V4 implementation,
    but without creating a persistent Python Record object.
    """

    entity_id = row_value(row, "entity_id")
    name_tokens = get_name_tokens(row)
    sorted_name = get_sorted_name(row)
    name_text = get_primary_name(row)

    address_tokens = get_address_tokens(row)
    address_text = " ".join(address_tokens)
    has_address = is_true(row_value(row, "has_address"))

    v3 = {
        "name_prefix6": WS_RE.sub("", name_text.lower())[:6],
        "name_first": get_first_name_token(row),
        "name_first_last": get_name_first_last(row),
        "address_number": get_address_number(row),
        "address_number_token": get_address_number_token(row),
        "address_prefix6": WS_RE.sub("", address_text.lower())[:6],
        "addr_number_two_tokens": get_addr_number_two_tokens(row),
        "addr_two_numbers": get_addr_two_numbers(row),
        "addr_number_last_token": get_addr_number_last_token(row),
        "addr_first_two_tokens": get_addr_first_two_tokens(row),
        "addr_first_three_tokens": get_addr_first_three_tokens(row),
    }

    phonetic_codes = set()

    if ENABLE_PHONETIC and doublemetaphone is not None:
        for token in name_tokens:
            code1, code2 = doublemetaphone(token)
            if code1:
                phonetic_codes.add(code1)
            if code2:
                phonetic_codes.add(code2)

    return (
        entity_id,
        name_tokens,
        sorted_name,
        name_text,
        address_text,
        has_address,
        v3,
        phonetic_codes,
    )


# ============================================================
# COMPACT POSTING INDEX
# ============================================================

class CompactPosting:
    """
    key -> compact array of uint32 local row IDs.

    This is much smaller than Python set[str] postings and avoids the
    previous multi-million-object memory spike.
    """

    def __init__(self, max_size=None):
        self.max_size = max_size
        self.data = {}
        self.oversized = set()

    def add(self, key, local_id: int):
        if not key or key in self.oversized:
            return

        bucket = self.data.get(key)

        if bucket is None:
            bucket = array("I")
            self.data[key] = bucket

        bucket.append(local_id)

        if self.max_size is not None and len(bucket) > self.max_size:
            del self.data[key]
            self.oversized.add(key)

    def get(self, key):
        if not key or key in self.oversized:
            return ()
        return self.data.get(key, ())


# ============================================================
# COUNTRY INDEX
# ============================================================

class CompactCountryIndex:
    """
    Memory-safe candidate-side index.

    Candidate files are streamed in CHUNK_SIZE chunks.
    Only compact integer postings are retained.
    """

    def __init__(self, s2_path: Path, s3_path: Path):
        self.entity_ids = []
        self.id_to_local = {}

        self.v3 = {
            rule: CompactPosting(max_size=None)
            for rule in V3_RULES
        }

        self.sorted_posting = CompactPosting(
            max_size=V4_MAX_BLOCK_SIZE
        )

        self.phonetic_posting = CompactPosting(
            max_size=V4_MAX_BLOCK_SIZE
        )

        self.token_df = Counter()
        self.token_posting = None
        self.token_idf = {}
        self.common_tokens = set()

        self.total_rows = 0

        self._build_first_pass(s2_path, s3_path)
        self._finalize_idf(s2_path, s3_path)

    def _iter_chunks(self, path: Path, usecols=None):
        if not path.exists():
            return

        kwargs = {
            "sep": "\t",
            "dtype": str,
            "keep_default_na": False,
            "chunksize": CHUNK_SIZE,
        }

        if usecols is not None:
            kwargs["usecols"] = usecols

        for chunk in pd.read_csv(path, **kwargs):
            yield chunk

    def _build_first_pass(self, s2_path, s3_path):
        paths = [p for p in (s2_path, s3_path) if p.exists()]

        for source_no, path in enumerate(paths, start=1):
            rows_seen_source = 0
            chunk_no = 0

            for chunk in self._iter_chunks(path):
                chunk_no += 1

                for row in chunk.itertuples(index=False, name="NormalizedRow"):
                    (
                        entity_id,
                        name_tokens,
                        sorted_name,
                        _name_text,
                        _address_text,
                        _has_address,
                        v3,
                        phonetic_codes,
                    ) = extract_features(row)

                    if not entity_id:
                        continue

                    if entity_id in self.id_to_local:
                        raise RuntimeError(
                            f"Duplicate candidate entity_id across S2/S3: {entity_id}"
                        )

                    local_id = len(self.entity_ids)
                    self.entity_ids.append(entity_id)
                    self.id_to_local[entity_id] = local_id

                    for rule in V3_RULES:
                        self.v3[rule].add(
                            v3[rule],
                            local_id,
                        )

                    self.sorted_posting.add(
                        sorted_name,
                        local_id,
                    )

                    for token in name_tokens:
                        self.token_df[token] += 1

                    if ENABLE_PHONETIC:
                        for code in phonetic_codes:
                            self.phonetic_posting.add(
                                code,
                                local_id,
                            )

                rows_seen_source += len(chunk)
                self.total_rows += len(chunk)

                if (
                    chunk_no == 1
                    or chunk_no % PROGRESS_EVERY_CHUNKS == 0
                ):
                    print(
                        f"    indexing {path.name}: "
                        f"{rows_seen_source:,} rows"
                    )

    def _finalize_idf(self, s2_path, s3_path):
        total_docs = max(1, len(self.entity_ids))

        self.token_idf = {
            token: math.log(
                (1 + total_docs) / (1 + df)
            ) + 1.0
            for token, df in self.token_df.items()
        }

        self.common_tokens = {
            token
            for token, df in self.token_df.items()
            if (
                df < MIN_TOKEN_DOCS
                or df / total_docs > MAX_TOKEN_DF_RATIO
                or self.token_idf[token] < MIN_TOKEN_IDF
            )
        }

        self.token_posting = CompactPosting(
            max_size=V4_MAX_BLOCK_SIZE
        )

        # Second streaming pass: build only informative-token postings.
        paths = [p for p in (s2_path, s3_path) if p.exists()]

        for path in paths:
            rows_seen = 0
            chunk_no = 0

            for chunk in self._iter_chunks(
                path,
                usecols=[
                    "entity_id",
                    "name_no_suffix",
                    "name_transliterated_no_suffix",
                    "name_full",
                    "name_transliterated",
                    "name_ascii",
                ],
            ):
                chunk_no += 1

                for row in chunk.itertuples(
                    index=False,
                    name="NameRow",
                ):
                    entity_id = row_value(row, "entity_id")
                    local_id = self.id_to_local.get(entity_id)

                    if local_id is None:
                        continue

                    name_tokens = get_name_tokens(row)

                    for token in name_tokens:
                        if token not in self.common_tokens:
                            self.token_posting.add(
                                token,
                                local_id,
                            )

                rows_seen += len(chunk)

                if (
                    chunk_no == 1
                    or chunk_no % PROGRESS_EVERY_CHUNKS == 0
                ):
                    print(
                        f"    IDF postings {path.name}: "
                        f"{rows_seen:,} rows"
                    )

    def release(self):
        self.v3.clear()
        self.sorted_posting = None
        self.phonetic_posting = None
        self.token_posting = None
        self.token_df.clear()
        self.token_idf.clear()
        self.common_tokens.clear()
        self.entity_ids.clear()
        self.id_to_local.clear()


# ============================================================
# V4 CHANNELS
# ============================================================

V4_BITS = {
    "B": 1 << 0,
    "D": 1 << 1,
    "G": 1 << 2,
}


def channel_b_idf(
    name_tokens: set[str],
    index: CompactCountryIndex,
):
    """
    TRUE IDF-weighted overlap.

    score(candidate) =
        sum(IDF(shared informative query tokens))
        / sum(IDF(all informative query tokens))
    """

    informative = [
        token
        for token in name_tokens
        if token not in index.common_tokens
    ]

    if not informative:
        return {}

    query_mass = sum(
        index.token_idf.get(token, 0.0)
        for token in informative
    )

    if query_mass <= 0:
        return {}

    weighted = defaultdict(float)

    for token in informative:
        weight = index.token_idf.get(token, 0.0)

        for local_id in index.token_posting.get(token):
            weighted[local_id] += weight

    return {
        local_id: score / query_mass
        for local_id, score in weighted.items()
    }


def channel_d_phonetic(
    phonetic_codes: set[str],
    index: CompactCountryIndex,
):
    if not ENABLE_PHONETIC or doublemetaphone is None:
        return ()

    hits = set()

    for code in phonetic_codes:
        hits.update(
            index.phonetic_posting.get(code)
        )

    return hits


# ============================================================
# CANDIDATE GENERATION + RANKING
# ============================================================

def add_candidate(
    candidates,
    local_id,
    *,
    v3=False,
    channel_bit=0,
    b_score=0.0,
):
    info = candidates.get(local_id)

    if info is None:
        info = [0, 0, 0.0]
        candidates[local_id] = info

    if v3:
        info[0] += 1

    if channel_bit:
        info[1] |= channel_bit

    if b_score > info[2]:
        info[2] = b_score


def popcount(value: int) -> int:
    return value.bit_count()


def generate_and_rank(
    record_features,
    index: CompactCountryIndex,
):
    (
        _entity_id,
        name_tokens,
        sorted_name,
        _name_text,
        _address_text,
        _has_address,
        v3,
        phonetic_codes,
    ) = record_features

    candidates = {}
    channel_hits = {}

    # --------------------------------------------------------
    # V3 — ALL PRESERVED
    # --------------------------------------------------------
    for rule in V3_RULES:
        hits = set(
            index.v3[rule].get(v3[rule])
        )
        channel_hits[f"V3:{rule}"] = hits

        for local_id in hits:
            add_candidate(
                candidates,
                local_id,
                v3=True,
            )

    # --------------------------------------------------------
    # V4 B — TRUE IDF WEIGHTING
    # --------------------------------------------------------
    b_hits = channel_b_idf(
        name_tokens,
        index,
    )

    for local_id, score in b_hits.items():
        add_candidate(
            candidates,
            local_id,
            channel_bit=V4_BITS["B"],
            b_score=score,
        )

    # --------------------------------------------------------
    # V4 D — PHONETIC
    # --------------------------------------------------------
    d_hits = channel_d_phonetic(
        phonetic_codes,
        index,
    )

    for local_id in d_hits:
        add_candidate(
            candidates,
            local_id,
            channel_bit=V4_BITS["D"],
        )

    # --------------------------------------------------------
    # V4 G — SORTED TOKEN EXACT
    # --------------------------------------------------------
    g_hits = index.sorted_posting.get(
        sorted_name
    )

    for local_id in g_hits:
        add_candidate(
            candidates,
            local_id,
            channel_bit=V4_BITS["G"],
        )

    # --------------------------------------------------------
    # Ranking
    #
    # This ranking is deliberately based only on features already
    # available in the compact indexes. It avoids loading millions
    # of candidate records just to calculate expensive edit distance.
    # --------------------------------------------------------
    ranked = []

    for local_id, info in candidates.items():
        v3_count, v4_mask, b_score = info
        v4_count = popcount(v4_mask)

        rank_score = (
            2.5 * v3_count
            + 1.0 * v4_count
            + 1.5 * b_score
        )

        ranked.append(
            (
                rank_score,
                index.entity_ids[local_id],
                local_id,
            )
        )

    ranked.sort(
        key=lambda x: (
            -x[0],
            x[1],
        )
    )

    if FINAL_CAP is not None:
        ranked = ranked[:FINAL_CAP]

    channel_hits["B"] = set(b_hits)
    channel_hits["D"] = set(d_hits)
    channel_hits["G"] = set(g_hits)

    return ranked, candidates, channel_hits


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth_counts():
    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f"Ground truth missing: {GROUND_TRUTH}"
        )

    counts = {}

    for chunk in pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    ):
        for row in chunk.itertuples(
            index=False,
            name="GT",
        ):
            sid = text(row.source1_entity_id)
            raw = text(row.matched_entity_ids)

            if not sid:
                continue

            counts[sid] = (
                len(
                    {
                        x.strip()
                        for x in raw.split(",")
                        if x.strip()
                    }
                )
                if raw
                else 0
            )

    return counts


def load_selected_ground_truth(selected_ids):
    result = {
        sid: set()
        for sid in selected_ids
    }

    for chunk in pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=100_000,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    ):
        for row in chunk.itertuples(
            index=False,
            name="GT",
        ):
            sid = text(row.source1_entity_id)

            if sid not in result:
                continue

            raw = text(row.matched_entity_ids)

            result[sid] = {
                x.strip()
                for x in raw.split(",")
                if x.strip()
            }

    return result


# ============================================================
# RANDOM / STRATIFIED VALIDATION SAMPLE
# ============================================================

def match_bucket(match_count: int) -> str:
    if match_count == 0:
        return "0"
    if match_count == 1:
        return "1"
    if match_count <= 4:
        return "2_4"
    return "5_plus"


def choose_validation_ids(country_files, match_counts):
    if FULL_RUN:
        return None

    if VALIDATION_IDS.exists():
        existing = pd.read_csv(
            VALIDATION_IDS,
            dtype=str,
        )

        column = (
            "source1_entity_id"
            if "source1_entity_id" in existing.columns
            else existing.columns[0]
        )

        ids = (
            existing[column]
            .dropna()
            .astype(str)
            .tolist()
        )

        if len(ids) >= VALIDATION_S1_LIMIT:
            print(
                f"Using existing validation sample: "
                f"{VALIDATION_IDS}"
            )
            return set(
                ids[:VALIDATION_S1_LIMIT]
            )

    # Store only IDs by stratum, not a giant DataFrame.
    strata_ids = defaultdict(list)

    for slug, s1_path, _, _ in country_files:
        for chunk in pd.read_csv(
            s1_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            usecols=["entity_id"],
        ):
            for sid in chunk["entity_id"]:
                sid = text(sid)
                if not sid:
                    continue

                count = match_counts.get(sid, 0)

                strata_ids[
                    (slug, match_bucket(count))
                ].append(sid)

    total_population = sum(
        len(ids)
        for ids in strata_ids.values()
    )

    target = min(
        VALIDATION_S1_LIMIT,
        total_population,
    )

    # Proportional allocation with largest-remainder correction.
    keys = sorted(strata_ids)
    populations = np.array(
        [len(strata_ids[k]) for k in keys],
        dtype=np.int64,
    )

    raw = (
        populations
        / max(1, populations.sum())
        * target
    )

    allocations = np.floor(raw).astype(int)
    remaining = target - int(
        allocations.sum()
    )

    fractions = raw - allocations

    if remaining > 0:
        order = np.argsort(
            -fractions,
            kind="stable",
        )

        for idx in order[:remaining]:
            allocations[idx] += 1

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selected = []

    manifest_rows = []

    for idx, key in enumerate(keys):
        slug, bucket = key
        ids = strata_ids[key]
        n = int(allocations[idx])

        if n <= 0:
            continue

        if n >= len(ids):
            chosen = list(ids)
        else:
            chosen = rng.choice(
                np.asarray(ids),
                size=n,
                replace=False,
            ).tolist()

        selected.extend(chosen)

        for sid in chosen:
            manifest_rows.append(
                (
                    sid,
                    slug,
                    match_counts.get(sid, 0),
                    bucket,
                )
            )

    # Exact-size correction.
    selected = list(dict.fromkeys(selected))

    if len(selected) < target:
        selected_set = set(selected)
        remaining_ids = []

        for ids in strata_ids.values():
            remaining_ids.extend(
                sid
                for sid in ids
                if sid not in selected_set
            )

        extra = rng.choice(
            np.asarray(remaining_ids),
            size=target - len(selected),
            replace=False,
        ).tolist()

        selected.extend(extra)

    elif len(selected) > target:
        selected = rng.choice(
            np.asarray(selected),
            size=target,
            replace=False,
        ).tolist()

    selected = list(
        dict.fromkeys(selected)
    )[:target]

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "entity_id",
            "country",
            "match_count",
            "bucket",
        ],
    )

    manifest = manifest[
        manifest["entity_id"].isin(selected)
    ]

    manifest.sort_values(
        ["country", "bucket", "entity_id"]
    ).to_csv(
        VALIDATION_DIR
        / "validation_sample_manifest.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        {
            "source1_entity_id": selected
        }
    ).to_csv(
        VALIDATION_IDS,
        index=False,
    )

    print(
        f"Created new random/stratified validation "
        f"sample: {len(selected):,} S1 entities"
    )
    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print("\nValidation strata:")
    if not manifest.empty:
        print(
            manifest.groupby(
                ["country", "bucket"]
            ).size().to_string()
        )

    return set(selected)


# ============================================================
# SELECTED S1 LOADING
# ============================================================

def load_selected_s1(
    path: Path,
    selected_ids: set[str] | None,
):
    if selected_ids is None:
        # FULL_RUN path: stream chunks outside this function.
        raise RuntimeError(
            "Use iter_s1_chunks() for FULL_RUN."
        )

    pieces = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        matched = chunk[
            chunk["entity_id"].isin(selected_ids)
        ]

        if not matched.empty:
            pieces.append(matched)

    if not pieces:
        return pd.DataFrame()

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def iter_s1_chunks(
    path: Path,
    selected_ids: set[str] | None,
):
    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        if selected_ids is None:
            yield chunk
        else:
            matched = chunk[
                chunk["entity_id"].isin(selected_ids)
            ]

            if not matched.empty:
                yield matched


# ============================================================
# COUNTRY PARTITION INTEGRITY
# ============================================================

def validate_country_partition(
    selected_ids,
    ground_truth,
    country_files,
):
    s1_country = {}
    candidate_country = {}

    # S1 country.
    for slug, s1_path, _, _ in country_files:
        for chunk in pd.read_csv(
            s1_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=CHUNK_SIZE,
            usecols=["entity_id", "country"],
        ):
            matched = chunk[
                chunk["entity_id"].isin(selected_ids)
            ]

            for row in matched.itertuples(
                index=False,
                name="CountryRow",
            ):
                s1_country[
                    text(row.entity_id)
                ] = text(row.country)

    needed_by_country = defaultdict(set)

    for sid in selected_ids:
        s1c = country_slug(
            s1_country.get(sid, "")
        )
        needed_by_country[s1c].update(
            ground_truth.get(sid, set())
        )

    # Candidate IDs are searched only in their own partition.
    for slug, _, s2_path, s3_path in country_files:
        wanted = needed_by_country.get(
            slug,
            set(),
        )

        if not wanted:
            continue

        for candidate_path in (
            s2_path,
            s3_path,
        ):
            if not candidate_path.exists():
                continue

            for chunk in pd.read_csv(
                candidate_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=CHUNK_SIZE,
                usecols=["entity_id", "country"],
            ):
                matched = chunk[
                    chunk["entity_id"].isin(wanted)
                ]

                for row in matched.itertuples(
                    index=False,
                    name="CandidateCountry",
                ):
                    candidate_country[
                        text(row.entity_id)
                    ] = text(row.country)

    total_true = 0
    missing = 0
    cross = 0
    examples_missing = []
    examples_cross = []

    for sid in selected_ids:
        s1c = s1_country.get(sid, "")
        true_ids = ground_truth.get(
            sid,
            set(),
        )

        for true_id in true_ids:
            total_true += 1

            if true_id not in candidate_country:
                missing += 1
                if len(examples_missing) < 10:
                    examples_missing.append(
                        [sid, true_id, s1c]
                    )
                continue

            if candidate_country[true_id] != s1c:
                cross += 1
                if len(examples_cross) < 10:
                    examples_cross.append(
                        [
                            sid,
                            true_id,
                            s1c,
                            candidate_country[true_id],
                        ]
                    )

    result = {
        "selected_s1_entities": len(
            selected_ids
        ),
        "true_pairs_checked": total_true,
        "true_ids_missing_from_stage1": missing,
        "cross_country_true_pairs": cross,
        "partition_true_pair_loss": (
            missing + cross
        ),
        "partition_true_pair_loss_rate": (
            (missing + cross) / total_true
            if total_true
            else 0.0
        ),
    }

    if examples_missing:
        result["missing_examples"] = examples_missing

    if examples_cross:
        result["cross_country_examples"] = examples_cross

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print("BLOCKING V4 — OPTIMIZED MEMORY-SAFE VALIDATION")
    print("=" * 90)
    print(
        f"Stage-1 train directory : {STAGE1_TRAIN_DIR}"
    )
    print(
        f"FULL_RUN                : {FULL_RUN}"
    )
    print(
        f"Validation S1 limit     : "
        f"{VALIDATION_S1_LIMIT:,}"
    )
    print(
        f"FINAL_CAP               : {FINAL_CAP}"
    )
    print(
        f"MinHash                 : {ENABLE_MINHASH} "
        f"(disabled for memory-safe first validation)"
    )
    print(
        f"Phonetic                : {ENABLE_PHONETIC}"
    )
    print(
        f"Embeddings              : {ENABLE_EMBEDDINGS}"
    )
    print(
        f"Random seed             : {RANDOM_SEED}"
    )
    print(
        "Posting storage         : compact uint32 arrays"
    )

    if not STAGE1_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Stage-1 train directory not found: "
            f"{STAGE1_TRAIN_DIR}"
        )

    country_files = discover_country_files()

    if not country_files:
        raise RuntimeError(
            f"No country partition files found in "
            f"{STAGE1_TRAIN_DIR}"
        )

    print(
        f"\nCountry partitions found: "
        f"{', '.join(x[0] for x in country_files)}"
    )

    for _, s1_path, s2_path, s3_path in country_files:
        for path in (
            s1_path,
            s2_path,
            s3_path,
        ):
            if path.exists():
                validate_input_columns(path)

    # --------------------------------------------------------
    # Ground truth / validation sample.
    # --------------------------------------------------------
    if FULL_RUN:
        validation_ids = None
        ground_truth = {}
        print(
            "\nFULL_RUN=True: recall diagnostics are disabled."
        )
    else:
        print(
            "\nLoading ground-truth match counts "
            "for stratified sampling..."
        )

        match_counts = load_ground_truth_counts()

        validation_ids = choose_validation_ids(
            country_files,
            match_counts,
        )

        print(
            f"\nSelected validation S1 IDs: "
            f"{len(validation_ids):,}"
        )

        print(
            "\nLoading ground truth only for "
            "the selected validation S1..."
        )

        ground_truth = load_selected_ground_truth(
            validation_ids
        )

    # --------------------------------------------------------
    # Country integrity.
    # --------------------------------------------------------
    if not FULL_RUN:
        print("\n" + "=" * 90)
        print("COUNTRY PARTITION INTEGRITY CHECK")
        print("=" * 90)

        partition_metrics = (
            validate_country_partition(
                validation_ids,
                ground_truth,
                country_files,
            )
        )

        for key, value in partition_metrics.items():
            print(f"{key}: {value}")

        if (
            partition_metrics[
                "partition_true_pair_loss"
            ]
            != 0
        ):
            raise RuntimeError(
                "Country partition integrity FAILED."
            )

        print(
            "\n✓ Country partitioning loses ZERO "
            "true pairs in the validation sample."
        )
    else:
        partition_metrics = {
            "status": "not_run_in_full_run"
        }

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_handle = open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    )

    candidate_handle.write(
        "source1_entity_id\tcandidate_entity_ids\n"
    )

    selected_s1_count = 0
    total_candidate_pairs = 0
    candidate_counts = []

    total_true_pairs = 0
    recovered_true_pairs = 0

    zero_match_s1_count = 0
    zero_match_empty_candidates = 0

    true_match_ranks = []

    channel_total_true = Counter()
    channel_recovered = Counter()
    channel_s1_with_true = Counter()
    channel_s1_recovered_any = Counter()

    # --------------------------------------------------------
    # Country-by-country processing.
    # --------------------------------------------------------
    for (
        slug,
        s1_path,
        s2_path,
        s3_path,
    ) in country_files:

        selected_s1 = load_selected_s1(
            s1_path,
            validation_ids,
        ) if not FULL_RUN else None

        if (
            not FULL_RUN
            and (
                selected_s1 is None
                or selected_s1.empty
            )
        ):
            continue

        print("\n" + "=" * 80)
        print(
            f"COUNTRY PARTITION: {slug}"
        )

        if not FULL_RUN:
            print(
                f"S1 selected: "
                f"{len(selected_s1):,}"
            )
        else:
            print("S1 mode: streamed FULL_RUN")

        print(
            f"S2 path: {s2_path}"
        )
        print(
            f"S3 path: {s3_path}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # MEMORY-SAFE candidate index.
        # ----------------------------------------------------
        print(
            "\nBuilding compact candidate index..."
        )

        index = CompactCountryIndex(
            s2_path,
            s3_path,
        )

        print(
            f"✓ Indexed candidate records: "
            f"{len(index.entity_ids):,}"
        )

        print(
            f"  V3 address_number keys: "
            f"{len(index.v3['address_number'].data):,}"
        )

        print(
            f"  IDF informative tokens: "
            f"{len(index.token_posting.data):,}"
        )

        print(
            f"  Sorted-name keys: "
            f"{len(index.sorted_posting.data):,}"
        )

        print(
            f"  Phonetic keys: "
            f"{len(index.phonetic_posting.data):,}"
        )

        # ----------------------------------------------------
        # S1 processing.
        # ----------------------------------------------------
        if FULL_RUN:
            s1_iterator = iter_s1_chunks(
                s1_path,
                None,
            )
        else:
            s1_iterator = [
                selected_s1
            ]

        country_s1_processed = 0

        for s1_chunk in s1_iterator:
            for row in s1_chunk.itertuples(
                index=False,
                name="NormalizedRow",
            ):
                features = extract_features(row)
                sid = features[0]

                if not sid:
                    continue

                ranked, _candidate_info, channel_hits = (
                    generate_and_rank(
                        features,
                        index,
                    )
                )

                candidate_ids = [
                    item[1]
                    for item in ranked
                ]

                candidate_handle.write(
                    sid
                    + "\t"
                    + ",".join(candidate_ids)
                    + "\n"
                )

                selected_s1_count += 1
                country_s1_processed += 1

                total_candidate_pairs += len(
                    candidate_ids
                )
                candidate_counts.append(
                    len(candidate_ids)
                )

                if not FULL_RUN:
                    true_matches = ground_truth.get(
                        sid,
                        set(),
                    )

                    total_true_pairs += len(
                        true_matches
                    )

                    if not true_matches:
                        zero_match_s1_count += 1

                        if not candidate_ids:
                            zero_match_empty_candidates += 1

                    recovered = (
                        set(candidate_ids)
                        & true_matches
                    )

                    recovered_true_pairs += len(
                        recovered
                    )

                    # Channel diagnostics.
                    for channel, hits in channel_hits.items():
                        channel_total_true[channel] += (
                            len(true_matches)
                        )

                        overlap = {
                            index.entity_ids[i]
                            for i in hits
                        } & true_matches

                        channel_recovered[channel] += (
                            len(overlap)
                        )

                        if true_matches:
                            channel_s1_with_true[
                                channel
                            ] += 1

                            if overlap:
                                channel_s1_recovered_any[
                                    channel
                                ] += 1

                    # Rank diagnostics.
                    rank_map = {
                        entity_id: pos + 1
                        for pos, (_, entity_id, _)
                        in enumerate(ranked)
                    }

                    for true_id in true_matches:
                        true_match_ranks.append(
                            rank_map.get(true_id)
                        )

                if (
                    country_s1_processed
                    % PROGRESS_EVERY_CHUNKS
                    == 0
                ):
                    print(
                        f"  S1 processed in {slug}: "
                        f"{country_s1_processed:,}"
                    )

        # Explicitly release the huge country index
        # before loading the next country.
        index.release()
        del index

        print(
            f"✓ Finished country {slug}: "
            f"{country_s1_processed:,} S1 rows"
        )

    candidate_handle.close()

    # --------------------------------------------------------
    # Metrics.
    # --------------------------------------------------------
    finite_ranks = [
        x
        for x in true_match_ranks
        if x is not None
    ]

    metrics = {
        "version": (
            "V4_OPTIMIZED_MEMORY_SAFE"
        ),
        "s1_entities": selected_s1_count,
        "true_pairs": total_true_pairs,
        "recovered_true_pairs": (
            recovered_true_pairs
        ),
        "pairs_completeness": (
            recovered_true_pairs
            / total_true_pairs
            if total_true_pairs
            else None
        ),
        "avg_candidates_per_s1": (
            total_candidate_pairs
            / max(1, selected_s1_count)
        ),
        "median_candidates_per_s1": (
            float(np.median(candidate_counts))
            if candidate_counts
            else 0.0
        ),
        "p95_candidates_per_s1": (
            float(
                np.percentile(
                    candidate_counts,
                    95,
                )
            )
            if candidate_counts
            else 0.0
        ),
        "max_candidates_per_s1": max(
            candidate_counts,
            default=0,
        ),
        "zero_match_s1_count": (
            zero_match_s1_count
        ),
        "zero_match_empty_candidate_rate": (
            zero_match_empty_candidates
            / zero_match_s1_count
            if zero_match_s1_count
            else None
        ),
        "v3_rules_preserved": list(
            V3_RULES
        ),
        "final_cap": FINAL_CAP,
        "minhash_enabled": ENABLE_MINHASH,
        "phonetic_enabled": ENABLE_PHONETIC,
        "embeddings_enabled": ENABLE_EMBEDDINGS,
        "validation_s1_limit": (
            VALIDATION_S1_LIMIT
        ),
        "random_seed": RANDOM_SEED,
        "country_partition": partition_metrics,
        "candidate_file_is_final_current_set": True,
        "ranking_is_compact_channel_score": True,
    }

    if finite_ranks:
        metrics.update(
            {
                "true_match_rank_count": len(
                    finite_ranks
                ),
                "true_match_rank_missing": (
                    len(true_match_ranks)
                    - len(finite_ranks)
                ),
                "true_match_rank_median": float(
                    np.median(finite_ranks)
                ),
                "true_match_rank_p90": float(
                    np.percentile(
                        finite_ranks,
                        90,
                    )
                ),
                "true_match_rank_p95": float(
                    np.percentile(
                        finite_ranks,
                        95,
                    )
                ),
                "true_match_rank_p99": float(
                    np.percentile(
                        finite_ranks,
                        99,
                    )
                ),
                "true_match_rank_max": int(
                    max(finite_ranks)
                ),
            }
        )

    channel_metrics = {}

    for channel in sorted(
        set(channel_total_true)
        | set(channel_recovered)
    ):
        true_pairs = channel_total_true[
            channel
        ]
        recovered = channel_recovered[
            channel
        ]
        s1_with_true = channel_s1_with_true[
            channel
        ]

        channel_metrics[channel] = {
            "s1_with_true_matches": (
                s1_with_true
            ),
            "true_pairs": true_pairs,
            "recovered_true_pairs": recovered,
            "pair_recall": (
                recovered / true_pairs
                if true_pairs
                else None
            ),
            "s1_recall_any_true": (
                channel_s1_recovered_any[
                    channel
                ] / s1_with_true
                if s1_with_true
                else None
            ),
        }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with open(
        CHANNEL_METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            channel_metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Console report.
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("BLOCKING V4 OPTIMIZED VALIDATION")
    print("=" * 90)

    print(
        f"S1 entities        : "
        f"{selected_s1_count:,}"
    )

    print(
        f"True pairs         : "
        f"{total_true_pairs:,}"
    )

    print(
        f"Recovered true     : "
        f"{recovered_true_pairs:,}"
    )

    print(
        f"Pair completeness  : "
        f"{metrics['pairs_completeness']}"
    )

    print(
        f"Average candidates : "
        f"{metrics['avg_candidates_per_s1']:.2f}"
    )

    print(
        f"Median candidates  : "
        f"{metrics['median_candidates_per_s1']:.2f}"
    )

    print(
        f"P95 candidates     : "
        f"{metrics['p95_candidates_per_s1']:.2f}"
    )

    print(
        f"Max candidates     : "
        f"{metrics['max_candidates_per_s1']}"
    )

    if finite_ranks:
        print(
            "\nTrue-match rank diagnostics:"
        )
        print(
            f"  median : "
            f"{metrics['true_match_rank_median']}"
        )
        print(
            f"  P90    : "
            f"{metrics['true_match_rank_p90']}"
        )
        print(
            f"  P95    : "
            f"{metrics['true_match_rank_p95']}"
        )
        print(
            f"  P99    : "
            f"{metrics['true_match_rank_p99']}"
        )
        print(
            f"  max    : "
            f"{metrics['true_match_rank_max']}"
        )

    print("\n" + "=" * 90)
    print("PER-CHANNEL TRUE-PAIR RECALL")
    print("=" * 90)

    for channel, values in channel_metrics.items():
        print(
            f"{channel:32s} "
            f"recovered="
            f"{values['recovered_true_pairs']:,} "
            f"pair_recall="
            f"{values['pair_recall']}"
        )

    print("\n" + "=" * 90)
    print("FINAL CURRENT CANDIDATE SET")
    print("=" * 90)

    print(
        f"Candidate pairs    : "
        f"{total_candidate_pairs:,}"
    )

    print(
        f"Candidate file     : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Metrics file       : "
        f"{METRICS_FILE}"
    )

    print(
        f"Channel metrics    : "
        f"{CHANNEL_METRICS_FILE}"
    )

    print(
        "\nBLOCKING V4 OPTIMIZED COMPLETE"
    )


if __name__ == "__main__":
    main()
