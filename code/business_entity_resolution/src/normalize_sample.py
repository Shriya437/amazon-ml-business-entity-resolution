"""
Stage 0 — Normalization on a country-stratified sample.

Architecture:
    train TSVs -> chunked stratified sampling -> name normalization +
    address feature extraction -> normalized sample TSVs.

Original input TSVs are read-only. Only dataset/processed/ is written.
"""

import random
import re
import unicodedata
from pathlib import Path

import pandas as pd
from indic_transliteration import sanscript


# ============================================================================
# CONFIG
# ============================================================================

DATA_DIR = Path("dataset/train")
OUT_DIR = Path("dataset/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = {
    "source1": DATA_DIR / "train_source1.tsv",
    "source2": DATA_DIR / "train_source2.tsv",
    "source3": DATA_DIR / "train_source3.tsv",
}

SAMPLE_SIZE_PER_SOURCE = 20_000
CHUNK_SIZE = 100_000
RANDOM_STATE = 42

# Keep disabled initially. OCR substitutions such as 0->o can create
# false changes, so this is deliberately conservative.
APPLY_OCR_CORRECTION = False


# ============================================================================
# LEGAL SUFFIXES
# ============================================================================

LEGAL_SUFFIX_MAP = {
    "corporation": "corp",
    "corp": "corp",

    "private": "pvt",
    "pvt": "pvt",

    "limited": "ltd",
    "ltd": "ltd",

    "incorporated": "inc",
    "inc": "inc",

    # IMPORTANT: these remain DISTINCT.
    "llp": "llp",
    "llc": "llc",
    "lp": "lp",

    "plc": "plc",
}

LEGAL_SUFFIX_STRIP_TOKENS = set(
    LEGAL_SUFFIX_MAP
) | {
    "co",
    "public",
    "partners",
    "associates",
    "group",
    "enterprises",
    "services",
    "solutions",
    "holdings",
    "ventures",
    "foundation",
    "trust",
    "society",
    "pc",
}


# ============================================================================
# ADDRESS MAPS
# ============================================================================

ADDRESS_NOISE_WORDS = {
    "unit",
    "floor",
    "flat",
    "no",
    "plot",
    "building",
    "block",
    "sector",
    "road",
    "street",
    "avenue",
    "drive",
    "lane",
    "near",
    "opp",
    "behind",
    "opposite",
    "apt",
    "apartment",
    "suite",
    "ste",
    "room",
    "rm",
}


# ============================================================================
# US STATES
# ============================================================================

US_STATE_MAP = {
    "al": "alabama",
    "ak": "alaska",
    "az": "arizona",
    "ar": "arkansas",
    "ca": "california",
    "co": "colorado",
    "ct": "connecticut",
    "de": "delaware",
    "fl": "florida",
    "ga": "georgia",
    "hi": "hawaii",
    "id": "idaho",
    "il": "illinois",
    "in": "indiana",
    "ia": "iowa",
    "ks": "kansas",
    "ky": "kentucky",
    "la": "louisiana",
    "me": "maine",
    "md": "maryland",
    "ma": "massachusetts",
    "mi": "michigan",
    "mn": "minnesota",
    "ms": "mississippi",
    "mo": "missouri",
    "mt": "montana",
    "ne": "nebraska",
    "nv": "nevada",
    "nh": "new hampshire",
    "nj": "new jersey",
    "nm": "new mexico",
    "ny": "new york",
    "nc": "north carolina",
    "nd": "north dakota",
    "oh": "ohio",
    "ok": "oklahoma",
    "or": "oregon",
    "pa": "pennsylvania",
    "ri": "rhode island",
    "sc": "south carolina",
    "sd": "south dakota",
    "tn": "tennessee",
    "tx": "texas",
    "ut": "utah",
    "vt": "vermont",
    "va": "virginia",
    "wa": "washington",
    "wv": "west virginia",
    "wi": "wisconsin",
}


# ============================================================================
# INDIA STATES / UTs
# ============================================================================

INDIA_STATE_MAP = {
    "ap": "andhra pradesh",
    "ar": "arunachal pradesh",
    "as": "assam",
    "br": "bihar",
    "cg": "chhattisgarh",
    "ga": "goa",
    "gj": "gujarat",
    "hr": "haryana",
    "hp": "himachal pradesh",
    "jh": "jharkhand",
    "ka": "karnataka",
    "kl": "kerala",
    "mp": "madhya pradesh",
    "mh": "maharashtra",
    "mn": "manipur",
    "ml": "meghalaya",
    "mz": "mizoram",
    "nl": "nagaland",
    "od": "odisha",
    "or": "odisha",
    "pb": "punjab",
    "rj": "rajasthan",
    "sk": "sikkim",
    "tn": "tamil nadu",
    "ts": "telangana",
    "tg": "telangana",
    "tr": "tripura",
    "up": "uttar pradesh",
    "uk": "uttarakhand",
    "ut": "uttarakhand",
    "wb": "west bengal",

    # Union Territories
    "an": "andaman and nicobar islands",
    "ch": "chandigarh",
    "dl": "delhi",
    "jk": "jammu and kashmir",
    "la": "ladakh",
    "ld": "lakshadweep",
    "py": "puducherry",
    "dn": "dadra and nagar haveli and daman and diu",
    "dd": "dadra and nagar haveli and daman and diu",
}

US_STATE_FULL_SET = set(US_STATE_MAP.values())
# Native-script state / UT aliases.
# These are used for state recognition BEFORE transliteration, so an
# Indic-script state name is matched exactly instead of relying on a
# phonetic transliteration such as "amdhrapradesh".
INDIA_NATIVE_STATE_MAP = {
    # Andhra Pradesh
    "ఆంధ్రప్రదేశ్": "andhra pradesh",
    "आंध्र प्रदेश": "andhra pradesh",

    # Arunachal Pradesh
    "अरुणाचल प्रदेश": "arunachal pradesh",

    # Assam
    "অসম": "assam",
    "অসমীয়া": "assam",

    # Bihar
    "बिहार": "bihar",

    # Chhattisgarh
    "छत्तीसगढ़": "chhattisgarh",

    # Goa
    "गोवा": "goa",

    # Gujarat
    "ગુજરાત": "gujarat",
    "ગુજરાત રાજ્ય": "gujarat",

    # Haryana
    "हरियाणा": "haryana",
    "ਹਰਿਆਣਾ": "haryana",

    # Himachal Pradesh
    "हिमाचल प्रदेश": "himachal pradesh",

    # Jharkhand
    "झारखंड": "jharkhand",

    # Karnataka
    "ಕರ್ನಾಟಕ": "karnataka",
    "ಕರ್ನಾಟಕ ರಾಜ್ಯ": "karnataka",

    # Kerala
    "കേരളം": "kerala",
    "കേരള": "kerala",

    # Madhya Pradesh
    "मध्य प्रदेश": "madhya pradesh",

    # Maharashtra
    "महाराष्ट्र": "maharashtra",
    "महाराष्ट्र राज्य": "maharashtra",

    # Manipur
    "मणिपुर": "manipur",

    # Meghalaya
    "मेघालय": "meghalaya",

    # Mizoram
    "मिजोरम": "mizoram",

    # Nagaland
    "नागालैंड": "nagaland",

    # Odisha
    "ओडिशा": "odisha",
    "ଓଡ଼ିଶା": "odisha",
    "ଓଡିଶା": "odisha",

    # Punjab
    "ਪੰਜਾਬ": "punjab",
    "पंजाब": "punjab",

    # Rajasthan
    "राजस्थान": "rajasthan",

    # Sikkim
    "सिक्किम": "sikkim",

    # Tamil Nadu
    "தமிழ்நாடு": "tamil nadu",
    "தமிழ்நாடு மாநிலம்": "tamil nadu",

    # Telangana
    "తెలంగాణ": "telangana",
    "తెలంగాణ రాష్ట్రం": "telangana",

    # Tripura
    "त्रिपुरा": "tripura",

    # Uttar Pradesh
    "उत्तर प्रदेश": "uttar pradesh",

    # Uttarakhand
    "उत्तराखंड": "uttarakhand",
    "उत्तराखण्ड": "uttarakhand",

    # West Bengal
    "পশ্চিমবঙ্গ": "west bengal",
    "পশ্চিম বঙ্গ": "west bengal",
    "पश्चिम बंगाल": "west bengal",

    # Union Territories
    "अंडमान और निकोबार द्वीपसमूह": "andaman and nicobar islands",
    "ਚੰਡੀਗੜ੍ਹ": "chandigarh",
    "चंडीगढ़": "chandigarh",
    "दिल्ली": "delhi",
    "दिल्ली राज्य": "delhi",
    "जम्मू और कश्मीर": "jammu and kashmir",
    "जम्मू कश्मीर": "jammu and kashmir",
    "लद्दाख": "ladakh",
    "लक्षद्वीप": "lakshadweep",
    "पुदुच्चेरी": "puducherry",
    "पुडुचेरी": "puducherry",
    "पुदुचेरी": "puducherry",
    "દાદરા અને નગર હવેલી અને દમણ અને દીવ": "dadra and nagar haveli and daman and diu",
}

# Canonicalized native aliases as token tuples.  Built once so state
# recognition remains simple and deterministic during row processing.
INDIA_NATIVE_STATE_TOKEN_MAP = {
    tuple("".join(c for c in unicodedata.normalize("NFKD", part.lower()) if not unicodedata.combining(c)) for part in alias.split()): canonical
    for alias, canonical in INDIA_NATIVE_STATE_MAP.items()
}

INDIA_STATE_FULL_SET = set(INDIA_STATE_MAP.values())


# ============================================================================
# TEXT / SCRIPT SETTINGS
# ============================================================================

DOMAIN_SUFFIXES = (
    ".co.in",
    ".com",
    ".net",
    ".org",
    ".in",
)

MULTI_SPACE = re.compile(r"\s+")

SCRIPT_RANGES = {
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "devanagari": (0x0900, 0x097F),
    "oriya": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}

TRANSLITERATION_SCHEMES = {
    "bengali": sanscript.BENGALI,
    "gurmukhi": sanscript.GURMUKHI,
    "gujarati": sanscript.GUJARATI,
    "devanagari": sanscript.DEVANAGARI,
    "oriya": sanscript.ORIYA,
    "tamil": sanscript.TAMIL,
    "telugu": sanscript.TELUGU,
    "kannada": sanscript.KANNADA,
    "malayalam": sanscript.MALAYALAM,
}


# ============================================================================
# OCR
# ============================================================================

OCR_DIGIT_MAP = str.maketrans({
    "6": "g",
    "0": "o",
    "5": "s",
    "1": "l",
})


# ============================================================================
# TEXT HELPERS
# ============================================================================

def unicode_safe_punctuation_to_spaces(text: str) -> str:
    """
    Replace punctuation/symbols with spaces while preserving:
        - Unicode letters
        - Unicode digits
        - Unicode combining marks
    """

    output = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith(("P", "S")):
            output.append(" ")
        else:
            output.append(char)

    return "".join(output)


def fold_for_lookup(text: str) -> str:
    """
    Remove combining marks ONLY for lookup/folding.

    Do NOT use this to create the primary normalized representation
    of an Indic name.
    """

    decomposed = unicodedata.normalize("NFKD", text)

    return "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char)
    )


def canonicalize_dotted_legal_terms(text: str) -> str:

    replacements = [

        (
            r"\bl\s*[\.\s]+\s*l\s*[\.\s]+\s*p\.?\b",
            "llp",
        ),

        (
            r"\bl\s*[\.\s]+\s*l\s*[\.\s]+\s*c\.?\b",
            "llc",
        ),

        (
            r"\bl\s*[\.\s]+\s*p\.?\b",
            "lp",
        ),

        (
            r"\bp\s*[\.\s]+\s*v\s*[\.\s]+\s*t\.?\b",
            "pvt",
        ),

        (
            r"\bpvt\.?\s+ltd\.?\b",
            "private limited",
        ),

        (
            r"\bprivate\s+limited\b",
            "private limited",
        ),

        (
            r"\bltd\.?\b",
            "ltd",
        ),

        (
            r"\bcorp\.?\b",
            "corp",
        ),

        (
            r"\binc\.?\b",
            "inc",
        ),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text


def strip_domain_suffix(name: str) -> tuple[str, bool]:

    lowered = name.lower()

    for suffix in DOMAIN_SUFFIXES:

        if lowered.endswith(suffix):

            return (
                name[:-len(suffix)],
                True,
            )

    if any(
        suffix in lowered
        for suffix in (
            ".com",
            ".net",
            ".org",
        )
    ):

        cleaned = re.sub(
            r"\.(com|net|org|in|co\.in)\b",
            "",
            name,
            flags=re.IGNORECASE,
        )

        return cleaned, True

    return name, False


def clean_text(text: str) -> str:

    text = canonicalize_dotted_legal_terms(text)

    text = unicode_safe_punctuation_to_spaces(text)

    text = text.lower()

    return MULTI_SPACE.sub(
        " ",
        text,
    ).strip()


def detect_script(text: str) -> str:

    if (
        not isinstance(text, str)
        or not text
    ):
        return "unknown"

    counts = {
        script: 0
        for script in SCRIPT_RANGES
    }

    latin_count = 0

    for char in text:

        if not char.isalpha():
            continue

        code_point = ord(char)
        matched = False

        for (
            script,
            (low, high),
        ) in SCRIPT_RANGES.items():

            if low <= code_point <= high:

                counts[script] += 1
                matched = True
                break

        if not matched:
            latin_count += 1

    indic_count = sum(counts.values())

    if indic_count == 0:
        return "latin"

    if latin_count > 0:
        return "mixed"

    return max(
        counts,
        key=counts.get,
    )


def canonicalize_legal_suffixes(
    tokens: list[str],
) -> list[str]:

    return [
        LEGAL_SUFFIX_MAP.get(
            fold_for_lookup(token),
            token,
        )
        for token in tokens
    ]


def strip_legal_suffix_tokens(
    tokens: list[str],
) -> list[str]:

    return [
        token
        for token in tokens
        if fold_for_lookup(token)
        not in LEGAL_SUFFIX_STRIP_TOKENS
    ]


def correct_ocr_digits(
    tokens: list[str],
) -> list[str]:

    output = []

    for token in tokens:

        has_letter = any(
            c.isalpha()
            for c in token
        )

        has_digit = any(
            c.isdigit()
            for c in token
        )

        if has_letter and has_digit:

            output.append(
                token.translate(
                    OCR_DIGIT_MAP
                )
            )

        else:

            output.append(token)

    return output


# ============================================================================
# NAME NORMALIZATION
# ============================================================================

def normalize_name_base(
    raw_name: str,
) -> tuple[str, bool, str]:

    if (
        not isinstance(raw_name, str)
        or not raw_name.strip()
    ):

        return (
            "",
            False,
            "unknown",
        )

    script_type = detect_script(raw_name)

    cleaned, is_domain = (
        strip_domain_suffix(raw_name)
    )

    cleaned = clean_text(cleaned)

    tokens = []

    for token in cleaned.split():

        if (
            not tokens
            or tokens[-1] != token
        ):

            tokens.append(token)

    if APPLY_OCR_CORRECTION:

        tokens = correct_ocr_digits(tokens)

    tokens = canonicalize_legal_suffixes(tokens)

    return (
        " ".join(tokens),
        is_domain,
        script_type,
    )


def transliterate_name(
    raw_name: str,
    script_type: str,
) -> str:

    if (
        not isinstance(raw_name, str)
        or not raw_name.strip()
    ):
        return ""

    source_scheme = (
        TRANSLITERATION_SCHEMES.get(
            script_type
        )
    )

    if source_scheme is None:
        return ""

    try:

        transliterated = (
            sanscript.transliterate(
                raw_name,
                source_scheme,
                sanscript.ITRANS,
                maybe_use_dravidian_variant="yes",
            )
        )

        transliterated = (
            canonicalize_dotted_legal_terms(
                transliterated
            )
        )

        transliterated = (
            unicode_safe_punctuation_to_spaces(
                transliterated
            )
        )

        transliterated = fold_for_lookup(
            transliterated
        )

        transliterated = (
            transliterated.lower()
        )

        transliterated = (
            MULTI_SPACE.sub(
                " ",
                transliterated,
            ).strip()
        )

        return transliterated

    except Exception:
        return ""


def transliterate_address(
    raw_address: str,
) -> str:
    """
    Create a lookup-friendly transliterated representation of an address.

    Important:
        - Latin text is preserved.
        - Indic-script portions are transliterated to ITRANS.
        - Mixed-script addresses are handled token/run by token/run.
        - The original business_address is never modified.
        - This is an ADDITIONAL representation for blocking, not a replacement.
    """

    if (
        not isinstance(raw_address, str)
        or not raw_address.strip()
        or raw_address.strip().lower() in NULL_ADDRESS_STRINGS
    ):
        return ""

    output = []
    current_script = None
    current_text = []

    def flush_run():
        nonlocal current_script, current_text

        if not current_text:
            return

        run = "".join(current_text)

        if current_script in TRANSLITERATION_SCHEMES:
            try:
                # Reuse the same transliteration path already used and
                # verified for business names.
                transliterated_run = transliterate_name(
                    run,
                    current_script,
                )
                if transliterated_run:
                    run = transliterated_run
            except Exception:
                # Preserve the original run if transliteration fails.
                pass

        output.append(run)
        current_text = []
        current_script = None

    for char in raw_address:
        code_point = ord(char)
        char_script = None

        for script, (low, high) in SCRIPT_RANGES.items():
            if low <= code_point <= high:
                char_script = script
                break

        # Keep Latin letters, digits, whitespace and punctuation together.
        # Only Indic script runs need transliteration.
        if char_script != current_script:
            flush_run()
            current_script = char_script

        current_text.append(char)

    flush_run()

    transliterated = "".join(output)
    transliterated = unicode_safe_punctuation_to_spaces(
        transliterated
    )
    transliterated = fold_for_lookup(transliterated)
    transliterated = transliterated.lower()

    return MULTI_SPACE.sub(
        " ",
        transliterated,
    ).strip()


def build_name_variants(
    raw_name: str,
) -> dict:

    (
        name_full,
        is_domain_name,
        script_type,
    ) = normalize_name_base(raw_name)

    tokens = name_full.split()

    no_suffix_tokens = (
        strip_legal_suffix_tokens(
            tokens
        )
    )

    transliterated = (
        transliterate_name(
            raw_name,
            script_type,
        )
    )

    transliterated_no_suffix = (
        strip_legal_suffix_tokens(
            transliterated.split()
        )
    )

    # name_ascii is only created for Latin names.
    if script_type == "latin":

        name_ascii = fold_for_lookup(
            name_full
        )

    else:

        name_ascii = ""

    return {

        "name_full": name_full,

        "name_no_suffix":
            " ".join(
                no_suffix_tokens
            ),

        "name_sorted":
            " ".join(
                sorted(
                    no_suffix_tokens
                )
            ),

        "name_first2":
            " ".join(
                no_suffix_tokens[:2]
            ),

        "name_ascii":
            name_ascii,

        "name_transliterated":
            transliterated,

        "name_transliterated_no_suffix":
            " ".join(
                transliterated_no_suffix
            ),

        "is_domain_name":
            is_domain_name,

        "script_type":
            script_type,
    }


# ============================================================================
# ADDRESS NORMALIZATION / EXTRACTION
# ============================================================================

NULL_ADDRESS_STRINGS = {
    "",
    "nan",
    "null",
    "none",
    "n/a",
    "na",
}


def normalize_country(
    country,
) -> str:

    if not isinstance(
        country,
        str,
    ):
        return ""

    return country.strip().lower()


def tokenize_address(
    raw_address: str,
) -> list[str]:

    cleaned = raw_address.lower()

    cleaned = re.sub(
        r"[,#;:]+",
        " ",
        cleaned,
    )

    cleaned = MULTI_SPACE.sub(
        " ",
        cleaned,
    ).strip()

    return [
        token.strip(".,")
        for token in cleaned.split()
        if token.strip(".,")
    ]


def state_maps_for_country(
    country: str,
):

    country = normalize_country(country)

    if country == "us":

        return (
            US_STATE_MAP,
            US_STATE_FULL_SET,
        )

    if country == "india":

        return (
            INDIA_STATE_MAP,
            INDIA_STATE_FULL_SET,
        )

    return (
        {},
        US_STATE_FULL_SET
        | INDIA_STATE_FULL_SET,
    )


def _normalized_tokens(tokens: list[str]) -> list[str]:
    """Normalize tokens for deterministic state lookup."""
    return [
        fold_for_lookup(
            token.lower()
        )
        for token in tokens
    ]


def _find_native_india_state(
    tokens: list[str],
) -> tuple[str, int]:
    """
    Find an exact native-script Indian state/UT name.

    Returns:
        (canonical_state, start_position)
    """
    normalized = _normalized_tokens(tokens)

    # Prefer the longest native state alias so multi-word names
    # such as "west bengal" are handled correctly.
    aliases = sorted(
        INDIA_NATIVE_STATE_TOKEN_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias_tokens, canonical in aliases:

        width = len(alias_tokens)

        if width == 0 or width > len(normalized):
            continue

        for i in range(
            len(normalized) - width + 1
        ):

            if tuple(
                normalized[
                    i:i + width
                ]
            ) == alias_tokens:

                return canonical, i

    return "", -1


def _abbreviation_allowed_position(
    index: int,
    token_count: int,
) -> bool:
    """
    State abbreviations are short and ambiguous.

    Examples:
        HP in "Opp. HP Petrol Bunk" is NOT a state.
        MH in "MH, Mumbai" IS a state.
        UP in "... Kushinagar, UP" IS a state.

    Therefore abbreviations are accepted only at the beginning
    or in the final three tokens of an address.
    """
    return (
        index == 0
        or index >= max(0, token_count - 3)
    )


def normalize_state_token(
    tokens: list[str],
    country: str,
) -> str:

    country = normalize_country(country)

    # ------------------------------------------------------------------------
    # INDIA: exact native-script state/UT matching first.
    # ------------------------------------------------------------------------
    if country == "india":

        native_state, _ = _find_native_india_state(tokens)

        if native_state:
            return native_state

    (
        abbreviation_map,
        full_states,
    ) = state_maps_for_country(country)

    normalized = _normalized_tokens(tokens)

    # ------------------------------------------------------------------------
    # Full state names can occur anywhere in the address.
    # ------------------------------------------------------------------------

    # Two-word state names first.
    for i in range(
        len(normalized) - 2,
        -1,
        -1,
    ):

        candidate = (
            normalized[i]
            + " "
            + normalized[i + 1]
        )

        if candidate in full_states:
            return candidate

    # One-word full state names.
    for token in reversed(normalized):

        if token in full_states:
            return token

    # ------------------------------------------------------------------------
    # Abbreviations are checked separately because they are ambiguous.
    # ------------------------------------------------------------------------

    for i in range(
        len(normalized) - 1,
        -1,
        -1,
    ):

        token = normalized[i]

        if (
            token in abbreviation_map
            and _abbreviation_allowed_position(
                i,
                len(normalized),
            )
        ):

            return abbreviation_map[token]

    return ""


def find_state_position(
    tokens: list[str],
    country: str,
) -> int:

    country = normalize_country(country)

    # ------------------------------------------------------------------------
    # INDIA: exact native-script state/UT matching first.
    # ------------------------------------------------------------------------
    if country == "india":

        _, native_position = _find_native_india_state(tokens)

        if native_position >= 0:
            return native_position

    (
        abbreviation_map,
        full_states,
    ) = state_maps_for_country(country)

    normalized = _normalized_tokens(tokens)

    # Two-word full state names.
    for i in range(
        len(normalized) - 2,
        -1,
        -1,
    ):

        candidate = (
            normalized[i]
            + " "
            + normalized[i + 1]
        )

        if candidate in full_states:
            return i

    # One-word full state names.
    for i in range(
        len(normalized) - 1,
        -1,
        -1,
    ):

        if normalized[i] in full_states:
            return i

    # Abbreviations only at safe positions.
    for i in range(
        len(normalized) - 1,
        -1,
        -1,
    ):

        if (
            normalized[i] in abbreviation_map
            and _abbreviation_allowed_position(
                i,
                len(normalized),
            )
        ):

            return i

    return -1

def extract_postal_code(
    tokens: list[str],
    country: str,
    state_position: int,
) -> tuple[str, str]:
    """
    Conservative postal-code extraction.

    IMPORTANT:
    For US addresses, a 5-digit number at the beginning of the
    address is NOT automatically treated as a ZIP code.

    Example:
        14399 540 Road, Oklahoma, Claremore

    14399 is likely a street/house number, not a ZIP.

    A ZIP is preferentially taken from the region immediately
    surrounding the state, especially after the state token.
    """

    country = normalize_country(country)

    if country == "india":

        pattern = re.compile(
            r"^[1-9]\d{5}$"
        )

    elif country == "us":

        pattern = re.compile(
            r"^\d{5}(?:-\d{4})?$"
        )

    else:

        return "", ""

    # ------------------------------------------------------------------------
    # INDIA
    # ------------------------------------------------------------------------

    if country == "india":

        candidate_indices = []

        if state_position >= 0:

            # Indian PIN codes can occur before or after state.
            candidate_indices.extend(
                range(
                    state_position + 1,
                    len(tokens),
                )
            )

            candidate_indices.extend(
                range(
                    state_position - 1,
                    -1,
                    -1,
                )
            )

        else:

            candidate_indices.extend(
                range(
                    len(tokens) - 1,
                    -1,
                    -1,
                )
            )

        seen = set()

        for index in candidate_indices:

            if index in seen:
                continue

            seen.add(index)

            token = tokens[index]

            if pattern.fullmatch(token):

                return token, ""

        return "", ""

    # ------------------------------------------------------------------------
    # UNITED STATES
    # ------------------------------------------------------------------------

    if country == "us":

        # Best case:
        # ZIP immediately AFTER the state.
        #
        # Example:
        #   Phoenix, AZ 85001
        #
        # This is the strongest positional signal.
        if state_position >= 0:

            for index in range(
                state_position + 1,
                len(tokens),
            ):

                token = tokens[index]

                if pattern.fullmatch(token):

                    return "", token

        # Second possibility:
        # ZIP immediately BEFORE the state.
        #
        # Example:
        #   Phoenix, 85001, AZ
        #
        # We only inspect a small local window, rather than searching
        # the entire address for any 5-digit number.
        if state_position >= 0:

            start = max(
                0,
                state_position - 3,
            )

            for index in range(
                state_position - 1,
                start - 1,
                -1,
            ):

                token = tokens[index]

                if pattern.fullmatch(token):

                    # Do not interpret the first token as ZIP.
                    if index == 0:
                        continue

                    return "", token

        # If no state is available, use a conservative fallback:
        # only accept a ZIP near the END of the address.
        #
        # This avoids turning an initial house number such as
        # 14399 into a ZIP.
        if state_position < 0:

            for index in range(
                len(tokens) - 1,
                max(-1, len(tokens) - 4),
                -1,
            ):

                token = tokens[index]

                if pattern.fullmatch(token):

                    # Never accept the first token as a ZIP.
                    if index == 0:
                        continue

                    return "", token

        return "", ""

    return "", ""


def extract_address_fields(
    raw_address,
    country,
) -> dict:

    if (
        not isinstance(
            raw_address,
            str,
        )
        or raw_address.strip().lower()
        in NULL_ADDRESS_STRINGS
    ):

        return {
            "has_address": False,
            "pin_code": "",
            "zip_code": "",
            "street_num": "",
            "state_token": "",
            "locality_tokens": "",
            "address_transliterated": "",
        }

    tokens = tokenize_address(
        raw_address
    )

    # Additional lookup representation for addresses written in
    # Indic scripts. The original address and original tokens remain
    # unchanged and are still used for the existing extraction logic.
    address_transliterated = transliterate_address(
        raw_address
    )

    transliterated_tokens = tokenize_address(
        address_transliterated
    )

    # First use the existing state matching on the original address.
    # If the state is written in an Indic script, fall back to the
    # transliterated representation.
    state_token = (
        normalize_state_token(
            tokens,
            country,
        )
    )

    state_position = (
        find_state_position(
            tokens,
            country,
        )
    )

    # Keep the original-token extraction as the primary path.
    postal_tokens = tokens
    postal_state_position = state_position

    # If the state is written in an Indic script, fall back to the
    # transliterated address for state recognition and postal context.
    if (
        not state_token
        and transliterated_tokens
    ):
        transliterated_state_token = (
            normalize_state_token(
                transliterated_tokens,
                country,
            )
        )

        if transliterated_state_token:
            state_token = transliterated_state_token
            postal_tokens = transliterated_tokens
            postal_state_position = (
                find_state_position(
                    transliterated_tokens,
                    country,
                )
            )

    (
        pin_code,
        zip_code,
    ) = extract_postal_code(
        postal_tokens,
        country,
        postal_state_position,
    )

    # If the original-script representation did not expose the
    # postal code cleanly, use the transliterated representation
    # as a fallback. Digits are preserved by transliteration.
    if (
        not pin_code
        and not zip_code
        and transliterated_tokens
        and postal_tokens is tokens
    ):
        transliterated_state_position = (
            find_state_position(
                transliterated_tokens,
                country,
            )
        )

        pin_code, zip_code = extract_postal_code(
            transliterated_tokens,
            country,
            transliterated_state_position,
        )

    postal_values = {
        pin_code,
        zip_code,
    }

    remaining = [
        token
        for token in tokens
        if token not in postal_values
    ]

    # ------------------------------------------------------------------------
    # Street / house number
    # ------------------------------------------------------------------------

    street_number_pattern = re.compile(
        r"^(?:[a-z]+[-/]?)?\d+"
        r"(?:[-/]\d+)*"
        r"(?:[-/][a-z]+)?"
        r"(?:[a-z])?$",
        flags=re.IGNORECASE,
    )

    number_labels = {
        "no",
        "number",
        "plot",
        "flat",
        "unit",
        "suite",
        "ste",
        "apt",
        "apartment",
        "door",
    }

    street_num = ""

    for i, token in enumerate(
        remaining[:12]
    ):

        candidate = token.strip(
            ".,;"
        )

        lower = candidate.lower()

        if lower in number_labels:
            continue

        if street_number_pattern.fullmatch(
            lower
        ):

            street_num = candidate
            break

        # Examples:
        # NO 25
        # PLOT 00358
        # DOOR 787
        if (
            i > 0
            and remaining[i - 1].lower()
            in number_labels
            and re.search(
                r"\d",
                candidate,
            )
        ):

            street_num = candidate
            break

    # ------------------------------------------------------------------------
    # Locality tokens
    # ------------------------------------------------------------------------

    country_norm = (
        normalize_country(
            country
        )
    )

    if country_norm == "us":

        full_states = US_STATE_FULL_SET
        abbreviations = set(
            US_STATE_MAP
        )

    elif country_norm == "india":

        full_states = INDIA_STATE_FULL_SET
        abbreviations = set(
            INDIA_STATE_MAP
        )

    else:

        full_states = (
            US_STATE_FULL_SET
            | INDIA_STATE_FULL_SET
        )

        abbreviations = set()

    state_words = set()

    for state in full_states:

        state_words.update(
            state.split()
        )

    locality = []

    # Exclude the native-script state/UT tokens from locality output as well.
    native_state_indices = set()

    if country_norm == "india":
        _, native_start = _find_native_india_state(tokens)

        if native_start >= 0:
            native_state_tokens, _ = _find_native_india_state(tokens)

            native_alias_tokens = None

            normalized_tokens = _normalized_tokens(tokens)

            for alias_tokens, canonical in sorted(
                INDIA_NATIVE_STATE_TOKEN_MAP.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            ):
                width = len(alias_tokens)

                if (
                    width > 0
                    and native_start + width <= len(normalized_tokens)
                    and tuple(
                        normalized_tokens[
                            native_start:native_start + width
                        ]
                    ) == alias_tokens
                    and canonical == native_state_tokens
                ):
                    native_alias_tokens = alias_tokens
                    for j in range(
                        native_start,
                        native_start + width,
                    ):
                        native_state_indices.add(j)
                    break

    # Map remaining tokens back to their original token positions.
    # This lets us exclude native-script state tokens precisely.
    remaining_with_positions = [
        (index, token)
        for index, token in enumerate(tokens)
        if token not in postal_values
    ]

    for original_index, token in remaining_with_positions:

        if original_index in native_state_indices:
            continue

        lookup = fold_for_lookup(
            token.lower()
        )

        if not lookup:
            continue

        if lookup in ADDRESS_NOISE_WORDS:
            continue

        if lookup in abbreviations:
            continue

        if lookup in state_words:
            continue

        if (
            street_num
            and token == street_num
        ):
            continue

        locality.append(token)

    return {
        "has_address": True,

        "pin_code":
            pin_code,

        "zip_code":
            zip_code,

        "street_num":
            street_num,

        "state_token":
            state_token,

        "locality_tokens":
            clean_text(
                " ".join(
                    locality
                )
            ),

        # Additional address representation for blocking. The
        # original address remains untouched.
        "address_transliterated":
            address_transliterated,
    }


# ============================================================================
# CHUNKED COUNTRY-STRATIFIED SAMPLING
# ============================================================================

def get_country_counts(
    path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> dict:

    counts_total = {}

    print(
        "  Counting countries..."
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        usecols=["country"],
        dtype={"country": "string"},
        chunksize=chunk_size,
    ):

        counts = (
            chunk["country"]
            .value_counts(
                dropna=False
            )
        )

        for country, count in counts.items():

            key = (
                "__MISSING__"
                if pd.isna(country)
                else str(country)
            )

            counts_total[key] = (
                counts_total.get(
                    key,
                    0,
                )
                + int(count)
            )

    return counts_total


def calculate_country_targets(
    country_counts: dict,
    sample_size: int,
) -> dict:

    total = sum(
        country_counts.values()
    )

    if total == 0:
        return {}

    sample_size = min(
        sample_size,
        total,
    )

    raw = {
        country:
            count / total * sample_size
        for country, count
        in country_counts.items()
    }

    targets = {
        country: int(value)
        for country, value
        in raw.items()
    }

    remaining = (
        sample_size
        - sum(targets.values())
    )

    order = sorted(
        country_counts,
        key=lambda c:
            raw[c] - targets[c],
        reverse=True,
    )

    for country in order[:remaining]:

        targets[country] += 1

    return {
        country: target
        for country, target
        in targets.items()
        if target > 0
    }


def stratified_sample_from_tsv(
    path: Path,
    sample_size: int,
    chunk_size: int = CHUNK_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:

    print(
        "  Pass 1/2: counting country distribution..."
    )

    country_counts = (
        get_country_counts(
            path,
            chunk_size,
        )
    )

    print(
        "  Country counts:",
        country_counts,
    )

    targets = (
        calculate_country_targets(
            country_counts,
            sample_size,
        )
    )

    print(
        "  Target sample per country:",
        targets,
    )

    reservoirs = {
        country: []
        for country in targets
    }

    seen = {
        country: 0
        for country in targets
    }

    rngs = {
        country:
            random.Random(
                random_state + i
            )
        for i, country
        in enumerate(targets)
    }

    print(
        "  Pass 2/2: performing "
        "country-stratified reservoir sampling..."
    )

    for (
        chunk_number,
        chunk,
    ) in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=chunk_size,
        )
    ):

        country_index = (
            chunk.columns.get_loc(
                "country"
            )
        )

        for row in chunk.itertuples(
            index=False,
            name=None,
        ):

            country = row[country_index]

            if pd.isna(country):

                country = "__MISSING__"

            else:

                country = str(country)

            if country not in targets:
                continue

            seen[country] += 1

            target = targets[country]

            if (
                len(
                    reservoirs[country]
                )
                < target
            ):

                reservoirs[country].append(
                    row
                )

            else:

                j = rngs[country].randint(
                    1,
                    seen[country],
                )

                if j <= target:

                    reservoirs[country][
                        j - 1
                    ] = row

        if chunk_number % 10 == 0:

            print(
                f"    processed chunk "
                f"{chunk_number:,}"
            )

    columns = (
        pd.read_csv(
            path,
            sep="\t",
            nrows=0,
        )
        .columns
        .tolist()
    )

    parts = [
        pd.DataFrame(
            reservoirs[country],
            columns=columns,
        )
        for country in targets
        if reservoirs[country]
    ]

    if not parts:

        return pd.DataFrame(
            columns=columns
        )

    sampled = pd.concat(
        parts,
        ignore_index=True,
    )

    return (
        sampled.sample(
            frac=1,
            random_state=random_state,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================================
# DATAFRAME NORMALIZATION
# ============================================================================

def normalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "  Normalizing names..."
    )

    names = pd.DataFrame(
        list(
            df["business_name"].apply(
                build_name_variants
            )
        )
    )

    print(
        "  Normalizing addresses..."
    )

    addresses = pd.DataFrame(
        list(
            df.apply(
                lambda row:
                    extract_address_fields(
                        row["business_address"],
                        row["country"],
                    ),
                axis=1,
            )
        )
    )

    return pd.concat(
        [
            df.reset_index(drop=True),
            names.reset_index(drop=True),
            addresses.reset_index(drop=True),
        ],
        axis=1,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STAGE 0 — NORMALIZATION SAMPLE"
    )

    print(
        "=" * 70
    )

    for (
        source_name,
        path,
    ) in SOURCE_FILES.items():

        if not path.exists():

            print(
                f"\n[SKIP] File not found: "
                f"{path}"
            )

            continue

        print(
            "\n" + "-" * 70
        )

        print(
            f"[{source_name}] {path}"
        )

        print(
            f"  Requested sample: "
            f"{SAMPLE_SIZE_PER_SOURCE:,} rows"
        )

        sample = (
            stratified_sample_from_tsv(
                path=path,
                sample_size=
                    SAMPLE_SIZE_PER_SOURCE,
                chunk_size=
                    CHUNK_SIZE,
                random_state=
                    RANDOM_STATE,
            )
        )

        print(
            f"  Sample obtained: "
            f"{len(sample):,} rows"
        )

        if sample.empty:

            print(
                "  No data sampled. "
                "Skipping."
            )

            continue

        print(
            "  Sample country distribution:"
        )

        print(
            sample["country"].value_counts(
                dropna=False
            )
        )

        normalized = (
            normalize_dataframe(
                sample
            )
        )

        output_path = (
            OUT_DIR
            / f"sample_{source_name}_normalized.tsv"
        )

        normalized.to_csv(
            output_path,
            sep="\t",
            index=False,
        )

        print(
            f"  Wrote "
            f"{len(normalized):,} rows "
            f"-> {output_path}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        "\nUseful name columns:"
    )

    print(
        "name_full, name_no_suffix, "
        "name_sorted, name_first2, "
        "name_ascii, name_transliterated, "
        "name_transliterated_no_suffix, "
        "is_domain_name, script_type"
    )

    print(
        "\nUseful address columns:"
    )

    print(
        "has_address, pin_code, zip_code, "
        "street_num, state_token, "
        "locality_tokens, address_transliterated"
    )


if __name__ == "__main__":
    main()