# ============================================================
# STAGE 1: COUNTRY PARTITIONING ONLY
# Amazon ML Challenge — Business Entity Resolution
#
# IMPORTANT:
#   - Stage 0 normalization is ALREADY COMPLETE.
#   - This script DOES NOT normalize anything.
#   - It reads the existing Stage-0 normalized TSV files.
#   - It partitions them by country in memory-safe chunks.
#
# INPUT:
#   processed_data/stage0/train_source1.tsv
#   processed_data/stage0/train_source2.tsv
#   processed_data/stage0/train_source3.tsv
#
# OUTPUT:
#   processed_data/stage1/train/<country>_s1.tsv
#   processed_data/stage1/train/<country>_s2.tsv
#   processed_data/stage1/train/<country>_s3.tsv
#
#   processed_data/meta/partition_summary.json
#
# The country-partitioning logic matches the merged pipeline:
#   - open-set countries
#   - no hard-coded US/India
#   - France/future countries handled automatically
#   - missing country -> __unknown__
# ============================================================


import os
import re
import json
import shutil
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 50_000

# IMPORTANT:
# These paths are relative to the STUDENT_RESOURCE ROOT.
# Run this script from student_resource.
OUTPUT_DIR = "processed_data"

STAGE0_DIR = os.path.join(
    OUTPUT_DIR,
    "stage0"
)

STAGE1_TRAIN_DIR = os.path.join(
    OUTPUT_DIR,
    "stage1",
    "train"
)

META_DIR = os.path.join(
    OUTPUT_DIR,
    "meta"
)


# ============================================================
# COUNTRY PARTITIONING
# EXACT LOGIC FROM THE MERGED PIPELINE
# ============================================================

def partition_by_country(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
) -> dict:
    """
    Partition S1, S2 and S3 by country.

    This matches the original merged normalization pipeline.

    Properties:
      - Open-set: no hard-coded countries
      - Automatically handles France and future countries
      - Missing/empty country -> __unknown__
      - Preserves every row exactly once
    """

    all_countries = (
        set(s1["country"].unique())
        | set(s2["country"].unique())
        | set(s3["country"].unique())
    )

    # Empty country handled separately as __unknown__
    all_countries.discard("")

    print(
        f"    Found {len(all_countries)} country partitions: "
        f"{sorted(all_countries)}"
    )

    partitions = {}

    for country in sorted(all_countries):

        partitions[country] = {
            "s1": s1[
                s1["country"] == country
            ].reset_index(drop=True),

            "s2": s2[
                s2["country"] == country
            ].reset_index(drop=True),

            "s3": s3[
                s3["country"] == country
            ].reset_index(drop=True),
        }

        p = partitions[country]

        print(
            f"      [{country}] "
            f"S1={len(p['s1']):>7,}  "
            f"S2={len(p['s2']):>7,}  "
            f"S3={len(p['s3']):>7,}"
        )

    # --------------------------------------------------------
    # Preserve original handling of missing country
    # --------------------------------------------------------

    uk_s1 = s1[
        s1["country"] == ""
    ].reset_index(drop=True)

    uk_s2 = s2[
        s2["country"] == ""
    ].reset_index(drop=True)

    uk_s3 = s3[
        s3["country"] == ""
    ].reset_index(drop=True)

    if len(uk_s1) + len(uk_s2) + len(uk_s3) > 0:

        partitions["__unknown__"] = {
            "s1": uk_s1,
            "s2": uk_s2,
            "s3": uk_s3,
        }

        print(
            f"      [__unknown__] "
            f"S1={len(uk_s1):>7,}  "
            f"S2={len(uk_s2):>7,}  "
            f"S3={len(uk_s3):>7,}"
        )

    return partitions


# ============================================================
# COUNTRY -> SAFE FILE NAME
# Matches original save_processed_data() behavior
# ============================================================

def country_slug(country: str) -> str:

    slug = re.sub(
        r"[^\w]",
        "_",
        country
    ).strip("_").lower()

    return slug or "unknown"


# ============================================================
# APPEND PARTITION TO TSV
# ============================================================

def append_partition(
    df: pd.DataFrame,
    path: str,
) -> None:

    if df.empty:
        return

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    file_exists = os.path.exists(path)

    df.to_csv(
        path,
        sep="\t",
        index=False,
        mode="a",
        header=not file_exists,
    )


# ============================================================
# CHUNKED TRAIN PARTITIONING
# ============================================================

def partition_train_in_chunks(
    chunk_size: int = CHUNK_SIZE,
) -> None:

    source_paths = {
        "s1": os.path.join(
            STAGE0_DIR,
            "train_source1.tsv"
        ),

        "s2": os.path.join(
            STAGE0_DIR,
            "train_source2.tsv"
        ),

        "s3": os.path.join(
            STAGE0_DIR,
            "train_source3.tsv"
        ),
    }

    print("=" * 75)
    print("STAGE 1 — COUNTRY PARTITIONING ONLY")
    print("=" * 75)

    print()
    print("Reading already-normalized Stage-0 files.")
    print("NO normalization will be performed.")
    print(f"Chunk size: {chunk_size:,}")
    print()

    # ========================================================
    # 1. VERIFY STAGE-0 INPUT FILES
    # ========================================================

    for src, path in source_paths.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nMissing Stage-0 file for {src}:\n"
                f"  {path}\n\n"
                "Stage-0 normalization must be completed first."
            )

    # ========================================================
    # 2. READ COLUMN NAMES ONLY
    #
    # This is important for the case where one source finishes
    # before another source.
    #
    # Example:
    #   S1 finishes first.
    #   We still give its empty DataFrame the same columns,
    #   including 'country'.
    #
    # This prevents KeyError: 'country'.
    # ========================================================

    source_columns = {}

    for src, path in source_paths.items():

        header_df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            nrows=0
        )

        header_df.columns = (
            header_df.columns
            .str.strip()
        )

        source_columns[src] = header_df.columns.tolist()

        if "country" not in source_columns[src]:

            raise ValueError(
                f"'country' column missing from {path}"
            )

    # ========================================================
    # 3. REMOVE OLD STAGE-1 TRAIN OUTPUT
    #
    # Prevents duplicate rows if this script is rerun.
    #
    # IMPORTANT:
    # Only Stage-1 TRAIN is removed.
    # Stage-0 is NEVER touched.
    # ========================================================

    if os.path.exists(STAGE1_TRAIN_DIR):

        print(
            "Removing existing Stage-1 TRAIN directory:"
        )

        print(
            f"  {STAGE1_TRAIN_DIR}"
        )

        shutil.rmtree(
            STAGE1_TRAIN_DIR
        )

    os.makedirs(
        STAGE1_TRAIN_DIR,
        exist_ok=True
    )

    # ========================================================
    # 4. OPEN CHUNK READERS
    # ========================================================

    readers = {}

    for src, path in source_paths.items():

        readers[src] = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=chunk_size,
        )

    # ========================================================
    # 5. TRACK INPUT / OUTPUT COUNTS
    #
    # This gives us a final no-loss/no-duplication check.
    # ========================================================

    input_counts = {
        "s1": 0,
        "s2": 0,
        "s3": 0,
    }

    partitioned_counts = {
        "s1": 0,
        "s2": 0,
        "s3": 0,
    }

    partition_counts = {}

    finished = {
        "s1": False,
        "s2": False,
        "s3": False,
    }

    chunk_number = 0

    # ========================================================
    # 6. PROCESS CHUNKS
    # ========================================================

    while not all(finished.values()):

        chunk_number += 1

        print()
        print("-" * 75)
        print(
            f"PARTITION CHUNK {chunk_number:,}"
        )
        print("-" * 75)

        chunks = {}

        # ----------------------------------------------------
        # Get next chunk from each source
        # ----------------------------------------------------

        for src in ["s1", "s2", "s3"]:

            if finished[src]:

                # IMPORTANT:
                # Empty DataFrame STILL has correct columns.
                chunks[src] = pd.DataFrame(
                    columns=source_columns[src]
                )

                continue

            try:

                chunk = next(
                    readers[src]
                )

                chunk.columns = (
                    chunk.columns
                    .str.strip()
                )

                # Convert missing country values to ""
                # exactly as expected by partition_by_country().
                chunk = chunk.fillna("")

                if "country" not in chunk.columns:

                    raise ValueError(
                        f"'country' column missing "
                        f"from Stage-0 {src} chunk."
                    )

                chunks[src] = chunk

                input_counts[src] += len(chunk)

                print(
                    f"  {src.upper()}: "
                    f"{len(chunk):,} rows"
                )

            except StopIteration:

                finished[src] = True

                # IMPORTANT:
                # Correct columns even after exhaustion.
                chunks[src] = pd.DataFrame(
                    columns=source_columns[src]
                )

                print(
                    f"  {src.upper()}: finished"
                )

        # ----------------------------------------------------
        # If all sources are exhausted, stop.
        # ----------------------------------------------------

        if all(
            finished[src]
            and chunks[src].empty
            for src in ["s1", "s2", "s3"]
        ):
            break

        # ====================================================
        # 7. CALL THE SAME COUNTRY PARTITION FUNCTION
        #    USED IN THE MERGED PIPELINE
        # ====================================================

        partitions = partition_by_country(
            s1=chunks["s1"],
            s2=chunks["s2"],
            s3=chunks["s3"],
        )

        # ====================================================
        # 8. WRITE EACH COUNTRY PARTITION
        # ====================================================

        for country, srcs in partitions.items():

            slug = country_slug(
                country
            )

            if country not in partition_counts:

                partition_counts[country] = {
                    "s1": 0,
                    "s2": 0,
                    "s3": 0,
                }

            for src_key in [
                "s1",
                "s2",
                "s3",
            ]:

                part_df = srcs[src_key]

                if part_df.empty:
                    continue

                output_path = os.path.join(
                    STAGE1_TRAIN_DIR,
                    f"{slug}_{src_key}.tsv"
                )

                append_partition(
                    part_df,
                    output_path,
                )

                partition_counts[country][src_key] += (
                    len(part_df)
                )

                partitioned_counts[src_key] += (
                    len(part_df)
                )

        # Explicit cleanup
        del chunks
        del partitions

    # ========================================================
    # 9. VERIFY NO ROWS WERE LOST
    # ========================================================

    print()
    print("=" * 75)
    print("VERIFYING PARTITION COUNTS")
    print("=" * 75)

    all_counts_match = True

    for src in ["s1", "s2", "s3"]:

        input_count = input_counts[src]
        output_count = partitioned_counts[src]

        print(
            f"{src.upper()}: "
            f"Stage-0={input_count:,}  "
            f"Stage-1={output_count:,}"
        )

        if input_count != output_count:

            all_counts_match = False

            print(
                f"  ERROR: count mismatch for {src}!"
            )

    if not all_counts_match:

        raise RuntimeError(
            "\nPARTITION VALIDATION FAILED.\n"
            "The Stage-0 and Stage-1 row counts do not match."
        )

    print()
    print("✓ All source row counts match.")
    print("✓ No rows were lost during partitioning.")
    print("✓ No rows were duplicated during partitioning.")

    # ========================================================
    # 10. SAVE PARTITION SUMMARY
    # ========================================================

    os.makedirs(
        META_DIR,
        exist_ok=True
    )

    summary_path = os.path.join(
        META_DIR,
        "partition_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "train": partition_counts
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # 11. FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 75)
    print("TRAIN COUNTRY PARTITIONING COMPLETE")
    print("=" * 75)

    print()
    print("Country partitions:")

    for country in sorted(
        partition_counts
    ):

        counts = partition_counts[country]

        print(
            f"  {country}: "
            f"S1={counts['s1']:,}  "
            f"S2={counts['s2']:,}  "
            f"S3={counts['s3']:,}"
        )

    print()
    print(
        f"Stage-1 output:"
        f" {STAGE1_TRAIN_DIR}/"
    )

    print(
        f"Partition summary:"
        f" {summary_path}"
    )

    print()
    print("Normalization was NOT run.")
    print("Only country partitioning was performed.")
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    partition_train_in_chunks(
        chunk_size=CHUNK_SIZE
    )