"""
TEST PIPELINE
-------------
Chunked normalization + country partitioning.

Execution strategy:
    S1 completely -> S2 completely -> S3 completely

For each source:
    1. Read 50,000 raw rows
    2. Normalize the chunk
    3. Append normalized rows to Stage-0
    4. Partition the normalized chunk by country
    5. Append country partitions to Stage-1
    6. Repeat until source is complete

This intentionally follows the TRAIN execution strategy while
combining normalization + partitioning into one pass per source.

TRAIN outputs are NEVER touched.
"""

from pathlib import Path
import json
import shutil
import pandas as pd

from normalization import preprocess_dataframe, partition_by_country


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 50_000

TEST_DIR = Path("dataset/test")

STAGE0_DIR = Path("processed_data/stage0")
STAGE1_TEST_DIR = Path("processed_data/stage1/test")

META_DIR = Path("processed_data/meta")
SUMMARY_FILE = META_DIR / "partition_summary.json"


SOURCES = {
    "s1": "test_source1.tsv",
    "s2": "test_source2.tsv",
    "s3": "test_source3.tsv",
}


# ============================================================
# HELPERS
# ============================================================

def country_slug(country):
    """
    Convert country name into a safe filename/folder component.
    """
    value = str(country).strip()

    if not value:
        return "__unknown__"

    return value.lower().replace(" ", "_").replace("/", "_")


def append_partition(
    partition_df,
    output_path,
    write_header_if_new=True
):
    """
    Append a partition dataframe to a TSV file.
    """
    if partition_df is None or len(partition_df) == 0:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()

    partition_df.to_csv(
        output_path,
        sep="\t",
        index=False,
        mode="a",
        header=(write_header_if_new and not file_exists),
    )

    return len(partition_df)


def clean_test_outputs():
    """
    Remove ONLY TEST Stage-0 files and TEST Stage-1 directory.

    TRAIN outputs are never touched.
    """

    print("\n" + "=" * 75)
    print("CLEANING OLD TEST OUTPUT")
    print("=" * 75)

    # --------------------------------------------------------
    # Remove old TEST Stage-0 files
    # --------------------------------------------------------

    for src, filename in SOURCES.items():

        path = STAGE0_DIR / filename

        if path.exists():
            path.unlink()
            print(f"  ✓ Removed old Stage-0: {path}")

    # --------------------------------------------------------
    # Remove old TEST Stage-1 directory
    # --------------------------------------------------------

    if STAGE1_TEST_DIR.exists():
        shutil.rmtree(STAGE1_TEST_DIR)
        print(f"  ✓ Removed old TEST Stage-1: {STAGE1_TEST_DIR}")

    print("\n  ✓ Old TEST outputs cleaned.")
    print("  ✓ TRAIN outputs were NOT touched.")


def verify_inputs():

    print("\n" + "=" * 75)
    print("VERIFYING TEST INPUT FILES")
    print("=" * 75)

    for src, filename in SOURCES.items():

        path = TEST_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing TEST input for {src}: {path}"
            )

        print(f"  ✓ {src.upper()}: {path}")


def get_raw_row_count(path):
    """
    Count rows in a TSV without loading the whole file.
    Header is excluded.
    """
    count = 0

    with open(path, "r", encoding="utf-8") as f:

        next(f, None)

        for _ in f:
            count += 1

    return count


def get_source_columns(path):
    """
    Read only the header so empty dataframes can preserve
    the correct schema when needed.
    """
    return pd.read_csv(
        path,
        sep="\t",
        nrows=0
    ).columns.tolist()


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(src, filename, expected_raw_count):
    """
    Process ONE source completely before moving to the next.

    S1 -> finish
    S2 -> finish
    S3 -> finish
    """

    input_path = TEST_DIR / filename
    stage0_path = STAGE0_DIR / filename

    print("\n\n" + "=" * 75)
    print(f"PROCESSING {src.upper()}")
    print("=" * 75)

    print(f"Input:     {input_path}")
    print(f"Stage-0:   {stage0_path}")
    print(f"Chunk:     {CHUNK_SIZE:,}")

    # --------------------------------------------------------
    # Read source in chunks
    # --------------------------------------------------------

    reader = pd.read_csv(
        input_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
    )

    total_raw = 0
    total_normalized = 0
    total_partitioned = 0

    chunk_number = 0

    # --------------------------------------------------------
    # Process chunks
    # --------------------------------------------------------

    for raw_chunk in reader:

        chunk_number += 1

        rows = len(raw_chunk)

        print("\n" + "-" * 75)
        print(
            f"{src.upper()} CHUNK {chunk_number}"
        )
        print("-" * 75)

        print(
            f"  {src.upper()}: read {rows:,} raw rows"
        )

        total_raw += rows

        # ====================================================
        # STEP 1 — NORMALIZATION
        # ====================================================

        normalized_chunk = preprocess_dataframe(
            raw_chunk,
            filename
        )

        normalized_rows = len(normalized_chunk)

        print(
            f"  {src.upper()}: normalized "
            f"{normalized_rows:,} rows"
        )

        if normalized_rows != rows:
            raise RuntimeError(
                f"ROW COUNT CHANGED during normalization!\n"
                f"{src.upper()} chunk {chunk_number}: "
                f"raw={rows}, normalized={normalized_rows}"
            )

        total_normalized += normalized_rows

        # ====================================================
        # STEP 2 — WRITE STAGE 0
        # ====================================================

        stage0_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        stage0_exists = stage0_path.exists()

        normalized_chunk.to_csv(
            stage0_path,
            sep="\t",
            index=False,
            mode="a",
            header=not stage0_exists,
        )

        print(
            f"  {src.upper()}: Stage-0 written "
            f"({normalized_rows:,} rows)"
        )

        # ====================================================
        # STEP 3 — COUNTRY PARTITIONING
        # ====================================================
        #
        # We call the SAME partition_by_country() function
        # used by the merged normalization logic.
        #
        # Since we are intentionally processing one source
        # at a time, the other two dataframes are empty but
        # preserve the same columns.
        # ====================================================

        empty_df = pd.DataFrame(
            columns=normalized_chunk.columns
        )

        if src == "s1":

            partitions = partition_by_country(
                normalized_chunk,
                empty_df,
                empty_df
            )

        elif src == "s2":

            partitions = partition_by_country(
                empty_df,
                normalized_chunk,
                empty_df
            )

        elif src == "s3":

            partitions = partition_by_country(
                empty_df,
                empty_df,
                normalized_chunk
            )

        else:
            raise ValueError(f"Unknown source: {src}")

        # ====================================================
        # STEP 4 — WRITE COUNTRY PARTITIONS
        # ====================================================

        chunk_partitioned = 0

        for country, partition_data in partitions.items():

            if src == "s1":
                partition_df = partition_data["s1"]

            elif src == "s2":
                partition_df = partition_data["s2"]

            else:
                partition_df = partition_data["s3"]

            if partition_df is None or len(partition_df) == 0:
                continue

            slug = country_slug(country)

            # Same style of country/source output used by
            # the partitioning pipeline.
            output_path = (
                STAGE1_TEST_DIR
                / f"{slug}_{src}.tsv"
            )

            written = append_partition(
                partition_df,
                output_path
            )

            chunk_partitioned += written

            print(
                f"    [{country}] "
                f"{src.upper()}={written:,}"
            )

        total_partitioned += chunk_partitioned

        # ====================================================
        # CHUNK VALIDATION
        # ====================================================

        if chunk_partitioned != normalized_rows:
            raise RuntimeError(
                f"PARTITION ROW COUNT MISMATCH!\n"
                f"{src.upper()} chunk {chunk_number}: "
                f"normalized={normalized_rows}, "
                f"partitioned={chunk_partitioned}"
            )

    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    print("\n" + "-" * 75)
    print(f"{src.upper()} COMPLETE")
    print("-" * 75)

    print(
        f"  Raw rows:         {total_raw:,}"
    )

    print(
        f"  Normalized rows:  {total_normalized:,}"
    )

    print(
        f"  Partitioned rows: {total_partitioned:,}"
    )

    if total_raw != expected_raw_count:
        raise RuntimeError(
            f"{src.upper()} raw count mismatch!\n"
            f"Expected: {expected_raw_count:,}\n"
            f"Found:    {total_raw:,}"
        )

    if total_raw != total_normalized:
        raise RuntimeError(
            f"{src.upper()} normalization count mismatch!"
        )

    if total_normalized != total_partitioned:
        raise RuntimeError(
            f"{src.upper()} partition count mismatch!"
        )

    print(
        f"  ✓ {src.upper()} row counts match."
    )

    return {
        "raw": total_raw,
        "normalized": total_normalized,
        "partitioned": total_partitioned,
        "chunks": chunk_number,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("TEST: CHUNKED NORMALIZATION + COUNTRY PARTITIONING")
    print("=" * 75)

    print(f"\nChunk size: {CHUNK_SIZE:,}")
    print(f"TEST input: {TEST_DIR}/")
    print(f"Stage-0 output: {STAGE0_DIR}/")
    print(f"Stage-1 output: {STAGE1_TEST_DIR}/")

    # --------------------------------------------------------
    # Verify inputs
    # --------------------------------------------------------

    verify_inputs()

    # --------------------------------------------------------
    # Clean ONLY TEST outputs
    # --------------------------------------------------------

    clean_test_outputs()

    # --------------------------------------------------------
    # Get expected raw row counts
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("COUNTING TEST INPUT ROWS")
    print("=" * 75)

    expected_counts = {}

    for src, filename in SOURCES.items():

        path = TEST_DIR / filename

        count = get_raw_row_count(path)

        expected_counts[src] = count

        print(
            f"  {src.upper()}: {count:,} rows"
        )

    # --------------------------------------------------------
    # PROCESS S1 COMPLETELY
    # --------------------------------------------------------

    results = {}

    results["s1"] = process_source(
        "s1",
        SOURCES["s1"],
        expected_counts["s1"]
    )

    # --------------------------------------------------------
    # PROCESS S2 COMPLETELY
    # --------------------------------------------------------

    results["s2"] = process_source(
        "s2",
        SOURCES["s2"],
        expected_counts["s2"]
    )

    # --------------------------------------------------------
    # PROCESS S3 COMPLETELY
    # --------------------------------------------------------

    results["s3"] = process_source(
        "s3",
        SOURCES["s3"],
        expected_counts["s3"]
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n\n" + "=" * 75)
    print("FINAL TEST VALIDATION")
    print("=" * 75)

    all_ok = True

    for src in ["s1", "s2", "s3"]:

        result = results[src]

        print(
            f"\n{src.upper()}:"
        )

        print(
            f"  Raw:         {result['raw']:,}"
        )

        print(
            f"  Normalized:  {result['normalized']:,}"
        )

        print(
            f"  Partitioned: {result['partitioned']:,}"
        )

        print(
            f"  Chunks:      {result['chunks']}"
        )

        if not (
            result["raw"]
            == result["normalized"]
            == result["partitioned"]
        ):
            all_ok = False

    # --------------------------------------------------------
    # Update partition summary
    # --------------------------------------------------------

    META_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {}

    if SUMMARY_FILE.exists():

        try:
            with open(
                SUMMARY_FILE,
                "r",
                encoding="utf-8"
            ) as f:
                summary = json.load(f)

        except Exception:
            summary = {}

    summary["test"] = {
        src: {
            "raw_rows": results[src]["raw"],
            "normalized_rows": results[src]["normalized"],
            "partitioned_rows": results[src]["partitioned"],
            "chunks": results[src]["chunks"],
        }
        for src in ["s1", "s2", "s3"]
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"\n  ✓ Updated: {SUMMARY_FILE}"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n" + "=" * 75)

    if all_ok:

        print("✓ TEST NORMALIZATION + PARTITIONING COMPLETE")
        print("✓ S1, S2, S3 row counts all match")
        print("✓ TRAIN outputs were NOT touched")

    else:

        print("✗ VALIDATION FAILED")
        raise RuntimeError(
            "One or more TEST source validations failed."
        )

    print("=" * 75)


if __name__ == "__main__":
    main()