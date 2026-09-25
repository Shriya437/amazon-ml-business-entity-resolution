from pathlib import Path
import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

PROCESSED_DIR = Path("dataset/processed")

FILES = [
    "sample_source1_normalized.tsv",
    "sample_source2_normalized.tsv",
    "sample_source3_normalized.tsv",
]


# ============================================================================
# EXPECTED COLUMNS
# ============================================================================

EXPECTED_NEW_COLUMNS = [
    "name_full",
    "name_no_suffix",
    "name_sorted",
    "name_first2",
    "name_ascii",
    "name_transliterated",
    "name_transliterated_no_suffix",
    "is_domain_name",
    "script_type",
    "has_address",
    "pin_code",
    "zip_code",
    "street_num",
    "state_token",
    "locality_tokens",
]


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_file(filename):

    path = PROCESSED_DIR / filename

    print("\n" + "=" * 80)
    print(f"VERIFYING: {filename}")
    print("=" * 80)

    if not path.exists():
        print("ERROR: File not found!")
        return

    # Read the processed sample.
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ------------------------------------------------------------------------
    # 1. ROW COUNT
    # ------------------------------------------------------------------------

    if len(df) == 20_000:
        print("[PASS] Exactly 20,000 rows")
    else:
        print(
            f"[CHECK] Expected 20,000 rows, found {len(df):,}"
        )

    # ------------------------------------------------------------------------
    # 2. ORIGINAL COLUMNS
    # ------------------------------------------------------------------------

    original_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    missing_original = [
        col
        for col in original_columns
        if col not in df.columns
    ]

    if not missing_original:
        print("[PASS] Original columns preserved")
    else:
        print(
            "[FAIL] Missing original columns:",
            missing_original
        )

    # ------------------------------------------------------------------------
    # 3. NEW NORMALIZATION COLUMNS
    # ------------------------------------------------------------------------

    missing_new = [
        col
        for col in EXPECTED_NEW_COLUMNS
        if col not in df.columns
    ]

    if not missing_new:
        print("[PASS] All normalization columns present")
    else:
        print(
            "[FAIL] Missing normalization columns:",
            missing_new
        )

    # ------------------------------------------------------------------------
    # 4. COUNTRY DISTRIBUTION
    # ------------------------------------------------------------------------

    print("\nCountry distribution:")
    print(
        df["country"].value_counts(
            dropna=False
        )
    )

    # ------------------------------------------------------------------------
    # 5. MISSING VALUES
    # ------------------------------------------------------------------------

    print("\nMissing values in original columns:")

    for col in original_columns:
        missing = (
            df[col]
            .isna()
            .sum()
        )

        print(
            f"  {col}: {missing:,}"
        )

    # ------------------------------------------------------------------------
    # 6. SCRIPT TYPES
    # ------------------------------------------------------------------------

    print("\nScript types:")
    print(
        df["script_type"]
        .value_counts(dropna=False)
    )

    # ------------------------------------------------------------------------
    # 7. DOMAIN NAMES
    # ------------------------------------------------------------------------

    print("\nDomain-like names:")

    domain_count = (
        df["is_domain_name"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    print(
        f"  {domain_count:,} / {len(df):,}"
    )

    # ------------------------------------------------------------------------
    # 8. ADDRESS AVAILABILITY
    # ------------------------------------------------------------------------

    print("\nAddress availability:")

    address_count = (
        df["has_address"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    print(
        f"  Has address:    {address_count:,}"
    )

    print(
        f"  Missing address: {len(df) - address_count:,}"
    )

    # ------------------------------------------------------------------------
    # 9. NAME EXAMPLES
    # ------------------------------------------------------------------------

    print("\nName normalization examples:")

    example_columns = [
    "business_name",
    "name_full",
    "name_no_suffix",
    "name_sorted",
    "name_first2",
    "name_ascii",
    "name_transliterated",
    "name_transliterated_no_suffix",
    "script_type",
    "is_domain_name",
]

    print(
        df[example_columns]
        .sample(
            n=min(15, len(df)),
            random_state=42,
        )
        .to_string(index=False)
    )

    # ------------------------------------------------------------------------
    # 10. ADDRESS EXAMPLES
    # ------------------------------------------------------------------------

    print("\nAddress normalization examples:")

    address_columns = [
        "business_address",
        "has_address",
        "pin_code",
        "zip_code",
        "street_num",
        "state_token",
        "locality_tokens",
    ]

    print(
        df[address_columns]
        .sample(
            n=min(15, len(df)),
            random_state=42,
        )
        .to_string(index=False)
    )

    # ------------------------------------------------------------------------
    # 11. LLP / LLC / LP CHECK
    # ------------------------------------------------------------------------

    print("\nLegal suffix check:")

    for suffix in ["llp", "llc", "lp"]:

        mask = (
            df["name_full"]
            .str.split()
            .apply(
                lambda tokens: suffix in tokens
            )
        )

        print(
            f"  {suffix.upper()}: {mask.sum():,} rows"
        )

    # ------------------------------------------------------------------------
    # 12. NAME EMPTY CHECK
    # ------------------------------------------------------------------------

    empty_names = (
        df["name_full"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print("\nEmpty normalized names:")

    if empty_names == 0:
        print("[PASS] No empty normalized names")
    else:
        print(
            f"[CHECK] {empty_names:,} empty normalized names"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("STAGE 0 — NORMALIZATION VERIFICATION")
    print("=" * 80)

    for filename in FILES:
        verify_file(filename)

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()