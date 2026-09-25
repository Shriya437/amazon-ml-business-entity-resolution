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
    "address_transliterated",
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

        # The rest of the verification depends on these columns.
        # Stop this file's verification rather than producing misleading
        # KeyError messages.
        return

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
        "address_transliterated",
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
    # 11. ADDRESS TRANSLITERATION — POPULATION CHECK
    # ------------------------------------------------------------------------

    print("\nAddress transliteration check:")

    # Rows where the original address is marked as present.
    address_rows = df[
        df["has_address"]
        .astype(str)
        .str.lower()
        .eq("true")
    ]

    address_rows_count = len(address_rows)

    # Count non-empty transliterated addresses among rows
    # that actually have an address.
    non_empty_transliterated = (
        address_rows["address_transliterated"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print(
        f"  Addresses present:              "
        f"{address_rows_count:,}"
    )

    print(
        f"  Non-empty transliterations:     "
        f"{non_empty_transliterated:,}"
    )

    if address_rows_count == 0:
        print(
            "[CHECK] No addresses available for transliteration test"
        )

    elif non_empty_transliterated > 0:
        print(
            "[PASS] Address transliteration is populated"
        )

    else:
        print(
            "[FAIL] Address transliteration column exists "
            "but contains no populated values"
        )

    # ------------------------------------------------------------------------
    # 12. MISSING ADDRESS TRANSLITERATION CHECK
    # ------------------------------------------------------------------------

    print("\nMissing-address transliteration check:")

    missing_address_rows = df[
        ~df["has_address"]
        .astype(str)
        .str.lower()
        .eq("true")
    ]

    if len(missing_address_rows) == 0:

        print(
            "[CHECK] No missing-address rows in this sample"
        )

    else:

        non_empty_for_missing = (
            missing_address_rows["address_transliterated"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        print(
            f"  Missing-address rows: "
            f"{len(missing_address_rows):,}"
        )

        print(
            f"  Non-empty transliterations among them: "
            f"{non_empty_for_missing:,}"
        )

        if non_empty_for_missing == 0:
            print(
                "[PASS] Missing addresses have empty "
                "address_transliterated"
            )
        else:
            print(
                "[CHECK] Some missing addresses have "
                "non-empty address_transliterated"
            )

    # ------------------------------------------------------------------------
    # 13. INDIC-SCRIPT ADDRESS CHECK
    # ------------------------------------------------------------------------

    print("\nIndic-script address check:")

    # Unicode ranges covering the Indic scripts supported by the
    # normalization pipeline.
    indic_address_pattern = (
        r"[\u0900-\u097F"   # Devanagari
        r"\u0980-\u09FF"    # Bengali
        r"\u0A00-\u0A7F"    # Gurmukhi
        r"\u0A80-\u0AFF"    # Gujarati
        r"\u0B00-\u0B7F"    # Oriya
        r"\u0B80-\u0BFF"    # Tamil
        r"\u0C00-\u0C7F"    # Telugu
        r"\u0C80-\u0CFF"    # Kannada
        r"\u0D00-\u0D7F"    # Malayalam
        r"]"
    )

    indic_address_mask = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .str.contains(
            indic_address_pattern,
            regex=True,
            na=False,
        )
    )

    indic_address_df = df[indic_address_mask]

    print(
        f"  Indic-script addresses found: "
        f"{len(indic_address_df):,}"
    )

    if len(indic_address_df) == 0:

        print(
            "[CHECK] No Indic-script addresses found "
            "in this random sample"
        )

    else:

        indic_example_columns = [
            "business_address",
            "address_transliterated",
            "state_token",
            "locality_tokens",
        ]

        print(
            "\n  Indic-script address examples:"
        )

        print(
            indic_address_df[indic_example_columns]
            .head(10)
            .to_string(index=False)
        )

        # Check whether the Indic-script addresses have
        # corresponding transliterated values.
        indic_non_empty = (
            indic_address_df["address_transliterated"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )

        print(
            f"\n  Indic addresses with transliteration: "
            f"{indic_non_empty:,} / {len(indic_address_df):,}"
        )

        if indic_non_empty > 0:
            print(
                "[PASS] Indic-script address transliteration "
                "is working on sampled rows"
            )
        else:
            print(
                "[FAIL] Indic-script addresses were found, "
                "but none have transliterated values"
            )

    # ------------------------------------------------------------------------
    # 14. LEGAL SUFFIX CHECK
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
    # 15. NAME EMPTY CHECK
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

        print(
            "[PASS] No empty normalized names"
        )

    else:

        print(
            f"[CHECK] {empty_names:,} empty normalized names"
        )

    # ------------------------------------------------------------------------
    # 16. ADDRESS TRANSLITERATION COLUMN SUMMARY
    # ------------------------------------------------------------------------

    print("\nAddress transliteration summary:")

    print(
        df[
            [
                "business_address",
                "address_transliterated",
            ]
        ]
        .head(5)
        .to_string(index=False)
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