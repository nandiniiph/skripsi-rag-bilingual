import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT
    / "data/manual_review/"
    / "final_evaluation_translation_review.csv"
)

# =========================================================
# LOAD
# =========================================================

df = pd.read_csv(FILE)

# =========================================================
# NORMALISASI TIPE DATA
# =========================================================

for col in ["revision", "notes"]:

    if col not in df.columns:
        df[col] = ""

    df[col] = (
        df[col]
        .fillna("")
        .astype("object")
    )

df["english_text"] = (
    df["english_text"]
    .fillna("")
    .astype("object")
)

df["indonesian_text"] = (
    df["indonesian_text"]
    .fillna("")
    .astype("object")
)

df["is_valid"] = pd.to_numeric(
    df["is_valid"],
    errors="coerce"
).astype("Int64")

# =========================================================
# BACKUP
# =========================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup = FILE.with_name(
    f"final_evaluation_translation_review_backup_{timestamp}.csv"
)

shutil.copy2(FILE, backup)

print("Backup:")
print(backup)

# =========================================================
# CEK STATUS PAIR
# =========================================================

def pair_complete(group):
    return group["is_valid"].notna().all()


groups = list(df.groupby("pair_id", sort=True))

completed = [
    pair_id
    for pair_id, g in groups
    if pair_complete(g)
]

pending = [
    pair_id
    for pair_id, g in groups
    if not pair_complete(g)
]

print("\n" + "=" * 70)
print("FINAL BILINGUAL TRANSLATION REVIEW")
print("=" * 70)

print("TOTAL PAIR      :", len(groups))
print("SUDAH DIREVIEW  :", len(completed))
print("BELUM DIREVIEW  :", len(pending))

# =========================================================
# REVIEW
# =========================================================

for nomor, pair_id in enumerate(
    pending,
    start=1
):

    idxs = df.index[
        df["pair_id"].eq(pair_id)
    ].tolist()

    print("\n" + "=" * 100)
    print(
        f"PERTANYAAN {nomor}/{len(pending)} | {pair_id}"
    )
    print("=" * 100)

    # Tampilkan seluruh row dalam pair
    for local_no, idx in enumerate(
        idxs,
        start=1
    ):

        row = df.loc[idx]

        print("\n" + "-" * 100)

        print(
            f"[{local_no}] TYPE: {row['type']}"
        )

        if pd.notna(row.get("content_id")):
            print(
                "CONTENT ID:",
                row["content_id"]
            )

        print("\nENGLISH:")
        print(row["english_text"])

        print("\nINDONESIAN:")
        print(row["indonesian_text"])

    print("\n" + "-" * 100)

    print("\nPilihan:")
    print(
        "  a        -> semua terjemahan dalam pair VALID"
    )
    print(
        "  e        -> ada terjemahan yang perlu diperbaiki"
    )
    print(
        "  s        -> skip sementara"
    )
    print(
        "  q        -> simpan & keluar"
    )

    choice = input("\nPilihan: ").strip().lower()

    # =====================================================
    # ALL VALID
    # =====================================================

    if choice == "a":

        for idx in idxs:

            df.at[idx, "is_valid"] = 1
            df.at[idx, "revision"] = ""
            df.at[idx, "notes"] = (
                "Valid setelah review manual"
            )

        print("✓ Semua row pada pair ditandai VALID.")

    # =====================================================
    # EDIT / INVALID
    # =====================================================

    elif choice == "e":

        print(
            "\nMasukkan nomor row yang perlu diperbaiki."
        )
        print(
            "Contoh: 1 atau 1,3"
        )

        raw = input(
            "Nomor row bermasalah: "
        ).strip()

        try:

            invalid_local = [
                int(x.strip())
                for x in raw.split(",")
                if x.strip()
            ]

        except ValueError:

            print(
                "Input tidak valid. Pair dilewati."
            )

            continue

        invalid_set = set(invalid_local)

        # Default: row lain valid
        for local_no, idx in enumerate(
            idxs,
            start=1
        ):

            if local_no not in invalid_set:

                df.at[idx, "is_valid"] = 1
                df.at[idx, "revision"] = ""
                df.at[idx, "notes"] = (
                    "Valid setelah review manual"
                )

        # Minta revision untuk yang bermasalah
        for local_no in invalid_local:

            if (
                local_no < 1
                or local_no > len(idxs)
            ):
                print(
                    f"Nomor {local_no} tidak valid."
                )
                continue

            idx = idxs[
                local_no - 1
            ]

            print("\n" + "=" * 80)
            print(
                f"REVISI ROW [{local_no}]"
            )
            print("=" * 80)

            print("\nINDONESIAN SAAT INI:")
            print(
                df.at[
                    idx,
                    "indonesian_text"
                ]
            )

            revision = input(
                "\nMasukkan terjemahan revisi:\n> "
            ).strip()

            if not revision:

                print(
                    "Revision kosong. "
                    "Row tetap pending."
                )

                df.at[
                    idx,
                    "is_valid"
                ] = pd.NA

                continue

            df.at[
                idx,
                "is_valid"
            ] = 1

            df.at[
                idx,
                "revision"
            ] = revision

            df.at[
                idx,
                "notes"
            ] = (
                "Terjemahan direvisi "
                "setelah review manual"
            )

            print("✓ Revision disimpan.")

    # =====================================================
    # SKIP
    # =====================================================

    elif choice == "s":

        print(
            "Pair di-skip sementara."
        )

    # =====================================================
    # QUIT
    # =====================================================

    elif choice == "q":

        df.to_csv(
            FILE,
            index=False
        )

        print("\nProgress disimpan.")
        break

    else:

        print(
            "Pilihan tidak dikenali. "
            "Pair dilewati."
        )

    # checkpoint setiap pair
    df.to_csv(
        FILE,
        index=False
    )

# =========================================================
# SAVE
# =========================================================

df.to_csv(
    FILE,
    index=False
)

# =========================================================
# FINAL VALIDATION
# =========================================================

check = pd.read_csv(FILE)

status = pd.to_numeric(
    check["is_valid"],
    errors="coerce"
)

pair_status = (
    check.assign(
        _status=status
    )
    .groupby("pair_id")["_status"]
    .apply(
        lambda x: x.notna().all()
    )
)

print("\n" + "=" * 70)
print("HASIL REVIEW")
print("=" * 70)

print(
    "TOTAL BARIS       :",
    len(check)
)

print(
    "TOTAL PAIR        :",
    check["pair_id"].nunique()
)

print(
    "VALID             :",
    int((status == 1).sum())
)

print(
    "BELUM DIREVIEW    :",
    int(status.isna().sum())
)

print(
    "PAIR SELESAI      :",
    int(pair_status.sum())
)

print(
    "PAIR PENDING      :",
    int((~pair_status).sum())
)

revision_count = (
    check["revision"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
    .sum()
)

print(
    "TOTAL REVISION    :",
    revision_count
)

if (
    status.isna().sum() == 0
    and pair_status.all()
):
    print("\n" + "=" * 70)
    print("SUKSES")
    print("=" * 70)
    print(
        "Seluruh evaluation translation "
        "selesai direview."
    )
else:
    print(
        "\nMasih ada translation "
        "yang perlu direview."
    )