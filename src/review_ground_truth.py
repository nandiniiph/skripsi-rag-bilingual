import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path(__file__).resolve().parents[1]

GT_PATH = (
    ROOT /
    "data/manual_review/ground_truth_annotation.csv"
)

# ============================================================
# BACKUP
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_path = GT_PATH.with_name(
    f"ground_truth_annotation_backup_{timestamp}.csv"
)

shutil.copy2(
    GT_PATH,
    backup_path
)

print("=" * 80)
print("GROUND TRUTH MANUAL REVIEW")
print("=" * 80)

print("\nBackup:")
print(backup_path)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(GT_PATH)

df["is_relevant"] = pd.to_numeric(
    df["is_relevant"],
    errors="coerce"
).astype("Int64")

df["notes"] = df["notes"].astype("object")

# ============================================================
# CARI PAIR YANG BELUM SELESAI
# ============================================================

pair_status = (
    df.groupby("pair_id")["is_relevant"]
    .apply(lambda x: x.notna().all())
)

pending_pairs = pair_status[
    ~pair_status
].index.tolist()

print("\nTOTAL PAIR        :", df["pair_id"].nunique())
print("SUDAH DIREVIEW   :", int(pair_status.sum()))
print("BELUM DIREVIEW   :", len(pending_pairs))

# ============================================================
# REVIEW PER PERTANYAAN
# ============================================================

for no, pair_id in enumerate(pending_pairs, start=1):

    group = df[
        df["pair_id"] == pair_id
    ].copy()

    first = group.iloc[0]

    print("\n\n")
    print("=" * 80)
    print(
        f"PERTANYAAN {no}/{len(pending_pairs)} | "
        f"{pair_id}"
    )
    print("=" * 80)

    print("\nQUESTION:")
    print(first["question"])

    print(
        f"\nJUMLAH KANDIDAT CHUNK: {len(group)}"
    )

    print("\n" + "-" * 80)

    # ========================================================
    # TAMPILKAN SEMUA CHUNK
    # ========================================================

    for nomor_chunk, (idx, row) in enumerate(
        group.iterrows(),
        start=1
    ):
        print(
            f"\n[{nomor_chunk}] "
            f"{row['content_id']}"
        )

        print("-" * 80)

        print(row["chunk_text"])

        print("-" * 80)

    # ========================================================
    # INPUT
    # ========================================================

    while True:

        print("\nPilih chunk yang RELEVAN.")

        print(
            "Contoh:"
            "\n  1        -> hanya chunk 1"
            "\n  1,3      -> chunk 1 dan 3"
            "\n  a        -> semua chunk relevan"
            "\n  s        -> skip sementara"
            "\n  q        -> simpan & keluar"
        )

        pilihan = input(
            "\nPilihan: "
        ).strip().lower()

        # ====================================================
        # QUIT
        # ====================================================

        if pilihan == "q":

            df.to_csv(
                GT_PATH,
                index=False
            )

            print("\nProgress disimpan.")
            print(
                "Jalankan script lagi untuk melanjutkan."
            )

            raise SystemExit

        # ====================================================
        # SKIP
        # ====================================================

        if pilihan == "s":

            print(
                "\nPertanyaan dilewati sementara."
            )

            break

        # ====================================================
        # ALL
        # ====================================================

        if pilihan == "a":

            selected_numbers = list(
                range(1, len(group) + 1)
            )

        else:

            try:

                selected_numbers = [
                    int(x.strip())
                    for x in pilihan.split(",")
                    if x.strip()
                ]

            except ValueError:

                print(
                    "\nINPUT SALAH."
                )
                continue

        # ====================================================
        # VALIDASI INPUT
        # ====================================================

        if not selected_numbers:

            print(
                "\nMinimal satu chunk harus relevan."
            )
            continue

        if any(
            x < 1 or x > len(group)
            for x in selected_numbers
        ):

            print(
                "\nNomor chunk di luar rentang."
            )
            continue

        selected_numbers = sorted(
            set(selected_numbers)
        )

        # ====================================================
        # SET SEMUA = 0
        # ====================================================

        group_indices = list(
            group.index
        )

        df.loc[
            group_indices,
            "is_relevant"
        ] = 0

        df.loc[
            group_indices,
            "notes"
        ] = "Tidak relevan langsung dengan pertanyaan"

        # ====================================================
        # SET PILIHAN = 1
        # ====================================================

        for x in selected_numbers:

            selected_idx = (
                group_indices[x - 1]
            )

            df.at[
                selected_idx,
                "is_relevant"
            ] = 1

            df.at[
                selected_idx,
                "notes"
            ] = "Relevan langsung dengan pertanyaan"

        # ====================================================
        # CHECKPOINT
        # ====================================================

        df.to_csv(
            GT_PATH,
            index=False
        )

        print(
            "\n✓ Disimpan."
        )

        print(
            "Chunk relevan:",
            ", ".join(
                str(x)
                for x in selected_numbers
            )
        )

        break


# ============================================================
# FINAL VALIDATION
# ============================================================

check = pd.read_csv(GT_PATH)

status = pd.to_numeric(
    check["is_relevant"],
    errors="coerce"
)

pair_summary = (
    check.assign(
        _status=status
    )
    .groupby("pair_id")
    .agg(
        total_chunks=(
            "content_id",
            "size"
        ),
        relevant=(
            "_status",
            lambda x: int(
                (x == 1).sum()
            )
        ),
        pending=(
            "_status",
            lambda x: int(
                x.isna().sum()
            )
        )
    )
)

no_relevant = pair_summary[
    pair_summary["relevant"] == 0
]

pending = pair_summary[
    pair_summary["pending"] > 0
]

print("\n\n")
print("=" * 80)
print("HASIL FINAL GROUND TRUTH")
print("=" * 80)

print("TOTAL BARIS        :", len(check))
print(
    "TOTAL PAIR         :",
    check["pair_id"].nunique()
)
print(
    "RELEVAN            :",
    int((status == 1).sum())
)
print(
    "TIDAK RELEVAN      :",
    int((status == 0).sum())
)
print(
    "BELUM DIISI        :",
    int(status.isna().sum())
)
print(
    "PAIR TANPA RELEVAN :",
    len(no_relevant)
)
print(
    "PAIR BELUM SELESAI :",
    len(pending)
)

if (
    status.isna().sum() == 0
    and len(no_relevant) == 0
):
    print("\n" + "=" * 80)
    print("SUKSES")
    print("=" * 80)

    print(
        "Semua ground truth selesai dianotasi."
    )

    print(
        "\nLanjutkan dengan:"
    )

    print(
        "python3 scripts/07_prepare_ground_truth.py --apply"
    )

else:

    print(
        "\nMasih ada ground truth "
        "yang perlu diselesaikan."
    )