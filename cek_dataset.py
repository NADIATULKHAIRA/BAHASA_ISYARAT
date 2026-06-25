import os

dataset_path = "data"

print("\nJumlah gambar per karakter:\n")

for kategori in ["Huruf", "Angka", "kalimat"]:

    kategori_path = os.path.join(
        dataset_path,
        kategori
    )

    if not os.path.exists(kategori_path):
        continue

    print("=== ", kategori, " ===")

    for label in sorted(os.listdir(kategori_path)):

        folder = os.path.join(
            kategori_path,
            label
        )

        if os.path.isdir(folder):

            jumlah = len([
                f for f in os.listdir(folder)
                if f.lower().endswith(
                    (".jpg",".png",".jpeg")
                )
            ])

            print(
                f"{label}: {jumlah} gambar"
            )

    print()