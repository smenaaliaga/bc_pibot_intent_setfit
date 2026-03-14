"""Clean all datasets: remove tildes, fix typos, deduplicate."""
import csv
from pathlib import Path
from collections import Counter

TILDE_MAP = str.maketrans(
    "áéíóúüñÁÉÍÓÚÜÑ",
    "aeiouunAEIOUUN",
)

def remove_tildes(text: str) -> str:
    return text.translate(TILDE_MAP)

def clean_csv(path: Path) -> None:
    print(f"\n{'='*50}")
    print(f"Procesando: {path.name}")
    print(f"{'='*50}")

    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                text = remove_tildes(row[0].strip())
                label = row[1].strip()
                # Fix typo
                if label == "methodologyy":
                    label = "methodology"
                if text and label:
                    rows.append((text, label))

    # Deduplicate keeping first occurrence
    seen = set()
    unique = []
    for text, label in rows:
        key = (text, label)
        if key not in seen:
            seen.add(key)
            unique.append((text, label))

    removed = len(rows) - len(unique)

    # Write
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for text, label in unique:
            writer.writerow([text, label])

    # Summary
    counts = Counter(label for _, label in unique)
    print(f"  Originales: {len(rows)}")
    print(f"  Duplicados eliminados: {removed}")
    print(f"  Total final: {len(unique)}")
    for label, count in counts.most_common():
        print(f"    {label}: {count}")

if __name__ == "__main__":
    data_dir = Path("data")
    for name in ["dataset_intent.csv", "dataset_macro.csv", "dataset_context.csv"]:
        p = data_dir / name
        if p.exists():
            clean_csv(p)
    print("\nListo!")
