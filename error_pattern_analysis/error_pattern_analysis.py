import json
from collections import Counter
from pathlib import Path


def load_errors(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["errors"]


def summarize_directional_confusions(errors):
    counts = Counter()
    for e in errors:
        key = f"{e['true']}→{e['predicted']}"
        counts[key] += 1
    return counts


def bucket_hot_take_lengths(errors, threshold: int = 80):
    """
    Return counts of hot_take misclassifications split by text length.

    short:  length <= threshold
    long:   length > threshold
    """
    short = 0
    long = 0
    for e in errors:
        if e["true"] != "hot_take":
            continue
        length = len(e["text"])
        if length <= threshold:
            short += 1
        else:
            long += 1
    return {"short_≤80": short, "long_>80": long}


def main():
    path = Path(__file__).resolve().parent / "misclassifications_tuned.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Export tuned-run errors into error_pattern_analysis/ first."
        )

    errors = load_errors(path)

    print("=== Tuned run misclassifications summary ===")
    print(f"Total errors recorded: {len(errors)}")
    print()

    # Directional confusion counts
    dir_counts = summarize_directional_confusions(errors)
    print("Directional confusions (true→predicted):")
    for key in sorted(dir_counts.keys()):
        print(f"  {key}: {dir_counts[key]}")
    print()

    # Hot_take-specific analysis
    hot_take_total = sum(1 for e in errors if e["true"] == "hot_take")
    hot_take_correct = sum(
        1 for e in errors if e["true"] == "hot_take" and e["predicted"] == "hot_take"
    )
    print("hot_take-specific summary (errors only):")
    print(f"  true hot_take in error set: {hot_take_total}")
    print(f"  correct hot_take in error set: {hot_take_correct}")
    print()

    length_buckets = bucket_hot_take_lengths(errors)
    print("hot_take misclassifications by text length bucket:")
    for bucket, count in length_buckets.items():
        print(f"  {bucket}: {count}")


if __name__ == "__main__":
    main()

