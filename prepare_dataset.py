# prepare_dataset.py - Downloads and prepares the AG News dataset for Homework 1.
# Run this script ONCE before starting the assignment:
#
#   pip install datasets
#   python prepare_dataset.py
#
# It will create the file "articles.txt" in the current directory.
# The script downloads 30 pre-selected articles from the AG News dataset
# (a well-known news classification benchmark) hosted on HuggingFace.

from datasets import load_dataset

# Pre-selected indices from the AG News test set (seed=42)
SELECTED_INDICES = [
    5910, 1663, 3976, 992, 2527, 2190, 7096, 6290, 761, 5317,
    1236, 1654, 5665, 5195, 4373, 5411, 4555, 2016, 3242, 1615,
    2218, 214, 2703, 6232, 1944, 4868, 6894, 1266, 3706, 211,
]


def clean_text(text: str) -> str:
    """Clean HTML entities and formatting artifacts from AG News text."""
    t = text.replace(" #39;", "'").replace("#39;", "'")
    t = t.replace(" amp;", " &").replace("amp;", "&")
    t = t.replace(" quot;", '"').replace("quot;", '"')
    t = t.replace(" lt;", "<").replace(" gt;", ">")
    t = t.replace("\\$", "$")
    return t.strip()


def main():
    print("Downloading AG News dataset from HuggingFace...")
    ds = load_dataset("fancyzhx/ag_news", split="test")
    print(f"Loaded {len(ds)} articles from AG News test set.")

    print("Extracting 30 selected articles...")
    with open("articles.txt", "w", encoding="utf-8") as f:
        for num, idx in enumerate(SELECTED_INDICES, 1):
            text = clean_text(ds[idx]["text"])
            f.write(f"{num}: {text}\n")

    print("Done! Created articles.txt with 30 articles.")
    print("You can now run hw1.py.")


if __name__ == "__main__":
    main()