# gt.py - Ground truth for news article classification
# Dataset: AG News (https://huggingface.co/datasets/fancyzhx/ag_news)
# 30 articles sampled from the AG News test set.
# Import this file into your program and call correct_category(num, category).

_ground_truth = {
    1: "world",
    2: "sports",
    3: "business",
    4: "world",
    5: "world",
    6: "business",
    7: "sci/tech",
    8: "business",
    9: "sci/tech",
    10: "sci/tech",
    11: "sci/tech",
    12: "business",
    13: "sports",
    14: "sports",
    15: "sports",
    16: "sci/tech",
    17: "sci/tech",
    18: "world",
    19: "business",
    20: "business",
    21: "world",
    22: "sports",
    23: "business",
    24: "sports",
    25: "sports",
    26: "sports",
    27: "world",
    28: "world",
    29: "sci/tech",
    30: "world",
}

# Instrctions says this function should given to us However, they did not provided it, so I implemented it myself.
def get_correct_category(num: int) -> str:
    """
    Returns the ground-truth category label for article number `num`.

    Args:
        num: The article number (1-30).

    Returns:
        The category string: "world", "sports", "business", or "sci/tech".

    Raises:
        KeyError: If `num` is not a valid article number in the ground truth.
    """
    return _ground_truth[num]


def correct_category(num: int, category: str) -> bool:
    """
    Returns True if article number `num` has category `category`.
    Otherwise returns False.
    Args:
        num: The article number (1-30).
        category: The category string (lowercase): "world", "sports",
                  "business", or "sci/tech".
    Returns:
        True if the category matches the ground truth, False otherwise.
    """
    return _ground_truth.get(num, "").lower() == category.lower()
