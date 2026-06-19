SYSTEM = (
    "You are a news article classifier. Classify each article into exactly ONE "
    "of these four categories: world, sports, business, sci/tech. Respond with "
    "ONLY the category label, in lowercase, exactly as written (including the "
    "slash in sci/tech). No explanation, no punctuation, no extra words."
)


def user_prompt(article: str) -> str:
    return f"Article: {article}\n\nCategory:"
