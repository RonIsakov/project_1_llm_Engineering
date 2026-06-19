import re

SYSTEM = """You are a news article classifier. Classify articles into exactly ONE of four categories: world, sports, business, sci/tech.

Reason step by step before giving your answer:
1. Identify the main topic and key entities mentioned in the article.
2. Note which category indicators are present (governments or conflicts suggest world; teams, matches, or athletes suggest sports; companies, markets, or earnings suggest business; technology, research, or gadgets suggest sci/tech).
3. On the very last line of your response, write exactly: FINAL: <label>
where <label> is one of: world, sports, business, sci/tech. The FINAL line must be lowercase with no extra words or punctuation."""

_FINAL_RE = re.compile(r"FINAL:\s*([A-Za-z/]+)", re.IGNORECASE)


def user_prompt(article: str) -> str:
    return f"Article: {article}"


def parse_final_label(raw: str) -> str:
    match = _FINAL_RE.search(raw)
    if not match:
        return ""
    return match.group(1).lower().strip()
