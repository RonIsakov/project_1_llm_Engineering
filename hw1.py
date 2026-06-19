import re
from collections import Counter
from pathlib import Path

from openai import BadRequestError

from gt import correct_category, get_correct_category
from llm_client import LLMClient
from prompts import chain_of_thought, few_shot, zero_shot

CATEGORIES = ("world", "sports", "business", "sci/tech")
TEMPERATURES = (0, 0.5, 0.9)
PART2_TEMPERATURE = 0.5
NUM_CALLS = 5
ARTICLES_FILE = Path("articles.txt")
CLASSIFICATIONS_FILE = Path("classifications.txt")
MAJORITY_VOTING_FILE = Path("majority_voting.txt")
STATISTICS_FILE = Path("statistics.txt")

_ARTICLE_LINE_RE = re.compile(r"^\s*(\d+):\s*(.*)$")
_ARTICLE_HEADER_RE = re.compile(r"^\s*(\d+):\s*$")
_CLASSIFICATION_LINE_RE = re.compile(r"^\s*(0|0\.5|0\.9):\s*(.*)$")
_PROMPT_HEADER_RE = re.compile(r"^\s*(zero-shot|few-shot|chain-of-thought):\s*$")
_MAJORITY_VOTE_LINE_RE = re.compile(r"^\s*majority vote:\s*(.*)$")
_STRIP_CHARS = " \t\r\n\"'`.:;,!?"


def read_articles(path: Path) -> list[tuple[int, str]]:
    articles: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = _ARTICLE_LINE_RE.match(line)
        if not match:
            continue
        articles.append((int(match.group(1)), match.group(2).strip()))
    return articles


def normalize(response: str) -> str:
    if not response:
        return "unknown"
    lowered = response.lower()
    trimmed = lowered.strip(_STRIP_CHARS)
    if trimmed in {"world", "sports", "business", "sci/tech"}:
        return trimmed
    else:
        return "unknown"


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def majority(votes: list[str]) -> str:
    counts = Counter(votes)
    label, count = counts.most_common(1)[0]
    if count >= 3 and label != "unknown":
        return label
    return "no majority"


def _safe_complete(client: LLMClient, system: str, user: str, temperature: float, ctx: str) -> str | None:
    try:
        return client.complete(system, user, temperature)
    except BadRequestError as e:
        if getattr(e, "code", None) != "content_filter":
            raise
        print(f"  WARNING: {ctx} blocked by content filter")
        return None


# [(1, ["world", "sports", "world"])] ...
def read_classifications() -> list[tuple[int, list[str]]]:
    results: list[tuple[int, list[str]]] = []
    current_article_num: int | None = None
    current: dict[float, str] = {}

    for line in CLASSIFICATIONS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = _ARTICLE_HEADER_RE.match(line)
        if match:
            if current_article_num is not None:
                results.append((current_article_num, [current.get(temp, "unknown") for temp in TEMPERATURES]))
            current_article_num = int(match.group(1))
            current = {}
            continue
        match = _CLASSIFICATION_LINE_RE.match(line)
        if match and current_article_num is not None:
            current[float(match.group(1))] = normalize(match.group(2))
    if current_article_num is not None:
        results.append((current_article_num, [current.get(temp, "unknown") for temp in TEMPERATURES]))
    return results

# "zero-shot": [(1, "unknown"), (2, "sports"), (3, "business"), ...],
# "few-shot": [(1, "unknown"), (2, "sports"), (3, "business"), ...],
# "chain-of-thought": [(1, "unknown"), (2, "sports"), (3, "world"), ...],
def read_majority_voting() -> dict[str, list[tuple[int, str]]]:
    results = {prompt_name: [] for prompt_name, _, _ in _PROMPTS}
    current_prompt: str | None = None
    current_article_num: int | None = None

    for line in MAJORITY_VOTING_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = _PROMPT_HEADER_RE.match(line)
        if match:
            current_prompt = match.group(1)
            current_article_num = None
            continue
        match = _ARTICLE_HEADER_RE.match(line)
        if match and current_prompt is not None:
            current_article_num = int(match.group(1))
            continue
        match = _MAJORITY_VOTE_LINE_RE.match(line)
        if match and current_prompt is not None and current_article_num is not None:
            results[current_prompt].append((current_article_num, normalize(match.group(1))))
    return results


def percent(correct: int, total: int) -> str:
    if total == 0:
        return "0.00"
    return f"{100 * correct / total:.2f}"


def run_part1(client: LLMClient) -> None:
    articles = read_articles(ARTICLES_FILE)
    total = len(articles)
    with CLASSIFICATIONS_FILE.open("w", encoding="utf-8") as out:
        for n, text in articles:
            out.write(f"{n}:\n")
            user = zero_shot.user_prompt(text)
            row_cats: list[str] = []
            for temp in TEMPERATURES:
                raw = _safe_complete(client, zero_shot.SYSTEM, user, temp, f"article {n} @ temp {temp}")
                if raw is None:
                    raw = "[blocked by Azure content filter]"
                    category = "unknown"
                else:
                    category = normalize(raw)
                out.write(f"{temp}: {category}\n")
                out.write(f"LLM response: {_one_line(raw)}\n")
                row_cats.append(category)
            print(f"Article {n}/{total}: {row_cats}")
    print(f"Wrote {CLASSIFICATIONS_FILE}")


_PROMPTS = (
    ("zero-shot", zero_shot, None),
    ("few-shot", few_shot, None),
    ("chain-of-thought", chain_of_thought, chain_of_thought.parse_final_label),
)


def run_part2(client: LLMClient) -> None:
    articles = read_articles(ARTICLES_FILE)
    total = len(articles)
    with MAJORITY_VOTING_FILE.open("w", encoding="utf-8") as out:
        for prompt_name, module, parser in _PROMPTS:
            out.write(f"{prompt_name}:\n")
            print(f"--- {prompt_name} ---")
            for n, text in articles:
                out.write(f"{n}:\n")
                user = module.user_prompt(text)
                votes: list[str] = []
                for call_i in range(1, NUM_CALLS + 1):
                    raw = _safe_complete(
                        client,
                        module.SYSTEM,
                        user,
                        PART2_TEMPERATURE,
                        f"{prompt_name} article {n} call {call_i}",
                    )
                    if raw is None:
                        vote = "unknown"
                    elif parser is not None:
                        vote = normalize(parser(raw))
                    else:
                        vote = normalize(raw)
                    votes.append(vote)
                    out.write(f"call {call_i}: {vote}\n")
                vote_result = majority(votes)
                out.write(f"majority vote: {vote_result}\n")
                print(f"Article {n}/{total}: {votes} -> {vote_result}")
    print(f"Wrote {MAJORITY_VOTING_FILE}")

def Per_category_metrics(best_temperature: int, temp_correct: dict[int, int], classifications: list[tuple[int, list[str]]]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    best_temperature_index = TEMPERATURES.index(best_temperature)
    predicted_per_category = {category: 0 for category in CATEGORIES}
    actual_per_category = {category: 0 for category in CATEGORIES}
    correct_per_category = {category: 0 for category in CATEGORIES}

    for n, categories in classifications:
        predicted_category = categories[best_temperature_index]
        actual_category = get_correct_category(n)
        actual_per_category[actual_category] += 1
        if predicted_category in predicted_per_category:
            predicted_per_category[predicted_category] += 1
            if predicted_category == actual_category:
                correct_per_category[predicted_category] += 1
    
    return predicted_per_category, actual_per_category, correct_per_category


def run_part3() -> None:
    classifications = read_classifications()
    temp_correct = {temp: 0 for temp in TEMPERATURES}
    prompt_correct = {prompt_name: 0 for prompt_name, _, _ in _PROMPTS}
    total_correct = 0
    majority_correct = 0

    with STATISTICS_FILE.open("w", encoding="utf-8") as out:
        for n, categories in classifications:
            correct_flags = [correct_category(n, category) for category in categories]
            num_correct = sum(correct_flags)
            majority_result = "correct" if num_correct >= 2 else "incorrect"

            out.write(f"{n}:\n")
            out.write(f"number correct: {num_correct}\n")
            out.write(f"majority voting: {majority_result}\n")

            if majority_result == "correct":
                majority_correct += 1
            for temp, is_correct in zip(TEMPERATURES, correct_flags):
                if is_correct:
                    temp_correct[temp] += 1
                    total_correct += 1

        best_temperature = max(temp_correct, key=temp_correct.get)

        out.write("summary\n")
        for temp in TEMPERATURES:
            out.write(f"percent correct temperature = {temp}: {percent(temp_correct[temp], len(classifications))}\n")
        out.write(f"percent correct total: {percent(total_correct, len(classifications) * len(TEMPERATURES))}\n")
        out.write(f"percent correct majority voting: {percent(majority_correct, len(classifications))}\n")

        majority_voting = read_majority_voting()
        for prompt_name, articles in majority_voting.items():
            for n, category in articles:
                if correct_category(n, category):
                    prompt_correct[prompt_name] += 1

        best_prompt = max(prompt_correct, key=prompt_correct.get)

        out.write("prompt comparison\n")
        out.write(f"percent correct zero-shot majority voting: {percent(prompt_correct['zero-shot'], len(classifications))}\n")
        out.write(f"percent correct few-shot majority voting: {percent(prompt_correct['few-shot'], len(classifications))}\n")
        out.write(f"percent correct chain-of-thought majority voting: {percent(prompt_correct['chain-of-thought'], len(classifications))}\n")
        out.write(f"best prompt: {best_prompt}\n")

        out.write(f"best configuration: {best_temperature}\n")

        predicted_per_category, actual_per_category, correct_per_category = Per_category_metrics(best_temperature, temp_correct, classifications)

        total_f1 = 0.0
        for category in CATEGORIES:
            precision_value = correct_per_category[category] / predicted_per_category[category] if predicted_per_category[category] != 0 else 0.0
            recall_value = correct_per_category[category] / actual_per_category[category] if actual_per_category[category] != 0 else 0.0
            f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if precision_value + recall_value != 0 else 0.0

            out.write(f"precision {category}: {100 * precision_value:.2f}\n")
            out.write(f"recall {category}: {100 * recall_value:.2f}\n")
            out.write(f"f1 {category}: {100 * f1_value:.2f}\n")
            total_f1 += f1_value

        out.write(f"average f1: {100 * total_f1 / len(CATEGORIES):.2f}\n")
        

def main() -> None:
    client = LLMClient()
    run_part1(client)
    run_part2(client)
    run_part3()


if __name__ == "__main__":
    main()
