# HW1: LLM-Based News Article Classification

Classify 30 AG News snippets into four categories using Azure OpenAI `gpt-4.1-mini`.

## Overview

`hw1.py` reads 30 AG News article snippets and classifies each one into a single
lowercase category: `world`, `sports`, `business`, or `sci/tech` (with `unknown` as a
fallback when a response cannot be mapped). It runs three parts in sequence, and a single
invocation of `python hw1.py` produces all three graded output files:

- `classifications.txt` — Part 1 results.
- `majority_voting.txt` — Part 2 results.
- `statistics.txt` — Part 3 metrics.

The model returns free-form text. The code normalizes each raw response to one of the five
allowed labels before scoring, so the category written to every output file is always the
normalized label, not the raw model text.

## Features

- **Part 1 — Classification.** Each article is classified three times, once at each
  temperature `0`, `0.5`, and `0.9`, using a zero-shot prompt.
- **Part 2 — Majority voting and prompt engineering.** Each article is classified five
  times per prompt technique at a fixed temperature of `0.5`. A label wins by majority when
  it receives at least 3 of 5 votes; otherwise the result is recorded as `no majority`.
  Three prompt techniques are compared: `zero-shot`, `few-shot` (four examples, one per
  category), and `chain-of-thought` (step-by-step reasoning with the final label parsed
  from the response).
- **Part 3 — Statistics.** Reports per-article correctness, per-temperature accuracy,
  overall accuracy, majority-voting accuracy, a prompt comparison with the best prompt, and
  per-category precision, recall, and F1 computed at the single best-performing Part 1
  temperature.
- **Response normalization.** Free-form model output is reduced to one of the five allowed
  labels via case-folding and punctuation stripping.
- **Graceful content-filter handling.** A response blocked by the Azure content filter is
  caught and recorded as `unknown` instead of crashing the run.
- **OS-agnostic paths.** All file access uses `pathlib`, so the code runs on Windows,
  macOS, and Linux without changes.

## Tech Stack

- **Python 3.10 or higher** (uses modern type-hint syntax such as `list[tuple[int, str]]`).
- **openai 1.30.0** — the OpenAI Python SDK, pointed at the Azure OpenAI endpoint via
  `base_url` and using the `gpt-4.1-mini` deployment.
- **datasets 3.0.1** — HuggingFace Datasets, used by `prepare_dataset.py` to download the
  AG News test set.
- **httpx 0.27.2** — HTTP client dependency.
- **python-dotenv 1.0.1** — loads Azure credentials from a local `.env` file.

All versions are pinned in `requirements.txt`.

## Project Structure

```
hw_1/
  hw1.py                 Main entry point. Runs Parts 1-3 and writes the three output files.
  llm_client.py          Thin wrapper around the OpenAI SDK; reads Azure config from the environment.
  prompts/
    __init__.py          Exposes the three prompt modules.
    zero_shot.py         Zero-shot system and user prompts.
    few_shot.py          Few-shot prompt with four examples, one per category.
    chain_of_thought.py  Chain-of-thought prompt plus the final-label parser.
  prepare_dataset.py     Provided. Run once to download AG News and create articles.txt.
  gt.py                  Provided. Ground truth: correct_category(num, category) and get_correct_category(num).
  requirements.txt       Pinned dependencies.
  .env.example           Template for the required Azure environment variables.
  articles.txt           Generated input (30 lines of "<n>: <text>"). Not committed.
  classifications.txt    Part 1 output (generated).
  majority_voting.txt    Part 2 output (generated).
  statistics.txt         Part 3 output (generated).
```

## Installation

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Generate the input dataset once. This downloads AG News from HuggingFace and writes
`articles.txt` to the current directory:

```bash
python prepare_dataset.py
```

## Configuration

Azure OpenAI credentials are read from the environment and are never hardcoded in source.
Copy the template and fill in your own values:

```bash
# Windows PowerShell:
Copy-Item .env.example .env
# macOS / Linux:
# cp .env.example .env
```

Then edit `.env`:

```
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
```

`AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are required; the client raises a clear
error if either is missing. `AZURE_OPENAI_DEPLOYMENT` is optional and defaults to
`gpt-4.1-mini`. The `.env` file is listed in `.gitignore` and must never be committed.

## Usage

Run the full pipeline in one invocation:

```bash
python hw1.py
```

This writes all three output files to the current directory:

- `classifications.txt` — for each article, three blocks (`<temperature>: <category>`
  followed by the raw `LLM response:`), one per temperature.
- `majority_voting.txt` — for each prompt technique and article, five `call N:` lines
  followed by a `majority vote:` line.
- `statistics.txt` — per-article correctness, accuracy summaries, prompt comparison, and
  per-category precision, recall, and F1.

## How It Works

1. **Read input.** `read_articles` parses `articles.txt` into `(number, text)` pairs,
   ignoring blank lines and surrounding whitespace.
2. **Classify and normalize.** `LLMClient.complete` sends a system and user prompt to the
   model. Each raw response passes through `normalize`, which case-folds the text, strips
   punctuation, and maps it to one of `world`, `sports`, `business`, `sci/tech`, or
   `unknown`. For the chain-of-thought technique, the final label is first extracted from
   the longer response before normalization.
3. **Vote.** In Part 2, `majority` counts the five votes per article and returns a label
   only when it has at least three votes; otherwise it returns `no majority`.
4. **Score.** Part 3 reads back `classifications.txt` and `majority_voting.txt`, compares
   each label against the ground truth in `gt.py`, and writes the statistics. Per-category
   precision, recall, and F1 are computed at the single best-performing Part 1 temperature,
   applied consistently across all four categories. All percentages are formatted to two
   decimal places, and a zero denominator is reported as `0.00`.

