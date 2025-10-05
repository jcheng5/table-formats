# Table Format Benchmarks

Recreates the "Which Table Format Do LLMs Understand Best?" blog experiment using the [Inspect](https://inspect.aisi.org.uk/llms.txt) evaluation framework. The evaluation feeds 1,000 synthetic employee records to a model in 11 different formats and asks 1,000 numeric lookup questions (salary, age, years of experience, project count) to measure accuracy and token usage. The dataset generation is deterministic so results are reproducible across runs and models.

## Project layout

- `evals/table_formats_eval.py` – Inspect tasks (one per format) with shared dataset/question generation utilities.
- `scripts/run_benchmarks.py` – Helper runner that loops over formats/models and shells out to `inspect eval`.
- `.env` – Provide `OPENAI_API_KEY` (or other provider keys) before running evaluations.

## Prerequisites

Create/activate the existing virtual environment and load API credentials:

```bash
source .venv/bin/activate
set -a && source .env && set +a
```

Verify that Inspect is available:

```bash
inspect --version
```

If that command is not on your `PATH`, use `INSPECT_BIN=.venv/bin/inspect` when running the helper script.

## Running a single format/model

Run an individual evaluation directly with Inspect (example uses OpenAI's `gpt-4.1-mini`):

```bash
inspect eval evals/table_formats_eval.py@table_formats_markdown_kv \
  --model openai/gpt-4.1-mini \
  --log-dir logs/gpt-4.1-mini/markdown-kv
```

Use `--limit 25` for a quick smoke test before running the full 1,000 samples.

## Running the full grid of formats

The helper script executes all (or a chosen subset of) formats across one or more models:

```bash
python scripts/run_benchmarks.py \
  --models openai/gpt-4.1-mini openai/gpt-4.1-nano \
  --formats markdown_kv markdown_table json csv \
  --limit 200 \
  --inspect-args --display plain
```

Omit `--limit` to reproduce the full benchmark. Logs for each run are written under `inspect-logs/<model>/<format>` by default; add `--no-logs` to suppress log files.

## Collecting accuracy & token metrics

Inspect prints aggregate accuracy after each run. Token usage per sample and per run is recorded in the log directories mentioned above (see the `metrics.json` files for structured data). These logs mirror the blog's reporting (accuracy plus usage) and make it easy to compare models side-by-side.

## Extending to new models or formats

- Add additional Inspect model identifiers to the `--models` list or set `INSPECT_EVAL_MODEL` globally.
- To prototype new formats, extend `FORMAT_SPECS`/`FORMAT_ORDER` in `evals/table_formats_eval.py` with a new formatter function. The helper script picks up new tasks automatically if you follow the naming scheme `table_formats_<format>`.

## Safety note

Each full benchmark run issues 11,000 model calls per model. Start with a small `--limit` to validate credentials and expected behaviour before committing to the full cost.
