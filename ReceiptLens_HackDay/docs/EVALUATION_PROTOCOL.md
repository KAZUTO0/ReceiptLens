# ReceiptLens evaluation protocol

## Objective

Measure whether LoRA specialization improves or worsens extraction of four specified fields from receipt **images**, and distinguish formatting improvements from real recognition improvements.

## Locked splits

- Training: shuffled SROIE official `train` samples `0:train_count` (after fixed random seed).
- Validation: immediately following shuffled training samples, used for trial settings and sanity checks.
- Final test: separate official SROIE `test` samples, selected with separate fixed seed; never train or tune on this set.

Report actual sample counts returned by the library; no hard-coded accuracy claims.

## Scoring

Each result records `valid_json` (parseable object found), `valid_schema` (exactly four string keys), and per-field normalized exact matches. `all_four` requires all fields correct.

- **Company/address**: case-insensitive, whitespace-normalized exact match. Punctuation must still match.
- **Date**: recognizes a small set of day-first formats and ISO date; no fuzzy guessing.
- **Total**: strips common currency prefixes, separators, standardizes 2 decimal places; can treat 10 and 10.00 equally.
- **Invalid schema / malformed JSON**: zero credits for all fields; prevents a blank ground-truth field from accidentally scoring as correct.

This is a fast hackathon metric, not full real-world OCR confidence calibration. Address punctuation and vendor formatting can cause lower matches despite semantically similar extracted text; show qualitative examples.

## Statistical caveats

At `test_count=30`, one receipt changes accuracy by **3.33 percentage points**. Avoid claiming statistical significance from small differences. Never select hyperparameters after looking at final-test results. The dataset maintainer notes reconstructed annotations for a handful of test images; manual inspection of surprising mismatches is useful.

## Practical ethical boundaries

Open public or consented test receipts only. Avoid portraying extraction accuracy as reliable for bank statements, tax filings, or expense accounting without further validation.
