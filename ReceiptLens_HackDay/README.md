# ReceiptLens — Hack Day Theme 3B

**Image → structured receipt JSON → Gemma 4 Vision LoRA fine-tuning → measured baseline comparison.**

A **pre-built, theme-specific** project designed for the [HacktoberFest Open-Source AI Hack Day](https://github.com/mithun-srinivas/HacktoberFest-HackDay). The challenge requires fine-tuning Gemma 4 on **ICDAR 2019 SROIE** receipts to extract four fields: `company`, `date`, `address`, and `total`.

> **Reproducibility / honesty:** The source, notebook, parser, and evaluation tests were built before the event. The package contains **no downloaded AI model, LoRA weights, receipts, or fabricated evaluation results**. Local tests pass; **actual Gemma 4 inference and LoRA training require a Colab GPU** and have not been executed in the packaging environment.

## Download / start

1. Download the `ReceiptLens_HackDay_Complete.zip` archive.
2. Open [Google Colab](https://colab.research.google.com/) and upload `notebooks/ReceiptLens_Colab_Runbook.ipynb` (extract the archive first).
3. Select **Runtime → Change runtime type → T4 GPU**; connect.
4. Run notebook **Step 1 (GPU check)**.
5. In notebook **Step 2**, upload the **same ZIP**. It extracts the source code and tests into Colab automatically.
6. Run remaining notebook cells **sequentially**. Download `receiptlens_results.zip` after training.

**Why two uploads?** The notebook needs a GPU runtime, and its source modules are included in the ZIP. Colab does not automatically mount the contents of the ZIP onto `sys.path` when you upload the notebook alone.

## Notebook stages

| Order | Output | What must be true |
|---|---|---|
| 1–4 | Colab GPU + tests | CUDA available and local tests pass |
| 5–6 | SROIE splits | Real receipt images and annotations; disjoint train/validation/test |
| 7–10 | Base Gemma predictions | Image-to-text inference works; baseline CSV saved **before** training |
| 11–14 | LoRA adapters | Vision SFT training completes; adapters saved immediately |
| 15–16 | Comparison table/chart | Same fixed official-test subset for both models |
| 17 | Gradio web demo | Image upload returns fine-tuned JSON; baseline only when truly available |
| 18–20 | Result export + submission | Notebook, README, test metrics, checkpoint, demo recorded |

## Project tree

```text
ReceiptLens_HackDay/
├── notebooks/
│   └── ReceiptLens_Colab_Runbook.ipynb
├── src/receiptlens/
│   ├── config.py       # experiment reproducibility
│   ├── data.py         # load and split SROIE; format image examples
│   ├── prompts.py      # shared inference/training instruction
│   ├── vision.py       # Gemma 4 loading, inference, LoRA adapters
│   ├── train.py        # vision fine-tuning trainer
│   ├── extract.py      # safe JSON parsing / schema validation
│   ├── metrics.py      # normalized field accuracy, evaluation
│   ├── reports.py      # CSV exports, comparison chart
│   ├── demo.py         # Gradio UI, no fake base-model answers
│   └── smoke.py        # model-free quick test
├── tests/              # local unit tests
├── docs/               # technical plan, troubleshooting, judge pitch
├── outputs/            # empty; results saved here in Colab
├── requirements-local.txt
└── pyproject.toml
```

## Tested locally (no GPU required)

From an extracted local project folder:

```bash
python -m pip install -r requirements-local.txt
python -m pip install -e .
python -m pytest -q
python -m receiptlens.smoke
```

These local tests validate dataset formatting, JSON parsing, accuracy scoring, image resizing, and configuration. **They do not claim inference/training succeeded.** Colab installs GPU dependencies using the organizer's pinned training setup.

## Output schema

```json
{
  "company": "ABC HO TRADING",
  "date": "09/01/2019",
  "address": "NO.2&4 JALAN HARMONI 3/2",
  "total": "31.00"
}
```

The example above is illustrative. Model output must be checked against annotation or source image.

## Evaluation

- Use 160 training and 20 validation examples from **SROIE's official training split** as a starting point.
- Reserve 30 examples from **SROIE's official test split** for a locked comparison. Do not tune against this split.
- **Baseline is collected before LoRA is attached.**
- Prediction settings and image preprocessing remain identical across before/after runs.
- Metrics: parseable JSON, strict schema validity, `company`, `date`, `address`, `total` normalized exact-match accuracy, and **all-four-fields** accuracy.
- Text normalization folds case and whitespace; date matching recognizes common day/month/year variants; monetary values compare as decimal amounts. This is a task-specific normalized exact-match measure, not a claim of human-verified correctness.
- Dataset annotations can be imperfect, including reconstructed test records. Inspect notable mismatches manually before drawing conclusions.

## LoRA settings (initial, not optimized)

| Setting | Default |
|---|---|
| Model | `unsloth/gemma-4-E2B-it` |
| Quantization | 4-bit |
| Max receipt image edge | 960 px |
| LoRA rank/alpha | 8 / 8 |
| Steps | 40 |
| Learning rate | 2e-4 |
| Batch / accumulation | 1 / 4 |
| Decoding | Greedy (`do_sample=False`) |

A T4 GPU might still OOM depending on software stack and available VRAM. Lower max image edge to 768 or use smaller LoRA rank for a *new documented experiment*, and repeat the baseline with the changed image settings.

## Honest limitations

- No **standalone** CPU app is bundled because the project is explicitly a GPU vision fine-tuning challenge.
- The free Colab GPU may be unavailable or resource limited. The latest trainer dependencies may change or need the organizer's updated notebook.
- Fine-tuning **may not improve** test accuracy; show actual data, including negative results.
- SROIE receipts are not representative of all locales, handwriting, poorly focused phone photos or Indian tax receipts.
- Gradio `share=True` makes a temporary *public* demo link; do not upload confidential receipts without authorization.
- Open-weight models need correct licenses; SROIE dataset is under CC BY 4.0. See source links and respect attribution.

## References

- [Event problem statement](https://github.com/mithun-srinivas/HacktoberFest-HackDay/blob/main/problem-statements/03_fine_tuning.md)
- [Official organizer starter](https://github.com/mithun-srinivas/HacktoberFest-HackDay/blob/main/templates/03_fine_tuning_starter_kit.ipynb)
- [Official Unsloth Gemma 4 vision example](https://github.com/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb)
- [Hugging Face SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE)
- [SROIE dataset maintainers' documentation](https://github.com/jsdnrs/ICDAR2019-SROIE)
