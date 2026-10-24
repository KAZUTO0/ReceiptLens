# Hack Day team playbook — Theme 3B

## Setup (first 20–30 minutes)

- [ ] Every team member can open Google Colab; one member owns the training runtime.
- [ ] Upload the supplied prebuilt notebook, select GPU, run the GPU cell, install packages.
- [ ] Agree project name, GitHub repo URL, team roles, and a single integration owner.
- [ ] Save project ZIP and notebook to an accessible shared location; **do not share private tokens**.
- [ ] Check organizer policy on reused code (confirmed allowed by the participant).

## Minimum winning path

1. **Dataset**: load, inspect an image, confirm four labels and split sizes.
2. **Plain baseline**: run single image, then save original model predictions on fixed test subset.
3. **LoRA**: attach adapter, train, save checkpoint.
4. **After**: run adapted model on exact same fixed test subset; report all metrics.
5. **UI**: upload a real unseen receipt through Gradio, show JSON.
6. **Showcase**: present measured before/after, one example where it helped, one where it didn't.

## Suggested team split (4–6)

- **ML lead (1)** — Colab runtime, model and LoRA experiments. Owns final adapter checkpoint.
- **Data lead (1)** — SROIE inspection, image preprocessing, annotation audits, avoiding leakage.
- **Evaluation lead (1)** — metric correctness, held-out set, qualitative failure analysis.
- **App/release lead (1)** — Gradio, README, repo, slide deck and live demonstration.
- **Optional (2)** — mentor consultation, alternate experiments on separate valid training splits; documentation and QA.

## Experiment protocol

- Choose `train_count`, `val_count`, `test_count`, seed, LoRA steps/rank **before** final-test evaluation.
- Keep one fixed `INSTRUCTION`, same image resize and greedy decoding for both models.
- Save baseline results **before** changing model weights.
- Never use the official `test` set for training or hyperparameter tuning.
- Run at least one smoke inference, and train on small sample before the big run if time is short.
- Never invent performance numbers: an improvement is not guaranteed.

## Judge demonstration (90 sec structure)

1. (0–15 sec) **Problem:** manually copying receipt merchant, date, address and total is slow and error-prone.
2. (15–30 sec) **Method:** Gemma 4 E2B vision receives an image, LoRA specializes it using labeled SROIE receipts.
3. (30–60 sec) **Live demo:** upload a held-out receipt, show JSON and—if supported—live base-vs-LoRA outputs.
4. (60–80 sec) **Evidence:** measured field-level accuracy and valid JSON on same unseen examples.
5. (80–90 sec) **Limitations:** dataset coverage, legibility and annotation ambiguity; next steps.

## Required evidence

- [ ] Working app in Colab; notebook saved
- [ ] Training loss / settings documented
- [ ] Plain and fine-tuned predictions archived, compared on same receipts
- [ ] At least one failure case explained
- [ ] Model adapter folder saved
- [ ] Source project repository / ZIP available
