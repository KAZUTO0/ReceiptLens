# Troubleshooting and recovery

| Symptom | Explanation / safe next step |
|---|---|
| `CUDA available: False` | Runtime → Change runtime type → T4 GPU; retry only after connected. |
| Colab free GPU unavailable | GPU availability is not guaranteed. Try team member allocation or continue with dataset, scorer and a clear limitation; don't claim training. |
| `No module named unsloth` | Run installation cell **in Colab**, not ordinary CPU notebook; restart runtime and re-run module imports. |
| `transformers` / `trl` incompatibility | Use current organizer's exact setup or official Unsloth Gemma 4 Vision notebook; check install output by removing `%%capture`. |
| CUDA out of memory | Stop, restart, reduce image max edge (960 → 768) and/or LoRA rank; redo base predictions if image preprocessing changes. |
| Model output isn't JSON | Inspect raw output and chat template, reduce ambiguity in prompt, examine if image is legible; don't treat it as correct. |
| `AttributeError` on `disable_adapter` | Base predictions remain cached as `baseline_predictions.csv`; live side-by-side is optional. Never label a tuned result "plain". |
| Loss goes down but scores worsen | Overfitting, small data, poor preprocessing; report results honestly and use *validation* data for further experiments. |
| Results inconsistent | Confirm identical images, seeds, prompt, image preprocessing, inference settings and evaluation code. |
| Unexpected zero accuracy | Check JSON schema, annotation formatting, date/total normalization and whether ground truth matches image. |
| App link disconnects | Gradio share URL works only during active Colab process; re-run demo cell. |
| Colab resets | Re-upload project ZIP, rerun notebook, reload saved adapters from Drive or download. Save model and CSVs promptly. |
| Dataset request fails | Check Colab internet and Hugging Face status; don't substitute random receipts without documenting source / license. |

## Security & responsible use

- `share=True` publishes a reachable URL; prefer synthetic or permissible public examples.
- Dataset `entities` provides labels, not model inputs. Feeding labels to inference causes evaluation leakage.
- Avoid publishing actual people's purchases or merchant/customer personally identifying information.
- Check reuse licensing before publishing trained adapters based on dataset and base model.
