# Code

| Folder | Script | What it does |
|---|---|---|
| `stage1_encoder_probes/` | `exp1_encoder_probes.py` | Per-layer VOICE / WORDS / PULL probes for eGeMAPS, WavLM, Whisper, Qwen2-Audio encoder on EMIS |
| `stage2_llm_behaviour/` | `exp2_llm_probes.py` | Answers of Qwen2-Audio / AF3 / Voxtral on EMIS + saved hidden states |
| `stage3_steering/` | `exp3_steer.py` | Hand-made voice note on the last token (probe / oracle / random), 7 layers × 3 strengths |

Fold file: `results/exp1/folds.csv` (5 folds grouped by `sentence_key`), reused by every stage.

Hardware used: RTX 5060 Ti 16 GB (Qwen2-Audio LLM body in 8-bit), RTX 5090 for bf16 checks.

```bash
pip install -r requirements.txt
python code/stage1_encoder_probes/exp1_encoder_probes.py --help
```
