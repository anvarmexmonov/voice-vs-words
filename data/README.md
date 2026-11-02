# Data

**No audio is redistributed in this repository.** Download each dataset from its source and accept its licence.

| Dataset | Used for | Source | Licence |
|---|---|---|---|
| EMIS | Main bench (Stages 1–3.0) | https://zenodo.org/records/19207001 | GPL-3.0 |
| CREMA-D | Early encoder check (neutral words) | https://github.com/CheyneyComputerScience/CREMA-D | ODbL |

Expected layout:

```
data/mixed/synthetic/EMIS/          # audio + labels.csv (words_say, voice_says, relation, sentence_key)
results/exp1/folds.csv              # 5 folds grouped by sentence_key
```

`labels.csv` is derived from the EMIS file names: `{id}_{text_emo}_{explicit/implicit}_{voice_emo}_{speaker}_{tts}`.
