# Run 20260711_160137

**Status geral:** TODAS OK  
**Tempo total:** 6.0 min  |  **Figuras:** 15

## Etapas
- `OK            ` metadata (build_metadata.py) (0s)
- `OK            ` upstream (pipeline_completa.py) (152s)
- `OK            ` exec 04_preprocessing_effects (16s)
- `OK            ` exec 06_feature_selection_k15 (44s)
- `OK            ` exec 07_hyperparameter_tuning (78s)
- `OK            ` exec 08_modelo_final (70s)

## Resultado final (nivel de amostra)
Protocolo: StratifiedKFold k=3 — nível de amostra | N=19 (3 resistentes)

| Pre-proc | F1 | ROC-AUC |
|---|---|---|
| MSC | 0.250 | 0.688 |
| RAW | 0.286 | 0.562 |
| SavGol | 0.267 | 0.521 |
| ZScore | 0.267 | 0.521 |

**Permutacao (RAW):** AUC=0.562, p=0.368 (nao distinguivel do acaso)

## Artefatos
- `run.log` — log completo de execucao
- `figuras/` — todas as figuras (.png)
- `*.json` — resumos (resumo_modelo_final.json etc.)
- `*.html` — notebooks executados (se --html)