#!/usr/bin/env python
r"""
run_all.py — Orquestrador end-to-end do pipeline HSI.

Roda a cadeia completa e salva TODOS os resultados (figuras, tabelas, JSON e,
opcionalmente, o HTML de cada notebook) em runs/<timestamp>/, com um log e um
SUMMARY.md pronto para leitura de manhã.

Filosofia: os NOTEBOOKS continuam sendo a fonte de verdade (ótimos para teste
interativo). Aqui eles são apenas EXECUTADOS headless via nbconvert — nada de
lógica duplicada. Assim você mantém o fluxo segregado E ganha o end-to-end.

Uso (rode com o python do .venv):
  python run_all.py                      # só modelagem (NB04,06,07,08), usa caches
  python run_all.py --fresh              # limpa caches -> recalcula do zero
  python run_all.py --upstream           # inclui bronze -> silver -> master_raw (pesado)
  python run_all.py --upstream --fresh --html   # OVERNIGHT: tudo do zero + HTML
  python run_all.py --extra              # inclui também NB01-03 e o NB05 (baseline, lento)

Ex. overnight no Windows:
  .\.venv\Scripts\python.exe run_all.py --upstream --fresh --html
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable  # usa o mesmo interpretador (o do .venv, se você o chamou)
GOLD = ROOT / "data" / "gold level" / "features_extraidas"

# Cadeia canônica de modelagem (ordem importa: 06->07->08).
NOTEBOOKS = [
    "04_preprocessing_effects",
    "06_feature_selection_k15",
    "07_hyperparameter_tuning",
    "08_modelo_final",
]
# Exploração/baseline opcionais (--extra). NB05 é o baseline pixel-level (lento).
EXTRA_ANTES = ["01_raw_visual_inspection", "02_band_removal_analysis", "03_roi_validation"]
EXTRA_DEPOIS = ["05_baseline_model_analysis"]

# Caches que, se apagados (--fresh), forçam recálculo do zero.
CACHES = (
    ["resultados_notebook06_roi.csv", "resultados_notebook08_roi.csv"]
    + [f"config_pre_{p}.json" for p in ("MSC", "RAW", "SavGol", "ZScore")]
)


def _run(cmd, log):
    log.write(f"\n{'='*70}\n$ {' '.join(str(c) for c in cmd)}\n{'='*70}\n")
    log.flush()
    return subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser(description="Orquestrador end-to-end do pipeline HSI.")
    ap.add_argument("--upstream", action="store_true",
                    help="roda pipeline_completa.py (bronze -> silver -> master_raw)")
    ap.add_argument("--metadata", action="store_true",
                    help="regenera data/metadata.csv (revise os rotulos depois!)")
    ap.add_argument("--fresh", action="store_true", help="limpa caches p/ recalculo")
    ap.add_argument("--html", action="store_true", help="exporta cada notebook p/ HTML")
    ap.add_argument("--extra", action="store_true",
                    help="inclui NB01-03 (exploracao) e NB05 (baseline pixel-level, lento)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="timeout por notebook em segundos (padrao 1800)")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "runs" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    logpath = outdir / "run.log"

    nbs = list(NOTEBOOKS)
    if args.extra:
        nbs = EXTRA_ANTES + nbs + EXTRA_DEPOIS

    steps = []  # (nome, status, segundos)
    print(f"[run_all] saida em: {outdir}")

    with open(logpath, "w", encoding="utf-8") as log:

        def stage(nome, cmd):
            t = time.time()
            rc = _run(cmd, log)
            dt = time.time() - t
            status = "OK" if rc == 0 else f"FALHOU(rc={rc})"
            steps.append((nome, status, dt))
            print(f"  [{'OK ' if rc == 0 else 'ERR'}] {nome}  ({dt:.0f}s)")
            return rc

        if args.metadata:
            stage("metadata (build_metadata.py)", [PY, "src/build_metadata.py"])

        if args.upstream:
            stage("upstream (pipeline_completa.py)", [PY, "pipeline_completa.py"])

        if args.fresh:
            apagados = [c for c in CACHES if (GOLD / c).exists()]
            for c in apagados:
                (GOLD / c).unlink()
            log.write(f"\n[fresh] caches apagados: {apagados}\n")
            print(f"  [OK ] caches limpos ({len(apagados)})")

        for nb in nbs:
            path = ROOT / "notebooks" / f"{nb}.ipynb"
            if not path.exists():
                continue
            rc = stage(f"exec {nb}", [
                PY, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                f"--ExecutePreprocessor.timeout={args.timeout}",
                "--ExecutePreprocessor.kernel_name=python3", str(path),
            ])
            if args.html and rc == 0:
                _run([PY, "-m", "nbconvert", "--to", "html",
                      "--output-dir", str(outdir), str(path)], log)

    # Snapshot dos artefatos gerados -> runs/<ts>/
    figs = outdir / "figuras"
    figs.mkdir(exist_ok=True)
    n_png = 0
    if GOLD.exists():
        for png in GOLD.glob("*.png"):
            shutil.copy2(png, figs / png.name)
            n_png += 1
        for j in GOLD.glob("*.json"):
            shutil.copy2(j, outdir / j.name)

    _write_summary(outdir, steps, n_png)
    print(f"[run_all] concluido. Resumo: {outdir / 'SUMMARY.md'}")


def _write_summary(outdir, steps, n_png):
    L = [f"# Run {outdir.name}", ""]
    total = sum(dt for _, _, dt in steps)
    falhas = [n for n, s, _ in steps if s != "OK"]
    L.append(f"**Status geral:** {'TODAS OK' if not falhas else 'FALHAS: ' + ', '.join(falhas)}  ")
    L.append(f"**Tempo total:** {total/60:.1f} min  |  **Figuras:** {n_png}")
    L.append("\n## Etapas")
    for nome, status, dt in steps:
        L.append(f"- `{status:14s}` {nome} ({dt:.0f}s)")

    resumo = outdir / "resumo_modelo_final.json"
    if resumo.exists():
        d = json.loads(resumo.read_text(encoding="utf-8"))
        L.append("\n## Resultado final (nivel de amostra)")
        L.append(f"Protocolo: {d.get('protocolo')} | N={d.get('n_amostras')} "
                 f"({d.get('n_resistentes')} resistentes)\n")
        L.append("| Pre-proc | F1 | ROC-AUC |")
        L.append("|---|---|---|")
        for k, v in d.get("resultados_cv", {}).items():
            L.append(f"| {k} | {v['F1']:.3f} | {v['AUC']:.3f} |")
        p = d.get("permutacao_melhor")
        if p:
            L.append(f"\n**Permutacao ({p['pre_proc']}):** AUC={p['auc_observado']:.3f}, "
                     f"p={p['p_valor']:.3f} "
                     f"({'nao ' if p['p_valor'] > 0.05 else ''}distinguivel do acaso)")

    L.append("\n## Artefatos")
    L.append("- `run.log` — log completo de execucao")
    L.append("- `figuras/` — todas as figuras (.png)")
    L.append("- `*.json` — resumos (resumo_modelo_final.json etc.)")
    L.append("- `*.html` — notebooks executados (se --html)")
    (outdir / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
