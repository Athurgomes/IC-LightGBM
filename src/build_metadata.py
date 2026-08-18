"""
build_metadata.py — Fonte única de verdade dos rótulos das amostras

Substitui a regra frágil `1 if "ATCC" in nome else 0` (antes embutida em
build_dataset.py) por um arquivo de metadados auditável e editável à mão.

O script varre a pasta bronze e extrai, do PRÓPRIO nome de cada amostra:
  - sample_id          : nome completo da pasta (chave de junção com os CSVs)
  - data / hora        : do sufixo `AAMMDD-HHMMSS` no nome
  - ordem_aquisicao    : posição na sessão de imageamento (1 = primeira)
  - tipo_cepa          : ATCC_referencia (prefixo ATCC) ou isolado_clinico

O FENÓTIPO (resistente/sensível) é pré-preenchido pela heurística histórica
(ATCC = resistente) APENAS como ponto de partida, e marcado em `fonte_rotulo`
como 'heuristica_ATCC_VERIFICAR'. Ele DEVE ser conferido contra os exames
microbiológicos do laboratório e corrigido manualmente no CSV — o pipeline
de modelagem passa a ler o rótulo daqui, não do nome do arquivo.

⚠️  CONFOUND CONHECIDO: neste lote, as 3 amostras resistentes são exatamente
    as 3 cepas ATCC e foram as 3 ÚLTIMAS imagens da sessão (ver ordem_aquisicao).
    As colunas tipo_cepa e ordem_aquisicao existem justamente para tornar esse
    confundimento explícito e testável (ver teste de permutação / Limitações).

Uso:
    python src/build_metadata.py
Gera: data/metadata.csv
"""

import csv
import re
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BRONZE_DIR = ROOT_DIR / "data" / "bronze level" / "hsi_original"
OUTPUT_CSV = ROOT_DIR / "data" / "metadata.csv"

# Sufixo de timestamp no nome da pasta, ex.: "ATCC13_240506-161053" -> "240506-161053"
_TIMESTAMP_RE = re.compile(r"(\d{6})-(\d{6})$")


def _parse_timestamp(sample_id: str):
    """Extrai datetime do sufixo AAMMDD-HHMMSS; retorna None se não casar."""
    m = _TIMESTAMP_RE.search(sample_id)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}-{m.group(2)}", "%y%m%d-%H%M%S")
    except ValueError:
        return None


def construir_metadata(bronze_dir: Path = BRONZE_DIR):
    """Monta a lista de registros de metadados a partir das pastas em bronze."""
    if not bronze_dir.exists():
        raise FileNotFoundError(f"Pasta bronze não encontrada: {bronze_dir}")

    amostras = sorted(p.name for p in bronze_dir.iterdir() if p.is_dir())
    if not amostras:
        raise RuntimeError(f"Nenhuma amostra encontrada em {bronze_dir}")

    registros = []
    for sample_id in amostras:
        ts = _parse_timestamp(sample_id)
        is_atcc = "ATCC" in sample_id.upper()
        registros.append({
            "sample_id": sample_id,
            "tipo_cepa": "ATCC_referencia" if is_atcc else "isolado_clinico",
            # Rótulo SEMENTE pela heurística — deve ser verificado no laboratório.
            "target": 1 if is_atcc else 0,
            "fenotipo": "resistente" if is_atcc else "sensivel",
            "fonte_rotulo": "heuristica_ATCC_VERIFICAR",
            "data": ts.strftime("%Y-%m-%d") if ts else "",
            "hora": ts.strftime("%H:%M:%S") if ts else "",
            "_ts": ts,  # auxiliar para ordenação; removido na escrita
        })

    # ordem_aquisicao: 1 = primeira imagem da sessão. Amostras sem timestamp
    # vão para o fim, preservando ordem alfabética estável.
    registros.sort(key=lambda r: (r["_ts"] is None, r["_ts"] or datetime.max, r["sample_id"]))
    for i, r in enumerate(registros, start=1):
        r["ordem_aquisicao"] = i
        del r["_ts"]

    return registros


def salvar_metadata(registros, output_csv: Path = OUTPUT_CSV):
    colunas = [
        "sample_id", "target", "fenotipo", "tipo_cepa",
        "fonte_rotulo", "data", "hora", "ordem_aquisicao",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        for r in registros:
            writer.writerow({c: r[c] for c in colunas})
    return output_csv


def main():
    registros = construir_metadata()
    caminho = salvar_metadata(registros)

    n_total = len(registros)
    n_pos = sum(r["target"] for r in registros)
    print(f"[OK] {caminho} gerado — {n_total} amostras ({n_pos} resistentes / "
          f"{n_total - n_pos} sensíveis)")
    print("\nOrdem de aquisição (conferir o confound resistente x fim-da-sessão):")
    for r in registros:
        marca = "  <-- resistente" if r["target"] == 1 else ""
        print(f"  {r['ordem_aquisicao']:2d}. {r['sample_id']:28s} "
              f"[{r['tipo_cepa']}] {r['hora']}{marca}")
    print("\n[ATENCAO] Revise a coluna 'target'/'fenotipo' contra os exames "
          "microbiologicos antes de modelar (fonte_rotulo = VERIFICAR).")


if __name__ == "__main__":
    main()
