"""
experiment.py — Experimento comparativo zero-shot vs one-shot vs few-shot
Requisito obrigatório do desafio IBM: tabela comparativa de estratégias de prompting.

Roda o mesmo conjunto de imagens BRESSAY com as 3 estratégias, calcula CER/WER
e gera saídas prontas para a apresentação:
  - experiment_results.json  (detalhes completos)
  - experiment_results.csv   (tabela limpa para colar num slide)

Uso:
  python experiment.py -i dataset/imagens/ -g dataset/textos/
  python experiment.py -i dataset/imagens/ -g dataset/textos/ -n 20 --preprocess
  python experiment.py -i dataset/imagens/ -g dataset/textos/ --concorrencia 3

Integra com:
  - evaluate.py  → transcrição via Claude + cálculo CER/WER
  - preprocess.py → pré-processamento opcional das imagens antes da transcrição
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from evaluate import (
    avaliar_imagem,
    calcular_metricas,
    transcrever_imagem,
    EXTENSOES_IMAGEM,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("experiment")

ESTRATEGIAS: list[str] = ["zero-shot", "one-shot", "few-shot"]


# ---------------------------------------------------------------------------
# Emparelhamento imagem ↔ ground truth
# ---------------------------------------------------------------------------

def emparelhar_arquivos(
    dir_imagens: Path,
    dir_ground_truths: Path,
) -> list[tuple[Path, Path]]:
    """Encontra pares (imagem, ground_truth) pelo stem do arquivo."""
    imagens = sorted(
        f for f in dir_imagens.iterdir()
        if f.suffix.lower() in EXTENSOES_IMAGEM
    )

    gts: dict[str, Path] = {
        f.stem: f
        for f in dir_ground_truths.iterdir()
        if f.suffix.lower() == ".txt"
    }

    pares: list[tuple[Path, Path]] = []
    sem_gt: int = 0

    for img in imagens:
        gt = gts.get(img.stem)
        if gt:
            pares.append((img, gt))
        else:
            sem_gt += 1

    if sem_gt:
        log.warning("%d imagens sem ground truth — ignoradas", sem_gt)

    log.info("Pares encontrados: %d", len(pares))
    return pares


# ---------------------------------------------------------------------------
# Pré-processamento opcional (integração com preprocess.py)
# ---------------------------------------------------------------------------

def preprocessar_imagens(
    pares: list[tuple[Path, Path]],
    dir_saida: Path,
) -> list[tuple[Path, Path]]:
    """Pré-processa todas as imagens e retorna novos pares com imagens processadas."""
    from preprocess import preprocessar

    dir_saida.mkdir(parents=True, exist_ok=True)
    novos_pares: list[tuple[Path, Path]] = []

    log.info("Pré-processando %d imagens...", len(pares))
    for img, gt in pares:
        destino = dir_saida / f"{img.stem}.png"
        try:
            preprocessar(
                img, destino,
                pular_binarizacao=True,  # Melhor para LLMs multimodais
            )
            novos_pares.append((destino, gt))
        except Exception as e:
            log.warning("Falha no pré-processamento de %s: %s — usando original", img.name, e)
            novos_pares.append((img, gt))

    log.info("Pré-processamento concluído: %d imagens", len(novos_pares))
    return novos_pares


# ---------------------------------------------------------------------------
# Semáforo para controle de concorrência na API
# ---------------------------------------------------------------------------

async def _avaliar_com_semaforo(
    semaforo: asyncio.Semaphore,
    caminho_imagem: Path,
    ground_truth: str,
    estrategia: str,
    api_key: Optional[str],
    delay: float,
) -> dict:
    """Avalia uma imagem respeitando o limite de concorrência."""
    async with semaforo:
        try:
            resultado = await avaliar_imagem(
                caminho_imagem, ground_truth,
                estrategia=estrategia,
                api_key=api_key,
            )
            resultado["status"] = "ok"
        except Exception as e:
            log.error("Erro em %s [%s]: %s", caminho_imagem.name, estrategia, e)
            resultado = {
                "arquivo": str(caminho_imagem),
                "estrategia": estrategia,
                "status": "erro",
                "erro": str(e),
                "cer": None,
                "wer": None,
            }
        # Delay entre chamadas para rate limit
        if delay > 0:
            await asyncio.sleep(delay)
        return resultado


# ---------------------------------------------------------------------------
# Execução do experimento por estratégia
# ---------------------------------------------------------------------------

async def executar_estrategia(
    pares: list[tuple[Path, Path]],
    estrategia: str,
    api_key: Optional[str] = None,
    concorrencia: int = 1,
    delay: float = 1.0,
) -> dict:
    """Executa uma estratégia inteira e retorna resumo + resultados individuais."""
    log.info("━━━ Estratégia: %s (%d imagens, concorrência=%d) ━━━",
             estrategia.upper(), len(pares), concorrencia)
    inicio = time.perf_counter()

    semaforo = asyncio.Semaphore(concorrencia)
    tarefas = []

    for img, gt_path in pares:
        gt_texto = gt_path.read_text(encoding="utf-8")
        tarefas.append(
            _avaliar_com_semaforo(
                semaforo, img, gt_texto, estrategia, api_key, delay,
            )
        )

    resultados = await asyncio.gather(*tarefas)
    duracao = time.perf_counter() - inicio

    # Separar sucessos e erros
    sucessos = [r for r in resultados if r.get("status") == "ok"]
    erros = [r for r in resultados if r.get("status") == "erro"]

    # Métricas
    cers = [r["cer"] for r in sucessos if r.get("cer") is not None]
    wers = [r["wer"] for r in sucessos if r.get("wer") is not None]

    def _stats(valores: list[float]) -> dict:
        if not valores:
            return {"media": None, "mediana": None, "desvio": None, "min": None, "max": None}
        return {
            "media": round(statistics.mean(valores), 4),
            "mediana": round(statistics.median(valores), 4),
            "desvio": round(statistics.stdev(valores), 4) if len(valores) > 1 else 0.0,
            "min": round(min(valores), 4),
            "max": round(max(valores), 4),
        }

    stats_cer = _stats(cers)
    stats_wer = _stats(wers)

    # Tokens consumidos
    tokens_in = sum(r.get("tokens_entrada", 0) for r in sucessos)
    tokens_out = sum(r.get("tokens_saida", 0) for r in sucessos)

    resumo = {
        "estrategia": estrategia,
        "total_imagens": len(pares),
        "sucesso": len(sucessos),
        "erros": len(erros),
        "cer": stats_cer,
        "wer": stats_wer,
        "tokens_entrada_total": tokens_in,
        "tokens_saida_total": tokens_out,
        "tempo_total_s": round(duracao, 2),
    }

    log.info("  CER: média=%.2f%% mediana=%.2f%% σ=%.2f%%",
             (stats_cer["media"] or 0) * 100,
             (stats_cer["mediana"] or 0) * 100,
             (stats_cer["desvio"] or 0) * 100)
    log.info("  WER: média=%.2f%% mediana=%.2f%% σ=%.2f%%",
             (stats_wer["media"] or 0) * 100,
             (stats_wer["mediana"] or 0) * 100,
             (stats_wer["desvio"] or 0) * 100)
    log.info("  Sucesso: %d/%d | Tempo: %.1fs | Tokens: %d in / %d out",
             len(sucessos), len(pares), duracao, tokens_in, tokens_out)

    return {
        "resumo": resumo,
        "resultados": sucessos,
        "erros": erros,
    }


# ---------------------------------------------------------------------------
# Experimento completo: 3 estratégias + comparação por imagem
# ---------------------------------------------------------------------------

async def executar_experimento(
    pares: list[tuple[Path, Path]],
    api_key: Optional[str] = None,
    concorrencia: int = 1,
    delay: float = 1.0,
) -> dict:
    """Executa as 3 estratégias sequencialmente e monta a tabela comparativa."""
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  EXPERIMENTO COMPARATIVO — zero-shot vs one-shot vs few-shot  ║")
    log.info("╚══════════════════════════════════════════════════════════╝")
    log.info("Imagens: %d | Chamadas totais à API: %d", len(pares), len(pares) * 3)

    inicio_total = time.perf_counter()
    resultados_por_estrategia: dict[str, dict] = {}

    for estrategia in ESTRATEGIAS:
        resultado = await executar_estrategia(
            pares, estrategia, api_key, concorrencia, delay,
        )
        resultados_por_estrategia[estrategia] = resultado

    duracao_total = time.perf_counter() - inicio_total

    # ── Tabela comparativa (para o slide) ──
    tabela: list[dict] = []
    for e in ESTRATEGIAS:
        r = resultados_por_estrategia[e]["resumo"]
        tabela.append({
            "Estratégia": e,
            "CER Médio (%)": round((r["cer"]["media"] or 0) * 100, 2),
            "CER Mediana (%)": round((r["cer"]["mediana"] or 0) * 100, 2),
            "CER Desvio (%)": round((r["cer"]["desvio"] or 0) * 100, 2),
            "CER Melhor (%)": round((r["cer"]["min"] or 0) * 100, 2),
            "CER Pior (%)": round((r["cer"]["max"] or 0) * 100, 2),
            "WER Médio (%)": round((r["wer"]["media"] or 0) * 100, 2),
            "WER Mediana (%)": round((r["wer"]["mediana"] or 0) * 100, 2),
            "Imagens OK": r["sucesso"],
            "Erros": r["erros"],
            "Tokens Entrada": r["tokens_entrada_total"],
            "Tokens Saída": r["tokens_saida_total"],
            "Tempo (s)": r["tempo_total_s"],
        })

    # ── Comparação por imagem (mesma imagem nas 3 estratégias) ──
    comparacao_por_imagem: list[dict] = []
    # Indexar resultados por arquivo
    por_arquivo: dict[str, dict[str, dict]] = {}
    for e in ESTRATEGIAS:
        for r in resultados_por_estrategia[e]["resultados"]:
            arq = r["arquivo"]
            if arq not in por_arquivo:
                por_arquivo[arq] = {}
            por_arquivo[arq][e] = r

    for arq, estrategias in sorted(por_arquivo.items()):
        entrada = {"arquivo": Path(arq).name}
        for e in ESTRATEGIAS:
            if e in estrategias:
                entrada[f"cer_{e}"] = estrategias[e].get("cer")
                entrada[f"wer_{e}"] = estrategias[e].get("wer")
            else:
                entrada[f"cer_{e}"] = None
                entrada[f"wer_{e}"] = None
        # Qual estratégia venceu para esta imagem?
        cers_img = {
            e: estrategias[e]["cer"]
            for e in ESTRATEGIAS
            if e in estrategias and estrategias[e].get("cer") is not None
        }
        if cers_img:
            entrada["melhor_estrategia"] = min(cers_img, key=cers_img.get)
        comparacao_por_imagem.append(entrada)

    # Contagem de vitórias por estratégia
    vitorias: dict[str, int] = {e: 0 for e in ESTRATEGIAS}
    for c in comparacao_por_imagem:
        melhor = c.get("melhor_estrategia")
        if melhor:
            vitorias[melhor] += 1

    # ── Log da tabela final ──
    log.info("")
    log.info("╔═══════════════════════════════════════════════════════════════╗")
    log.info("║              TABELA COMPARATIVA — RESULTADOS                ║")
    log.info("╠══════════════╦══════════╦══════════╦══════════╦═════════════╣")
    log.info("║ Estratégia   ║ CER Méd  ║ WER Méd  ║ CER Med  ║ Vitórias   ║")
    log.info("╠══════════════╬══════════╬══════════╬══════════╬═════════════╣")
    for linha in tabela:
        e = linha["Estratégia"]
        log.info("║ %-12s ║ %6.2f%%  ║ %6.2f%%  ║ %6.2f%%  ║ %4d        ║",
                 e, linha["CER Médio (%)"], linha["WER Médio (%)"],
                 linha["CER Mediana (%)"], vitorias.get(e, 0))
    log.info("╚══════════════╩══════════╩══════════╩══════════╩═════════════╝")
    log.info("Tempo total do experimento: %.1fs", duracao_total)

    # Benchmark ICDAR 2024 para referência
    log.info("")
    log.info("Referência BRESSAY (ICDAR 2024): melhor CER = 2.88%% (linha), 3.75%% (parágrafo)")

    return {
        "experimento": {
            "data": datetime.now(timezone.utc).isoformat(),
            "total_imagens": len(pares),
            "total_chamadas_api": len(pares) * 3,
            "tempo_total_s": round(duracao_total, 2),
            "estrategias": ESTRATEGIAS,
        },
        "tabela_comparativa": tabela,
        "vitorias_por_estrategia": vitorias,
        "comparacao_por_imagem": comparacao_por_imagem,
        "detalhes": {
            e: resultados_por_estrategia[e]["resumo"]
            for e in ESTRATEGIAS
        },
        "benchmark_referencia": {
            "dataset": "BRESSAY (ICDAR 2024)",
            "melhor_cer_linha": 2.88,
            "melhor_cer_paragrafo": 3.75,
            "nota": "Resultados publicados usam modelos HTR dedicados, não LLMs multimodais",
        },
    }


# ---------------------------------------------------------------------------
# Exportação CSV (pronta para colar no slide)
# ---------------------------------------------------------------------------

def salvar_csv(resultado: dict, caminho: Path) -> None:
    """Gera CSV com a tabela comparativa formatada para apresentação."""
    tabela = resultado["tabela_comparativa"]
    vitorias = resultado["vitorias_por_estrategia"]

    # ── Tabela principal ──
    with open(caminho, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")

        # Cabeçalho do experimento
        writer.writerow(["EXPERIMENTO COMPARATIVO — BancaIA HTR"])
        writer.writerow(["Data", resultado["experimento"]["data"]])
        writer.writerow(["Total de imagens", resultado["experimento"]["total_imagens"]])
        writer.writerow([])

        # Tabela comparativa
        writer.writerow([
            "Estratégia",
            "CER Médio (%)",
            "CER Mediana (%)",
            "CER Desvio (%)",
            "CER Melhor (%)",
            "CER Pior (%)",
            "WER Médio (%)",
            "WER Mediana (%)",
            "Imagens OK",
            "Vitórias",
            "Tokens Entrada",
            "Tokens Saída",
            "Tempo (s)",
        ])

        for linha in tabela:
            e = linha["Estratégia"]
            writer.writerow([
                e,
                f'{linha["CER Médio (%)"]:.2f}',
                f'{linha["CER Mediana (%)"]:.2f}',
                f'{linha["CER Desvio (%)"]:.2f}',
                f'{linha["CER Melhor (%)"]:.2f}',
                f'{linha["CER Pior (%)"]:.2f}',
                f'{linha["WER Médio (%)"]:.2f}',
                f'{linha["WER Mediana (%)"]:.2f}',
                linha["Imagens OK"],
                vitorias.get(e, 0),
                linha["Tokens Entrada"],
                linha["Tokens Saída"],
                linha["Tempo (s)"],
            ])

        writer.writerow([])

        # Referência
        ref = resultado["benchmark_referencia"]
        writer.writerow(["REFERÊNCIA BRESSAY (ICDAR 2024)"])
        writer.writerow(["Melhor CER linha (%)", ref["melhor_cer_linha"]])
        writer.writerow(["Melhor CER parágrafo (%)", ref["melhor_cer_paragrafo"]])
        writer.writerow(["Nota", ref["nota"]])

        writer.writerow([])

        # Comparação por imagem (top 10 melhores e piores)
        comparacao = resultado.get("comparacao_por_imagem", [])
        if comparacao:
            writer.writerow(["COMPARAÇÃO POR IMAGEM (top 10 melhor CER few-shot)"])
            writer.writerow([
                "Arquivo",
                "CER zero-shot (%)",
                "CER one-shot (%)",
                "CER few-shot (%)",
                "Melhor",
            ])

            # Ordena por CER few-shot
            ordenado = sorted(
                comparacao,
                key=lambda x: x.get("cer_few-shot") or 999,
            )

            for c in ordenado[:10]:
                writer.writerow([
                    c["arquivo"],
                    f'{(c.get("cer_zero-shot") or 0) * 100:.2f}',
                    f'{(c.get("cer_one-shot") or 0) * 100:.2f}',
                    f'{(c.get("cer_few-shot") or 0) * 100:.2f}',
                    c.get("melhor_estrategia", "—"),
                ])

    log.info("CSV salvo em %s (delimitador: ;  encoding: utf-8-sig para Excel)", caminho)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main_async() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Experimento comparativo zero-shot vs one-shot vs few-shot "
            "(requisito obrigatório do desafio IBM)"
        ),
    )
    parser.add_argument(
        "--imagens", "-i", required=True, type=Path,
        help="Diretório com imagens do BRESSAY",
    )
    parser.add_argument(
        "--ground-truths", "-g", required=True, type=Path,
        help="Diretório com arquivos .txt de ground truth",
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("experiment_results.json"),
        help="Saída JSON (padrão: experiment_results.json)",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("experiment_results.csv"),
        help="Saída CSV para apresentação (padrão: experiment_results.csv)",
    )
    parser.add_argument(
        "--limite", "-n", type=int, default=None,
        help="Limitar a N imagens (útil para testes rápidos)",
    )
    parser.add_argument(
        "--concorrencia", "-c", type=int, default=1,
        help="Chamadas simultâneas à API por estratégia (padrão: 1)",
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=1.0,
        help="Delay entre chamadas à API em segundos (padrão: 1.0)",
    )
    parser.add_argument(
        "--preprocess", "-p", action="store_true",
        help="Pré-processar imagens com preprocess.py antes da transcrição",
    )
    parser.add_argument(
        "--preprocess-dir", type=Path, default=None,
        help="Diretório para imagens pré-processadas (padrão: temp dir)",
    )
    parser.add_argument(
        "--api-key", "-k", type=str, default=None,
        help="Chave da API Anthropic (ou ANTHROPIC_API_KEY no .env)",
    )

    args = parser.parse_args()

    # Validações
    if not args.imagens.is_dir():
        log.error("Diretório de imagens não encontrado: %s", args.imagens)
        sys.exit(1)

    if not args.ground_truths.is_dir():
        log.error("Diretório de ground truths não encontrado: %s", args.ground_truths)
        sys.exit(1)

    # Emparelhar
    pares = emparelhar_arquivos(args.imagens, args.ground_truths)
    if not pares:
        log.error("Nenhum par imagem/ground-truth encontrado!")
        sys.exit(1)

    # Limite
    if args.limite and args.limite < len(pares):
        log.info("Limitando a %d imagens (de %d)", args.limite, len(pares))
        pares = pares[:args.limite]

    # Pré-processamento opcional
    if args.preprocess:
        dir_prep = args.preprocess_dir or Path(tempfile.mkdtemp(prefix="bancaia_prep_"))
        pares = preprocessar_imagens(pares, dir_prep)

    # Executar experimento
    resultado = await executar_experimento(
        pares,
        api_key=args.api_key,
        concorrencia=args.concorrencia,
        delay=args.delay,
    )

    # Salvar JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("JSON salvo em %s", args.output_json)

    # Salvar CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    salvar_csv(resultado, args.output_csv)

    log.info("Experimento concluído com sucesso!")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
