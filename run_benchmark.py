"""
run_benchmark.py — Benchmark completo: itera sobre imagens do BRESSAY,
transcreve via Claude e calcula CER/WER contra o ground truth.
Salva resultados individuais + médias em results.json.

Uso:
  python run_benchmark.py --imagens dataset/imagens/ --ground-truths dataset/textos/
  python run_benchmark.py --imagens dataset/imagens/ --ground-truths dataset/textos/ --estrategia zero-shot --limite 10
  python run_benchmark.py --imagens dataset/imagens/ --ground-truths dataset/textos/ --todas-estrategias

Estrutura esperada do BRESSAY:
  imagens/          textos/
    001.png           001.txt
    002.png           002.txt
    ...               ...

O script emparelha imagem ↔ ground truth pelo nome do arquivo (stem).
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os

load_dotenv()

# Importa do evaluate.py
from evaluate import (
    avaliar_imagem,
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
log = logging.getLogger("benchmark")

ESTRATEGIAS: list[str] = ["zero-shot", "one-shot", "few-shot"]


# ---------------------------------------------------------------------------
# Emparelhamento imagem ↔ ground truth
# ---------------------------------------------------------------------------

def emparelhar_arquivos(
    dir_imagens: Path,
    dir_ground_truths: Path,
) -> list[tuple[Path, Path]]:
    """
    Encontra pares (imagem, ground_truth) pelo nome do arquivo.
    Ex: 001.png ↔ 001.txt
    """
    imagens = sorted(
        f for f in dir_imagens.iterdir()
        if f.suffix.lower() in EXTENSOES_IMAGEM
    )

    # Mapeia stems dos ground truths disponíveis
    gts_disponiveis: dict[str, Path] = {}
    for f in dir_ground_truths.iterdir():
        if f.suffix.lower() == ".txt":
            gts_disponiveis[f.stem] = f

    pares: list[tuple[Path, Path]] = []
    sem_gt: list[str] = []

    for img in imagens:
        gt = gts_disponiveis.get(img.stem)
        if gt:
            pares.append((img, gt))
        else:
            sem_gt.append(img.name)

    if sem_gt:
        log.warning(
            "%d imagens sem ground truth correspondente: %s",
            len(sem_gt),
            ", ".join(sem_gt[:5]) + ("..." if len(sem_gt) > 5 else ""),
        )

    log.info("Pares encontrados: %d imagens com ground truth", len(pares))
    return pares


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

async def executar_benchmark(
    pares: list[tuple[Path, Path]],
    estrategia: str = "few-shot",
    api_key: Optional[str] = None,
    delay_entre_chamadas: float = 1.0,
) -> dict:
    """
    Executa avaliação em todos os pares e retorna resultado consolidado.
    Inclui delay entre chamadas para respeitar rate limits da API.
    """
    resultados_individuais: list[dict] = []
    erros: list[dict] = []
    total = len(pares)

    log.info("=== Benchmark: %d imagens | estratégia: %s ===", total, estrategia)
    inicio_total = time.perf_counter()

    for i, (img, gt_path) in enumerate(pares, 1):
        log.info("--- [%d/%d] %s ---", i, total, img.name)

        gt_texto = gt_path.read_text(encoding="utf-8")

        try:
            resultado = await avaliar_imagem(
                img, gt_texto,
                estrategia=estrategia,
                api_key=api_key,
            )
            resultado["status"] = "ok"
            resultados_individuais.append(resultado)

        except Exception as e:
            log.error("Erro em %s: %s", img.name, e)
            erros.append({
                "arquivo": str(img),
                "status": "erro",
                "erro": str(e),
            })

        # Delay para rate limit (exceto na última iteração)
        if i < total and delay_entre_chamadas > 0:
            await asyncio.sleep(delay_entre_chamadas)

    duracao_total = time.perf_counter() - inicio_total

    # Calcular médias
    cers = [r["cer"] for r in resultados_individuais if r.get("cer") is not None]
    wers = [r["wer"] for r in resultados_individuais if r.get("wer") is not None]

    media_cer = round(sum(cers) / len(cers), 4) if cers else None
    media_wer = round(sum(wers) / len(wers), 4) if wers else None

    # Melhor e pior resultado
    melhor_cer = min(cers) if cers else None
    pior_cer = max(cers) if cers else None
    melhor_wer = min(wers) if wers else None
    pior_wer = max(wers) if wers else None

    resumo = {
        "estrategia": estrategia,
        "total_imagens": total,
        "total_sucesso": len(resultados_individuais),
        "total_erros": len(erros),
        "media_cer": media_cer,
        "media_wer": media_wer,
        "melhor_cer": melhor_cer,
        "pior_cer": pior_cer,
        "melhor_wer": melhor_wer,
        "pior_wer": pior_wer,
        "tempo_total_s": round(duracao_total, 2),
        "tempo_medio_por_imagem_s": round(duracao_total / total, 2) if total else 0,
    }

    log.info("=== RESUMO (%s) ===", estrategia)
    log.info("  CER médio: %.2f%%", (media_cer or 0) * 100)
    log.info("  WER médio: %.2f%%", (media_wer or 0) * 100)
    log.info("  Sucesso: %d/%d | Erros: %d", len(resultados_individuais), total, len(erros))
    log.info("  Tempo total: %.1fs", duracao_total)

    return {
        "resumo": resumo,
        "resultados": resultados_individuais,
        "erros": erros,
    }


async def executar_todas_estrategias(
    pares: list[tuple[Path, Path]],
    api_key: Optional[str] = None,
    delay_entre_chamadas: float = 1.0,
) -> dict:
    """Executa benchmark para zero-shot, one-shot e few-shot."""
    resultados_por_estrategia: dict = {}
    tabela_comparativa: list[dict] = []

    for estrategia in ESTRATEGIAS:
        log.info("\n{'='*60}")
        log.info("ESTRATÉGIA: %s", estrategia.upper())
        log.info("{'='*60}\n")

        resultado = await executar_benchmark(
            pares, estrategia, api_key, delay_entre_chamadas,
        )
        resultados_por_estrategia[estrategia] = resultado

        tabela_comparativa.append({
            "estrategia": estrategia,
            "media_cer": resultado["resumo"]["media_cer"],
            "media_wer": resultado["resumo"]["media_wer"],
            "melhor_cer": resultado["resumo"]["melhor_cer"],
            "pior_cer": resultado["resumo"]["pior_cer"],
            "sucesso": resultado["resumo"]["total_sucesso"],
            "erros": resultado["resumo"]["total_erros"],
            "tempo_total_s": resultado["resumo"]["tempo_total_s"],
        })

    # Tabela comparativa no log
    log.info("\n=== TABELA COMPARATIVA ===")
    log.info("%-12s | %8s | %8s | %8s | %8s", "Estratégia", "CER médio", "WER médio", "Melhor CER", "Pior CER")
    log.info("-" * 60)
    for linha in tabela_comparativa:
        log.info(
            "%-12s | %7.2f%% | %7.2f%% | %7.2f%% | %7.2f%%",
            linha["estrategia"],
            (linha["media_cer"] or 0) * 100,
            (linha["media_wer"] or 0) * 100,
            (linha["melhor_cer"] or 0) * 100,
            (linha["pior_cer"] or 0) * 100,
        )

    return {
        "tabela_comparativa": tabela_comparativa,
        "detalhes": resultados_por_estrategia,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark de transcrição HTR: Claude vs ground truth BRESSAY",
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
        "--output", "-o", type=Path, default=Path("results.json"),
        help="Arquivo de saída JSON (padrão: results.json)",
    )
    parser.add_argument(
        "--estrategia", "-e", default="few-shot",
        choices=ESTRATEGIAS,
        help="Estratégia de prompting (padrão: few-shot)",
    )
    parser.add_argument(
        "--todas-estrategias", "-t", action="store_true",
        help="Executa benchmark com zero-shot, one-shot E few-shot (comparação completa)",
    )
    parser.add_argument(
        "--limite", "-n", type=int, default=None,
        help="Limitar a N imagens (útil para testes rápidos)",
    )
    parser.add_argument(
        "--delay", "-d", type=float, default=1.0,
        help="Delay em segundos entre chamadas à API (padrão: 1.0)",
    )
    parser.add_argument(
        "--api-key", "-k", type=str, default=None,
        help="Chave da API Anthropic (ou use ANTHROPIC_API_KEY no .env)",
    )

    args = parser.parse_args()

    # Validações
    if not args.imagens.is_dir():
        log.error("Diretório de imagens não encontrado: %s", args.imagens)
        sys.exit(1)

    if not args.ground_truths.is_dir():
        log.error("Diretório de ground truths não encontrado: %s", args.ground_truths)
        sys.exit(1)

    # Emparelhar arquivos
    pares = emparelhar_arquivos(args.imagens, args.ground_truths)

    if not pares:
        log.error("Nenhum par imagem/ground-truth encontrado!")
        sys.exit(1)

    # Aplicar limite
    if args.limite and args.limite < len(pares):
        log.info("Limitando a %d imagens (de %d disponíveis)", args.limite, len(pares))
        pares = pares[:args.limite]

    # Executar
    if args.todas_estrategias:
        resultado_final = await executar_todas_estrategias(
            pares, args.api_key, args.delay,
        )
    else:
        resultado_benchmark = await executar_benchmark(
            pares, args.estrategia, args.api_key, args.delay,
        )
        resultado_final = resultado_benchmark

    # Salvar
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(resultado_final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Resultados salvos em %s", args.output)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
