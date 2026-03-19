"""
run_bressay.py — Processa imagens do BRESSAY em lote com Google Gemini.
Rate limiting: max 10 req/min (pausa de 6s entre chamadas).
Salva resultados parciais em results.json a cada 10 imagens.

Uso:
  python run_bressay.py --imagens ../dataset/imagens/ --ground-truths ../dataset/textos/
  python run_bressay.py --imagens ../dataset/imagens/ --ground-truths ../dataset/textos/ -n 20
  python run_bressay.py --resumir results.json
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_bressay")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODELO: str = "gemini-2.0-flash"
DELAY_ENTRE_CHAMADAS: float = 6.0  # 6s = max 10 req/min
SALVAR_A_CADA: int = 10  # salvar parcial a cada N imagens
MAX_RETRIES: int = 3

EXTENSOES_IMAGEM: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

PROMPT_TRANSCRICAO: str = (
    "Voce e um especialista em HTR (Handwritten Text Recognition) de redacoes "
    "manuscritas em portugues brasileiro. Transcreva EXATAMENTE o texto manuscrito "
    "desta imagem, preservando ortografia original (incluindo erros), pontuacao "
    "e quebras de paragrafo. Responda APENAS com o texto transcrito, sem "
    "comentarios ou formatacao adicional."
)


# ---------------------------------------------------------------------------
# Emparelhamento imagem <-> ground truth
# ---------------------------------------------------------------------------

def emparelhar(dir_imagens: Path, dir_gts: Path) -> list[tuple[Path, Path]]:
    """Encontra pares (imagem, ground_truth) pelo stem do arquivo."""
    imagens = sorted(
        f for f in dir_imagens.iterdir()
        if f.suffix.lower() in EXTENSOES_IMAGEM
    )
    gts = {f.stem: f for f in dir_gts.iterdir() if f.suffix.lower() == ".txt"}

    pares = []
    sem_gt = 0
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
# Metricas CER/WER
# ---------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Normaliza texto para comparacao."""
    linhas = texto.strip().splitlines()
    linhas = [l.strip() for l in linhas if l.strip()]
    return "\n".join(linhas)


def calcular_cer(ref: str, hyp: str) -> float:
    """Calcula Character Error Rate (Levenshtein / len(ref))."""
    r = normalizar(ref)
    h = normalizar(hyp)
    if not r:
        return 0.0 if not h else 1.0

    # Levenshtein por caractere
    n, m = len(r), len(h)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    return prev[m] / n


def calcular_wer(ref: str, hyp: str) -> float:
    """Calcula Word Error Rate."""
    r = normalizar(ref).split()
    h = normalizar(hyp).split()
    if not r:
        return 0.0 if not h else 1.0

    n, m = len(r), len(h)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    return prev[m] / n


# ---------------------------------------------------------------------------
# Chamada ao Gemini com retry
# ---------------------------------------------------------------------------

async def transcrever_imagem(caminho: Path) -> str:
    """Envia imagem ao Gemini e retorna a transcricao."""
    img = Image.open(caminho)
    model = genai.GenerativeModel(MODELO)

    for tentativa in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content([PROMPT_TRANSCRICAO, img])
            texto = response.text

            # Extrair texto caso venha com markdown
            match = re.search(r"```(?:\w*)\s*([\s\S]*?)```", texto)
            if match:
                texto = match.group(1)

            return texto.strip()

        except Exception as e:
            erro_str = str(e).lower()
            is_rate_limit = (
                "429" in erro_str
                or "resourceexhausted" in erro_str
                or "resource_exhausted" in erro_str
                or "quota" in erro_str
            )

            if is_rate_limit and tentativa < MAX_RETRIES:
                espera = 60 * tentativa
                log.warning(
                    "Rate limit (tentativa %d/%d). Aguardando %ds...",
                    tentativa, MAX_RETRIES, espera,
                )
                await asyncio.sleep(espera)
            else:
                raise


# ---------------------------------------------------------------------------
# Salvar resultados parciais
# ---------------------------------------------------------------------------

def salvar_resultados(resultados: list[dict], caminho: Path) -> None:
    """Salva resultados em JSON com resumo de metricas."""
    cers = [r["cer"] for r in resultados if r.get("cer") is not None]
    wers = [r["wer"] for r in resultados if r.get("wer") is not None]

    resumo = {
        "total_processadas": len(resultados),
        "total_sucesso": len(cers),
        "total_erros": len(resultados) - len(cers),
        "media_cer": round(sum(cers) / len(cers), 4) if cers else None,
        "media_wer": round(sum(wers) / len(wers), 4) if wers else None,
        "melhor_cer": round(min(cers), 4) if cers else None,
        "pior_cer": round(max(cers), 4) if cers else None,
    }

    saida = {"resumo": resumo, "resultados": resultados}
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Resultados salvos em %s (%d imagens)", caminho.name, len(resultados))


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

async def processar_lote(
    pares: list[tuple[Path, Path]],
    output: Path,
) -> None:
    """Processa todas as imagens com rate limiting e salvamento parcial."""
    resultados: list[dict] = []

    # Carregar resultados anteriores se existirem (para continuar de onde parou)
    ja_processados: set[str] = set()
    if output.exists():
        try:
            dados = json.loads(output.read_text(encoding="utf-8"))
            resultados = dados.get("resultados", [])
            ja_processados = {r["arquivo"] for r in resultados}
            log.info("Retomando: %d imagens ja processadas", len(ja_processados))
        except Exception:
            pass

    # Filtrar pares ja processados
    pares_pendentes = [(img, gt) for img, gt in pares if img.name not in ja_processados]
    total = len(pares_pendentes)

    if total == 0:
        log.info("Todas as imagens ja foram processadas!")
        salvar_resultados(resultados, output)
        return

    log.info("Processando %d imagens (delay=%.1fs entre chamadas)", total, DELAY_ENTRE_CHAMADAS)
    inicio_total = time.perf_counter()

    for i, (img, gt_path) in enumerate(pares_pendentes, 1):
        log.info("--- [%d/%d] %s ---", i, total, img.name)
        gt_texto = gt_path.read_text(encoding="utf-8")

        inicio = time.perf_counter()
        try:
            transcricao = await transcrever_imagem(img)
            tempo = round(time.perf_counter() - inicio, 2)

            cer = round(calcular_cer(gt_texto, transcricao), 4)
            wer = round(calcular_wer(gt_texto, transcricao), 4)

            log.info("  CER: %.2f%% | WER: %.2f%% | Tempo: %.1fs", cer * 100, wer * 100, tempo)

            resultados.append({
                "arquivo": img.name,
                "status": "ok",
                "cer": cer,
                "wer": wer,
                "tempo_s": tempo,
                "transcricao_trecho": transcricao[:200],
                "gt_trecho": gt_texto[:200],
            })

        except Exception as e:
            log.error("  ERRO: %s", str(e)[:150])
            resultados.append({
                "arquivo": img.name,
                "status": "erro",
                "cer": None,
                "wer": None,
                "erro": str(e)[:300],
            })

        # Salvar parcial a cada N imagens
        if i % SALVAR_A_CADA == 0:
            salvar_resultados(resultados, output)

        # Rate limiting (exceto na ultima)
        if i < total:
            await asyncio.sleep(DELAY_ENTRE_CHAMADAS)

    # Salvar final
    duracao = time.perf_counter() - inicio_total
    log.info("Concluido em %.1fs", duracao)
    salvar_resultados(resultados, output)

    # Resumo final
    cers = [r["cer"] for r in resultados if r.get("cer") is not None]
    if cers:
        log.info("=== RESUMO FINAL ===")
        log.info("  CER medio: %.2f%%", (sum(cers) / len(cers)) * 100)
        log.info("  Melhor CER: %.2f%%", min(cers) * 100)
        log.info("  Pior CER: %.2f%%", max(cers) * 100)
        log.info("  Sucesso: %d/%d", len(cers), len(resultados))


# ---------------------------------------------------------------------------
# Resumir resultados existentes
# ---------------------------------------------------------------------------

def resumir(caminho: Path) -> None:
    """Imprime resumo de um results.json existente."""
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    resumo = dados.get("resumo", {})
    resultados = dados.get("resultados", [])

    print(f"\nTotal processadas: {resumo.get('total_processadas', 0)}")
    print(f"Sucesso: {resumo.get('total_sucesso', 0)}")
    print(f"Erros: {resumo.get('total_erros', 0)}")
    print(f"CER medio: {(resumo.get('media_cer') or 0) * 100:.2f}%")
    print(f"WER medio: {(resumo.get('media_wer') or 0) * 100:.2f}%")
    print(f"Melhor CER: {(resumo.get('melhor_cer') or 0) * 100:.2f}%")
    print(f"Pior CER: {(resumo.get('pior_cer') or 0) * 100:.2f}%")

    # Top 5 melhores e piores
    ok = [r for r in resultados if r.get("cer") is not None]
    if ok:
        ok.sort(key=lambda r: r["cer"])
        print("\nTop 5 melhores:")
        for r in ok[:5]:
            print(f"  {r['arquivo']}: CER={r['cer']*100:.2f}%")
        print("\nTop 5 piores:")
        for r in ok[-5:]:
            print(f"  {r['arquivo']}: CER={r['cer']*100:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Processa imagens BRESSAY em lote com Gemini + rate limiting",
    )
    parser.add_argument("--imagens", "-i", type=Path, help="Diretorio de imagens")
    parser.add_argument("--ground-truths", "-g", type=Path, help="Diretorio de ground truths (.txt)")
    parser.add_argument("--output", "-o", type=Path, default=Path("results.json"), help="Arquivo de saida (padrao: results.json)")
    parser.add_argument("--limite", "-n", type=int, default=None, help="Limitar a N imagens")
    parser.add_argument("--resumir", "-r", type=Path, default=None, help="Resumir resultados de um JSON existente")

    args = parser.parse_args()

    if args.resumir:
        resumir(args.resumir)
        return

    if not args.imagens or not args.ground_truths:
        parser.error("--imagens e --ground-truths sao obrigatorios (ou use --resumir)")

    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY nao definida! Configure o .env")
        sys.exit(1)

    genai.configure(api_key=GEMINI_API_KEY)

    if not args.imagens.is_dir():
        log.error("Diretorio de imagens nao encontrado: %s", args.imagens)
        sys.exit(1)
    if not args.ground_truths.is_dir():
        log.error("Diretorio de ground truths nao encontrado: %s", args.ground_truths)
        sys.exit(1)

    pares = emparelhar(args.imagens, args.ground_truths)
    if not pares:
        log.error("Nenhum par imagem/ground-truth encontrado!")
        sys.exit(1)

    if args.limite and args.limite < len(pares):
        log.info("Limitando a %d imagens", args.limite)
        pares = pares[:args.limite]

    asyncio.run(processar_lote(pares, args.output))


if __name__ == "__main__":
    main()
