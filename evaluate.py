"""
evaluate.py — Avaliação CER/WER da transcrição do Claude vs ground truth BRESSAY
Uso como módulo: from evaluate import avaliar_imagem, calcular_metricas
Uso CLI:        python evaluate.py --imagem redacao.png --ground-truth texto.txt
"""

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import jiwer
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")

MODELO: str = "claude-sonnet-4-20250514"
API_URL: str = "https://api.anthropic.com/v1/messages"
MAX_TOKENS: int = 4096

# Extensões de imagem aceitas
EXTENSOES_IMAGEM: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# Exemplos few-shot (pares imagem-descrição para calibrar o modelo)
# Estes exemplos são textuais — simulam o padrão de transcrição esperado.
# Para few-shot com imagens reais, coloque-as em exemplos/ e ajuste os caminhos.
# ---------------------------------------------------------------------------

EXEMPLOS_FEW_SHOT: list[dict] = [
    {
        "role": "user",
        "content": (
            "Exemplo de transcrição — o texto manuscrito abaixo diz:\n"
            "\"A educação é o principal pilar para o desenvolvimento de uma nação. "
            "Sem investimento adequado, o país permanece estagnado.\""
        ),
    },
    {
        "role": "assistant",
        "content": (
            "A educação é o principal pilar para o desenvolvimento de uma nação. "
            "Sem investimento adequado, o país permanece estagnado."
        ),
    },
    {
        "role": "user",
        "content": (
            "Exemplo de transcrição — o texto manuscrito abaixo diz:\n"
            "\"Diante disso, é necessário que o Ministério da Educação, em parceria "
            "com as secretarias estaduais, promova campanhas de conscientização sobre "
            "a importância da leitura nas escolas públicas.\""
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Diante disso, é necessário que o Ministério da Educação, em parceria "
            "com as secretarias estaduais, promova campanhas de conscientização sobre "
            "a importância da leitura nas escolas públicas."
        ),
    },
]

SYSTEM_PROMPT: str = (
    "Você é um especialista em HTR (Handwritten Text Recognition) de redações "
    "manuscritas em português brasileiro. Sua tarefa é transcrever EXATAMENTE o "
    "texto manuscrito da imagem, preservando:\n"
    "- Ortografia original (incluindo erros do autor)\n"
    "- Pontuação original\n"
    "- Quebras de parágrafo (use \\n\\n entre parágrafos)\n"
    "- Quebras de linha dentro de parágrafos (use \\n)\n\n"
    "Responda APENAS com o texto transcrito, sem comentários, explicações ou "
    "formatação adicional. Não corrija erros de ortografia ou gramática."
)


# ---------------------------------------------------------------------------
# Funções de métricas
# ---------------------------------------------------------------------------

def normalizar_texto(texto: str) -> str:
    """Normaliza texto para comparação justa de métricas."""
    # Remove espaços extras e normaliza quebras de linha
    linhas = texto.strip().splitlines()
    linhas = [linha.strip() for linha in linhas if linha.strip()]
    return "\n".join(linhas)


def calcular_metricas(transcricao: str, ground_truth: str) -> dict:
    """
    Calcula CER e WER entre transcrição e ground truth.
    Retorna dict com cer, wer e métricas detalhadas.
    """
    trans_norm = normalizar_texto(transcricao)
    gt_norm = normalizar_texto(ground_truth)

    if not gt_norm:
        log.warning("Ground truth vazio — métricas indefinidas")
        return {"cer": None, "wer": None, "erro": "ground_truth_vazio"}

    # CER — Character Error Rate
    cer = jiwer.cer(gt_norm, trans_norm)

    # WER — Word Error Rate
    wer = jiwer.wer(gt_norm, trans_norm)

    # Métricas detalhadas por palavra
    medidas_wer = jiwer.compute_measures(gt_norm, trans_norm)

    resultado = {
        "cer": round(cer, 4),
        "wer": round(wer, 4),
        "substituicoes": medidas_wer["substitutions"],
        "delecoes": medidas_wer["deletions"],
        "insercoes": medidas_wer["insertions"],
        "palavras_referencia": medidas_wer["hits"] + medidas_wer["substitutions"] + medidas_wer["deletions"],
        "caracteres_referencia": len(gt_norm),
        "caracteres_transcricao": len(trans_norm),
    }

    return resultado


def calcular_metricas_por_linha(transcricao: str, ground_truth: str) -> list[dict]:
    """Calcula CER e WER linha a linha."""
    linhas_trans = normalizar_texto(transcricao).splitlines()
    linhas_gt = normalizar_texto(ground_truth).splitlines()

    # Alinhar quantidade de linhas (pad com string vazia)
    max_linhas = max(len(linhas_trans), len(linhas_gt))
    linhas_trans.extend([""] * (max_linhas - len(linhas_trans)))
    linhas_gt.extend([""] * (max_linhas - len(linhas_gt)))

    resultados: list[dict] = []
    for i, (lt, lgt) in enumerate(zip(linhas_trans, linhas_gt)):
        if not lgt and not lt:
            continue
        if not lgt:
            resultados.append({"linha": i + 1, "cer": 1.0, "wer": 1.0, "nota": "linha extra na transcrição"})
            continue
        if not lt:
            resultados.append({"linha": i + 1, "cer": 1.0, "wer": 1.0, "nota": "linha faltando na transcrição"})
            continue

        cer = jiwer.cer(lgt, lt)
        wer = jiwer.wer(lgt, lt)
        resultados.append({
            "linha": i + 1,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "referencia": lgt[:80] + ("..." if len(lgt) > 80 else ""),
            "transcricao": lt[:80] + ("..." if len(lt) > 80 else ""),
        })

    return resultados


# ---------------------------------------------------------------------------
# Chamada à API do Claude
# ---------------------------------------------------------------------------

def _imagem_para_base64(caminho: Path) -> tuple[str, str]:
    """Lê imagem e retorna (base64_data, media_type)."""
    extensao = caminho.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".bmp": "image/bmp",
        ".webp": "image/webp", ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    media_type = media_types.get(extensao, "image/png")
    dados = caminho.read_bytes()
    return base64.b64encode(dados).decode("utf-8"), media_type


def _montar_mensagens(
    imagem_b64: str,
    media_type: str,
    estrategia: str = "few-shot",
) -> list[dict]:
    """Monta a lista de mensagens para a API conforme a estratégia de prompting."""
    mensagens: list[dict] = []

    # Few-shot: adiciona exemplos textuais antes da imagem real
    if estrategia == "few-shot":
        mensagens.extend(EXEMPLOS_FEW_SHOT)
    elif estrategia == "one-shot":
        # Apenas o primeiro par de exemplo
        mensagens.extend(EXEMPLOS_FEW_SHOT[:2])
    # zero-shot: nenhum exemplo

    # Mensagem principal com a imagem
    mensagens.append({
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": imagem_b64,
                },
            },
            {
                "type": "text",
                "text": "Transcreva exatamente o texto manuscrito desta imagem.",
            },
        ],
    })

    return mensagens


async def transcrever_imagem(
    caminho_imagem: Path,
    estrategia: str = "few-shot",
    api_key: Optional[str] = None,
) -> dict:
    """
    Envia imagem para o Claude e retorna a transcrição.
    Retorna dict com: transcricao, modelo, estrategia, tempo_s, tokens_entrada, tokens_saida
    """
    chave = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not chave:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY não encontrada. Defina no .env ou passe via --api-key"
        )

    imagem_b64, media_type = _imagem_para_base64(caminho_imagem)
    mensagens = _montar_mensagens(imagem_b64, media_type, estrategia)

    payload = {
        "model": MODELO,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": mensagens,
    }

    headers = {
        "x-api-key": chave,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    inicio = time.perf_counter()

    async with httpx.AsyncClient(timeout=120.0) as client:
        resposta = await client.post(API_URL, json=payload, headers=headers)
        resposta.raise_for_status()

    duracao = time.perf_counter() - inicio
    dados = resposta.json()

    # Extrair texto da resposta
    texto = ""
    for bloco in dados.get("content", []):
        if bloco.get("type") == "text":
            texto += bloco["text"]

    uso = dados.get("usage", {})

    return {
        "transcricao": texto.strip(),
        "modelo": MODELO,
        "estrategia": estrategia,
        "tempo_s": round(duracao, 2),
        "tokens_entrada": uso.get("input_tokens", 0),
        "tokens_saida": uso.get("output_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Função principal de avaliação
# ---------------------------------------------------------------------------

async def avaliar_imagem(
    caminho_imagem: Path,
    ground_truth: str,
    estrategia: str = "few-shot",
    api_key: Optional[str] = None,
) -> dict:
    """
    Pipeline completo: transcreve imagem via Claude e compara com ground truth.
    Retorna dict com cer, wer, transcricao e metadados.
    """
    log.info("Avaliando: %s (estratégia: %s)", caminho_imagem.name, estrategia)

    # 1. Transcrever
    resultado_api = await transcrever_imagem(caminho_imagem, estrategia, api_key)
    transcricao = resultado_api["transcricao"]

    # 2. Calcular métricas globais
    metricas = calcular_metricas(transcricao, ground_truth)

    # 3. Métricas por linha
    metricas_linhas = calcular_metricas_por_linha(transcricao, ground_truth)

    log.info(
        "  → CER: %.2f%% | WER: %.2f%% | Tempo: %.1fs",
        (metricas.get("cer") or 0) * 100,
        (metricas.get("wer") or 0) * 100,
        resultado_api["tempo_s"],
    )

    return {
        "arquivo": str(caminho_imagem),
        "transcricao": transcricao,
        "ground_truth_trecho": ground_truth[:200] + ("..." if len(ground_truth) > 200 else ""),
        **metricas,
        "metricas_por_linha": metricas_linhas,
        **{k: v for k, v in resultado_api.items() if k != "transcricao"},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Avalia transcrição do Claude vs ground truth BRESSAY",
    )
    parser.add_argument(
        "--imagem", "-i", required=True, type=Path,
        help="Caminho da imagem da redação",
    )
    parser.add_argument(
        "--ground-truth", "-g", required=True, type=Path,
        help="Caminho do arquivo .txt com o ground truth",
    )
    parser.add_argument(
        "--estrategia", "-e", default="few-shot",
        choices=["zero-shot", "one-shot", "few-shot"],
        help="Estratégia de prompting (padrão: few-shot)",
    )
    parser.add_argument(
        "--api-key", "-k", type=str, default=None,
        help="Chave da API Anthropic (ou use ANTHROPIC_API_KEY no .env)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Salvar resultado em arquivo JSON (opcional)",
    )

    args = parser.parse_args()

    if not args.imagem.exists():
        log.error("Imagem não encontrada: %s", args.imagem)
        sys.exit(1)

    if not args.ground_truth.exists():
        log.error("Ground truth não encontrado: %s", args.ground_truth)
        sys.exit(1)

    gt_texto = args.ground_truth.read_text(encoding="utf-8")

    resultado = await avaliar_imagem(
        args.imagem, gt_texto,
        estrategia=args.estrategia,
        api_key=args.api_key,
    )

    # Imprimir resultado
    print(json.dumps(resultado, ensure_ascii=False, indent=2))

    # Salvar em arquivo se solicitado
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Resultado salvo em %s", args.output)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
