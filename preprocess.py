"""
preprocess.py — Pré-processamento de imagens de redações manuscritas (BRESSAY)
Pipeline: Deskew → CLAHE → Binarização → Redimensionamento
Uso: python preprocess.py --input imagem.jpg --output resultado.png
      python preprocess.py --input pasta_entrada/ --output pasta_saida/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")

# Extensões de imagem aceitas
EXTENSOES_VALIDAS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Resolução máxima de saída (lado maior) — otimizado para envio ao LLM
RESOLUCAO_MAX: int = 2048


# ---------------------------------------------------------------------------
# Etapas do pipeline
# ---------------------------------------------------------------------------

def carregar_imagem(caminho: Path) -> np.ndarray:
    """Carrega imagem em escala de cinza."""
    img = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar: {caminho}")
    log.info("Imagem carregada: %s (%dx%d)", caminho.name, img.shape[1], img.shape[0])
    return img


def deskew(img: np.ndarray, angulo_max: float = 15.0) -> tuple[np.ndarray, float]:
    """Corrige inclinação usando projeção horizontal (Hough Lines)."""
    # Binarização temporária para detectar linhas
    _, binarizada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detecta coordenadas de pixels brancos (texto)
    coords = np.column_stack(np.where(binarizada > 0))

    if coords.shape[0] < 100:
        log.warning("Poucos pixels de texto detectados — pulando deskew")
        return img, 0.0

    # minAreaRect retorna ângulo da rotação mínima
    retangulo = cv2.minAreaRect(coords)
    angulo: float = retangulo[-1]

    # Normaliza ângulo para intervalo [-45, 45]
    if angulo < -45:
        angulo = -(90 + angulo)
    else:
        angulo = -angulo

    # Limita correção a angulo_max graus
    if abs(angulo) > angulo_max:
        log.warning("Ângulo detectado (%.2f°) excede limite — pulando deskew", angulo)
        return img, 0.0

    if abs(angulo) < 0.3:
        log.info("Inclinação desprezível (%.2f°) — sem correção", angulo)
        return img, angulo

    # Aplica rotação
    h, w = img.shape[:2]
    centro = (w // 2, h // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    rotacionada = cv2.warpAffine(
        img, matriz, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    log.info("Deskew aplicado: %.2f°", angulo)
    return rotacionada, angulo


def aplicar_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    resultado = clahe.apply(img)
    log.info("CLAHE aplicado (clip=%.1f, tile=%d)", clip_limit, tile_size)
    return resultado


def binarizar(img: np.ndarray, metodo: str = "adaptativo") -> np.ndarray:
    """Binariza a imagem usando Otsu ou limiar adaptativo."""
    if metodo == "otsu":
        _, resultado = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        log.info("Binarização Otsu aplicada")
    elif metodo == "adaptativo":
        resultado = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,
            C=10,
        )
        log.info("Binarização adaptativa aplicada (block=21, C=10)")
    else:
        raise ValueError(f"Método de binarização desconhecido: {metodo}")
    return resultado


def redimensionar(img: np.ndarray, max_lado: int = RESOLUCAO_MAX) -> tuple[np.ndarray, float]:
    """Redimensiona mantendo proporção, limitando o lado maior."""
    h, w = img.shape[:2]
    maior = max(h, w)

    if maior <= max_lado:
        log.info("Redimensionamento desnecessário (%dx%d já dentro do limite)", w, h)
        return img, 1.0

    escala = max_lado / maior
    novo_w = int(w * escala)
    novo_h = int(h * escala)
    resultado = cv2.resize(img, (novo_w, novo_h), interpolation=cv2.INTER_AREA)
    log.info("Redimensionado: %dx%d → %dx%d (escala %.2f)", w, h, novo_w, novo_h, escala)
    return resultado, escala


def remover_ruido(img: np.ndarray) -> np.ndarray:
    """Remove ruído leve com filtro morfológico (abertura)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    resultado = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    log.info("Remoção de ruído (abertura morfológica) aplicada")
    return resultado


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def preprocessar(
    caminho_entrada: Path,
    caminho_saida: Path,
    metodo_binarizacao: str = "adaptativo",
    max_resolucao: int = RESOLUCAO_MAX,
    pular_binarizacao: bool = False,
) -> dict:
    """
    Executa o pipeline completo de pré-processamento em uma imagem.
    Retorna dicionário com metadados do processamento.
    """
    inicio = time.perf_counter()
    meta: dict = {
        "arquivo_entrada": str(caminho_entrada),
        "arquivo_saida": str(caminho_saida),
        "etapas": [],
    }

    # 1. Carregar
    img = carregar_imagem(caminho_entrada)
    meta["resolucao_original"] = [img.shape[1], img.shape[0]]

    # 2. Deskew
    img, angulo = deskew(img)
    meta["angulo_deskew"] = round(angulo, 2)
    meta["etapas"].append("deskew")

    # 3. CLAHE
    img = aplicar_clahe(img)
    meta["etapas"].append("clahe")

    # 4. Binarização (opcional — pode pular para manter tons de cinza)
    if not pular_binarizacao:
        img = binarizar(img, metodo=metodo_binarizacao)
        meta["etapas"].append(f"binarizacao_{metodo_binarizacao}")

    # 5. Remoção de ruído (só se binarizou)
    if not pular_binarizacao:
        img = remover_ruido(img)
        meta["etapas"].append("remocao_ruido")

    # 6. Redimensionar
    img, escala = redimensionar(img, max_lado=max_resolucao)
    meta["escala_redimensionamento"] = round(escala, 4)
    meta["resolucao_final"] = [img.shape[1], img.shape[0]]
    meta["etapas"].append("redimensionamento")

    # Salvar
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(caminho_saida), img)

    duracao = time.perf_counter() - inicio
    meta["tempo_processamento_s"] = round(duracao, 3)
    log.info("Salvo em %s (%.3fs)", caminho_saida.name, duracao)

    return meta


def processar_lote(
    entrada: Path,
    saida: Path,
    metodo_binarizacao: str = "adaptativo",
    max_resolucao: int = RESOLUCAO_MAX,
    pular_binarizacao: bool = False,
) -> list[dict]:
    """Processa todas as imagens de um diretório."""
    arquivos = sorted(
        f for f in entrada.iterdir()
        if f.suffix.lower() in EXTENSOES_VALIDAS
    )

    if not arquivos:
        log.warning("Nenhuma imagem encontrada em %s", entrada)
        return []

    log.info("Processando lote: %d imagens em %s", len(arquivos), entrada)
    resultados: list[dict] = []

    for i, arq in enumerate(arquivos, 1):
        log.info("--- [%d/%d] %s ---", i, len(arquivos), arq.name)
        destino = saida / f"{arq.stem}_preprocessado.png"
        try:
            meta = preprocessar(
                arq, destino,
                metodo_binarizacao=metodo_binarizacao,
                max_resolucao=max_resolucao,
                pular_binarizacao=pular_binarizacao,
            )
            meta["status"] = "ok"
        except Exception as e:
            log.error("Erro ao processar %s: %s", arq.name, e)
            meta = {"arquivo_entrada": str(arq), "status": "erro", "erro": str(e)}
        resultados.append(meta)

    return resultados


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pré-processamento de imagens de redações manuscritas (BRESSAY)",
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Caminho da imagem ou diretório de imagens",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Caminho de saída (arquivo ou diretório)",
    )
    parser.add_argument(
        "--binarizacao", "-b", default="adaptativo",
        choices=["adaptativo", "otsu"],
        help="Método de binarização (padrão: adaptativo)",
    )
    parser.add_argument(
        "--resolucao", "-r", type=int, default=RESOLUCAO_MAX,
        help=f"Resolução máxima do lado maior (padrão: {RESOLUCAO_MAX})",
    )
    parser.add_argument(
        "--sem-binarizacao", action="store_true",
        help="Pula binarização (mantém tons de cinza — melhor para LLMs multimodais)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Imprime metadados do processamento em JSON no stdout",
    )

    args = parser.parse_args()

    if args.input.is_dir():
        resultados = processar_lote(
            args.input, args.output,
            metodo_binarizacao=args.binarizacao,
            max_resolucao=args.resolucao,
            pular_binarizacao=args.sem_binarizacao,
        )
    elif args.input.is_file():
        meta = preprocessar(
            args.input, args.output,
            metodo_binarizacao=args.binarizacao,
            max_resolucao=args.resolucao,
            pular_binarizacao=args.sem_binarizacao,
        )
        resultados = [meta]
    else:
        log.error("Entrada não encontrada: %s", args.input)
        sys.exit(1)

    if args.json:
        print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
