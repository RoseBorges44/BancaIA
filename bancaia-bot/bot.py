"""
bot.py — BancaIA Telegram Bot
Corrige redacoes manuscritas do ENEM usando Banca Virtual multi-agente.
Usa Google Gemini para transcricao + avaliacao.

Uso: python bot.py
Requer: TELEGRAM_TOKEN e GEMINI_API_KEY no arquivo .env
"""

import asyncio
import json
import logging
import os
import re
from io import BytesIO

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bancaia-bot")

TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODELO: str = "gemini-2.0-flash"

# Configura a API do Gemini
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT: str = (
    "Voce e a BancaIA — uma banca virtual multi-agente que corrige "
    "redacoes manuscritas do ENEM.\n\n"
    "Pipeline:\n"
    "1. Transcreva a imagem da redacao manuscrita (HTR).\n"
    "2. Avalie as 5 competencias do ENEM (C1-C5), cada uma de 0 a 200 "
    "(multiplos de 40).\n"
    "3. Se houver divergencia entre competencias (>40 pontos entre runs), "
    "ative o Arbitro.\n"
    "4. Gere feedback socratico, plano de estudos e fingerprint de erros.\n\n"
    "Responda EXCLUSIVAMENTE com JSON valido (sem markdown, sem texto extra) "
    "neste formato:\n"
    '{\n'
    '  "transcricao": "texto completo transcrito",\n'
    '  "confianca_transcricao": 0.87,\n'
    '  "tema_detectado": "Tema identificado",\n'
    '  "competencias": {\n'
    '    "C1": { "nota": 160, "titulo": "Norma culta", "razao": "..." },\n'
    '    "C2": { "nota": 140, "titulo": "Proposta e repertorio", "razao": "..." },\n'
    '    "C3": { "nota": 120, "titulo": "Argumentacao", "razao": "..." },\n'
    '    "C4": { "nota": 120, "titulo": "Coesao", "razao": "..." },\n'
    '    "C5": { "nota": 80,  "titulo": "Intervencao", "razao": "..." }\n'
    '  },\n'
    '  "nota_total": 620,\n'
    '  "divergencia_detectada": false,\n'
    '  "motivo_divergencia": "",\n'
    '  "parecer_arbitro": "",\n'
    '  "feedback_socratico": ["Pergunta 1?", "Pergunta 2?", "Pergunta 3?"],\n'
    '  "plano_estudos": ["Acao 1", "Acao 2", "Acao 3"],\n'
    '  "fingerprint_erros": ["Padrao 1", "Padrao 2"]\n'
    '}'
)

# Armazena transcricoes para o botao "Ver transcricao"
transcricoes: dict[int, str] = {}


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def barra_unicode(nota: int, max_nota: int = 200) -> str:
    """Gera barra visual com blocos unicode."""
    total = 10
    preenchidos = round((nota / max_nota) * total)
    return "\u2588" * preenchidos + "\u2591" * (total - preenchidos)


def formatar_resultado(r: dict) -> str:
    """Formata o resultado da avaliacao para mensagem do Telegram."""
    nota = r.get("nota_total", 0)

    if nota >= 800:
        emoji = "\U0001f31f"
    elif nota >= 600:
        emoji = "\u2705"
    elif nota >= 400:
        emoji = "\u26a0\ufe0f"
    else:
        emoji = "\U0001f534"

    linhas = [f"{emoji} *NOTA TOTAL: {nota}/1000*", ""]

    tema = r.get("tema_detectado", "")
    if tema:
        linhas.append(f"\U0001f4cb *Tema:* {tema}")
        linhas.append("")

    linhas.append("*Competencias:*")
    for cid in ["C1", "C2", "C3", "C4", "C5"]:
        comp = r.get("competencias", {}).get(cid, {})
        n = comp.get("nota", 0)
        titulo = comp.get("titulo", cid)
        barra = barra_unicode(n)
        linhas.append(f"`{cid}` {barra} *{n}*/200")
        linhas.append(f"    _{titulo}_")

    if r.get("divergencia_detectada"):
        linhas.append("")
        linhas.append("\u26a0\ufe0f *Divergencia detectada!*")
        motivo = r.get("motivo_divergencia", "")
        if motivo:
            linhas.append(f"_{motivo}_")
        parecer = r.get("parecer_arbitro", "")
        if parecer:
            linhas.append(f"*Arbitro:* {parecer[:300]}")

    conf = r.get("confianca_transcricao", 0)
    linhas.append("")
    linhas.append(f"*Confianca da transcricao:* {conf:.0%}")

    socratico = r.get("feedback_socratico", [])
    if socratico:
        linhas.append("")
        linhas.append("*Perguntas para reflexao:*")
        for i, p in enumerate(socratico[:3], 1):
            linhas.append(f"  {i}. _{p}_")

    plano = r.get("plano_estudos", [])
    if plano:
        linhas.append("")
        linhas.append("*Plano de estudos:*")
        for a in plano[:3]:
            linhas.append(f"  \u2022 {a}")

    fingerprint = r.get("fingerprint_erros", [])
    if fingerprint:
        linhas.append("")
        linhas.append("*Padroes de erro:*")
        linhas.append("  " + " | ".join(fingerprint[:5]))

    return "\n".join(linhas)


async def chamar_gemini(imagem_bytes: bytes, msg=None, max_tentativas: int = 3) -> dict:
    """Envia imagem para Gemini e retorna o JSON de avaliacao.
    Retry automatico com espera de 60s em caso de rate limit (429/ResourceExhausted).
    Se msg (mensagem do Telegram) for passada, atualiza o status para o usuario.
    """
    img = Image.open(BytesIO(imagem_bytes))
    model = genai.GenerativeModel(MODELO)

    for tentativa in range(1, max_tentativas + 1):
        try:
            response = model.generate_content(
                [
                    SYSTEM_PROMPT + "\n\nTranscreva e avalie esta redacao manuscrita.",
                    img,
                ],
            )

            texto = response.text

            # Extrair JSON caso venha com markdown
            json_str = texto
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", texto)
            if match:
                json_str = match.group(1)

            return json.loads(json_str.strip())

        except Exception as e:
            erro_str = str(e).lower()
            is_rate_limit = (
                "429" in erro_str
                or "resourceexhausted" in erro_str
                or "resource_exhausted" in erro_str
                or "quota" in erro_str
            )

            if is_rate_limit and tentativa < max_tentativas:
                espera = 60
                log.warning(
                    "Rate limit atingido (tentativa %d/%d). "
                    "Aguardando %ds antes de tentar novamente...",
                    tentativa, max_tentativas, espera,
                )
                # Avisar o usuario no Telegram
                if msg:
                    try:
                        await msg.edit_text(
                            f"\u23f3 Limite da API atingido, aguardando {espera}s... "
                            f"(tentativa {tentativa + 1}/{max_tentativas})",
                            parse_mode="Markdown",
                        )
                    except Exception:
                        pass  # Ignora erro de edicao (ex: mensagem nao modificada)
                await asyncio.sleep(espera)
            else:
                raise


# ---------------------------------------------------------------------------
# Handlers do Telegram
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler do comando /start."""
    await update.message.reply_text(
        "*BancaIA \u2014 Corretor Inteligente de Redacoes*\n\n"
        "Envie uma _foto da redacao manuscrita_ (foto ou documento de imagem) "
        "e a Banca Virtual vai:\n\n"
        "1. Transcrever o texto (HTR)\n"
        "2. Avaliar as 5 competencias do ENEM\n"
        "3. Gerar feedback socratico + plano de estudos\n\n"
        "_Tire uma foto agora ou envie uma imagem da galeria!_",
        parse_mode="Markdown",
    )


async def _processar_imagem(
    update: Update, imagem_bytes: bytes, media_type: str
) -> None:
    """Logica compartilhada para processar foto ou documento."""
    msg = await update.message.reply_text(
        "*Banca em deliberacao...*\n\n"
        "\u23f3 Etapa 1/3: Transcrevendo imagem...\n"
        "\u2b1c Etapa 2/3: Avaliando competencias\n"
        "\u2b1c Etapa 3/3: Gerando feedback",
        parse_mode="Markdown",
    )

    try:
        await msg.edit_text(
            "*Banca em deliberacao...*\n\n"
            "\u2705 Etapa 1/3: Transcrevendo imagem\n"
            "\u23f3 Etapa 2/3: Avaliando competencias...\n"
            "\u2b1c Etapa 3/3: Gerando feedback",
            parse_mode="Markdown",
        )

        resultado = await chamar_gemini(imagem_bytes, msg=msg)

        await msg.edit_text(
            "*Banca em deliberacao...*\n\n"
            "\u2705 Etapa 1/3: Transcrevendo imagem\n"
            "\u2705 Etapa 2/3: Avaliando competencias\n"
            "\u23f3 Etapa 3/3: Gerando feedback...",
            parse_mode="Markdown",
        )

        chat_id = update.message.chat_id
        transcricoes[chat_id] = resultado.get("transcricao", "(sem transcricao)")

        texto_resultado = formatar_resultado(resultado)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(
                "\U0001f4c4 Ver transcricao completa",
                callback_data=f"transcricao_{chat_id}",
            )]
        ])

        await msg.edit_text(
            texto_resultado,
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

    except json.JSONDecodeError:
        log.error("Erro ao parsear JSON do Gemini")
        await msg.edit_text(
            "\u274c *Erro:* Resposta invalida do modelo. "
            "Tente com foto mais nitida.",
            parse_mode="Markdown",
        )
    except Exception as e:
        log.error("Erro: %s", e, exc_info=True)
        await msg.edit_text(
            f"\u274c *Erro:* {str(e)[:200]}",
            parse_mode="Markdown",
        )


async def processar_foto(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handler para fotos comprimidas pelo Telegram."""
    foto = update.message.photo[-1]
    arquivo = await foto.get_file()
    bio = BytesIO()
    await arquivo.download_to_memory(bio)
    await _processar_imagem(update, bio.getvalue(), "image/jpeg")


async def processar_documento(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handler para imagens enviadas como documento."""
    doc = update.message.document
    if not doc.mime_type or not doc.mime_type.startswith("image/"):
        await update.message.reply_text(
            "Envie uma _imagem_ da redacao (JPG, PNG, WebP).",
            parse_mode="Markdown",
        )
        return

    arquivo = await doc.get_file()
    bio = BytesIO()
    await arquivo.download_to_memory(bio)
    await _processar_imagem(update, bio.getvalue(), doc.mime_type)


async def callback_transcricao(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handler do botao inline Ver transcricao completa."""
    query = update.callback_query
    await query.answer()

    try:
        chat_id = int(query.data.split("_")[1])
    except (IndexError, ValueError):
        chat_id = query.message.chat_id

    transcricao = transcricoes.get(chat_id, "(transcricao nao disponivel)")

    # Telegram tem limite de 4096 chars por mensagem
    if len(transcricao) > 3800:
        transcricao = transcricao[:3800] + "\n\n_(...texto truncado)_"

    await query.message.reply_text(
        f"*Transcricao completa:*\n\n{transcricao}",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Inicia o bot."""
    if not TELEGRAM_TOKEN:
        log.error("TELEGRAM_TOKEN nao definido! Configure o .env")
        return
    if not GEMINI_API_KEY:
        log.error("GEMINI_API_KEY nao definida! Configure o .env")
        return

    log.info("Iniciando BancaIA Bot (Gemini)...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(MessageHandler(filters.PHOTO, processar_foto))
    app.add_handler(MessageHandler(filters.Document.ALL, processar_documento))
    app.add_handler(CallbackQueryHandler(
        callback_transcricao, pattern=r"^transcricao_"
    ))

    log.info("Bot rodando! Ctrl+C para parar.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
