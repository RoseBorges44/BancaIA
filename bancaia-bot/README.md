# BancaIA — Bot Telegram

Bot que corrige redacoes manuscritas do ENEM usando Banca Virtual multi-agente.

## Como usar

1. Crie um bot no [@BotFather](https://t.me/BotFather) e copie o token
2. Obtenha uma chave da [API Anthropic](https://console.anthropic.com/)
3. Configure o ambiente:

```bash
cp .env.example .env
# Edite o .env com seu TELEGRAM_TOKEN e ANTHROPIC_API_KEY
```

4. Instale e rode:

```bash
pip install -r requirements.txt
python bot.py
```

## Fluxo

1. Usuario envia `/start` -> instrucoes
2. Usuario envia foto da redacao (comprimida ou documento)
3. Bot mostra "Banca em deliberacao..." com etapas
4. Claude Sonnet transcreve e avalia a redacao
5. Bot responde com nota total, barras C1-C5, feedback socratico, plano de estudos
6. Botao inline "Ver transcricao completa" expande o texto

## Variaveis de ambiente

| Variavel | Descricao |
|---|---|
| `TELEGRAM_TOKEN` | Token do BotFather |
| `ANTHROPIC_API_KEY` | Chave da API Anthropic |
