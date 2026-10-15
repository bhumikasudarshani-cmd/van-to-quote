# Van-to-Quote 🔧
Talk about a job, get a priced PDF quote. Runs fully offline once models are downloaded.

## Setup
1. Install Ollama (https://ollama.com), then: `ollama pull qwen2.5:7b` (or `qwen2.5:3b` for small laptops)
2. `pip install -r requirements.txt`
3. Edit `prices.json` with your business name, rates, VAT and price list
4. `python app.py` and open http://localhost:5000

Mic access works on localhost. To use from a phone on your wifi, serve over HTTPS or use the text box.

## How it works
Whisper (faster-whisper) transcribes -> a local LLM extracts items, quantities and hours as JSON ->
your own price list supplies every price and all maths is plain code. The model never makes up a price.
Config via env vars: LLM_MODEL, WHISPER_MODEL, OLLAMA_URL.
