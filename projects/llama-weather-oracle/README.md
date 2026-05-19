# llama-weather-oracle

A tiny local weather assistant that combines:

- Open-Meteo for live weather data
- Ollama for the local Llama response
- A funny editable personality prompt

## Setup

From this folder:

```bash
uv sync
```

Make sure Ollama is running in another terminal:

```bash
ollama serve
```

Make sure the model is downloaded:

```bash
ollama run llama3.2
```

## Run

```bash
uv run llama-weather Miami
```

Or ask for Celsius:

```bash
uv run llama-weather "New York" --unit celsius
```

Try another local model:

```bash
uv run llama-weather London --model qwen3:8b
```

## Change the personality

Edit the `PERSONALITY` text in `src/llama_weather_oracle/weather_oracle.py`.
