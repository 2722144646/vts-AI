# dsh-vts-avatar · standalone

Drive a VTube Studio Live2D avatar from **any** agent or chat frontend. No DSH
required: the bridge runs as a local service and your host talks to it over a
WebSocket protocol.

```text
your agent / chat UI  ──WS──▶  bridge (127.0.0.1:8765)  ──▶  VTube Studio
                                   │
                                   └─ Piper TTS + viseme timeline + expressions
```

## 1. Install

```powershell
cd dsh-vts-avatar-standalone
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

Requirements: Windows, Python 3.11+, VTube Studio with API enabled, a Piper
Chinese voice under `models/`. See `PROTOCOL.md` for VTS settings.

## 2. Start

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

The bridge listens on `ws://127.0.0.1:8765`.

## 3. Speak from your own program

```python
import asyncio, json, websockets

async def main():
    async with websockets.connect("ws://127.0.0.1:8765") as ws:
        await ws.send(json.dumps({"type": "hello", "client": "my-agent", "protocol": 1}))
        await ws.send(json.dumps({"type": "utterance.start", "utterance_id": "u1"}))
        await ws.send(json.dumps({"type": "text.delta", "text": "你好呀，这是我自己的程序接进来的。"}))
        await ws.send(json.dumps({"type": "utterance.end"}))
        await asyncio.sleep(6)

asyncio.run(main())
```

See `example_client.py` for a runnable version that also sets an emotion.

## 4. What your host gets

| Message | Purpose |
|---|---|
| `utterance.start` / `text.delta` / `utterance.end` | speech with phoneme-timed mouth |
| `cancel` | interrupt |
| `feel` | emotion labels, intensity, confidence, cause |
| `emotion` | explicit expression preset |
| `action` | one-shot body/head action |
| `mood.get` | read current mood |
| `local.chat` | ask the configured local model (optionally `show_text`) |
| `status` / `ping` | health |

Full contract: `PROTOCOL.md`. Machine-readable: `manifest.json`.

## 5. Local model (optional)

`config.toml` → `[llm]`:

```toml
enabled = true
backend = "ollama"      # ollama | openai | none
endpoint = "http://127.0.0.1:11434"
model = "qwen3:8b"
```

`backend = "openai"` works with any OpenAI-compatible `/v1/chat/completions`
server (LM Studio, llama.cpp server, vLLM, hosted APIs). With `none` the
`local.chat` path is disabled.

## 6. Privacy

`data/` holds emotion events, relationship, diary, private notes and the local
chat transcript. Nothing is uploaded; the bridge has no telemetry.
