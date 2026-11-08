# Bridge protocol (v1)

Transport: WebSocket, default `ws://127.0.0.1:8765`. JSON messages. First message
must be:

```json
{"type": "hello", "client": "<your name>", "protocol": 1}
```

Requests may carry `request_id`; replies echo it plus a `.result` type.

## Speech

```json
{"type": "utterance.start", "utterance_id": "u1", "request_id": "r1"}
{"type": "text.delta", "text": "一句", "request_id": "r2"}
{"type": "utterance.end", "request_id": "r3"}
```

`text.delta` may be sent repeatedly (streaming). The bridge synthesizes each
segment, builds a viseme timeline, and drives mouth parameters against the
audio clock. `cancel` stops speech and clears the mouth.

## Emotion

```json
{"type": "feel",
 "labels": {"happy": 0.6, "shy": 0.3},
 "intensity": 0.7,
 "confidence": 0.8,
 "cause": "他说了句好听的"}
```

`emotion` sets an explicit preset (`{"name": "shy", "intensity": 0.7}`);
`action` triggers one body/head cue (`{"kind": "nod", "intensity": 0.6}`).
`mood.get` returns the current blended mood.

## Local model

```json
{"type": "local.chat", "text": "在吗", "show_text": false, "request_id": "r4"}
```

Without `show_text` the reply is spoken but only the character count is
returned (the host never sees it). With `show_text: true` the reply text is
included — use it only for a local UI that renders on this machine.

## Status

`ping` → `{"type": "pong"}`; `status` → bridge/audio/VTS summary.
