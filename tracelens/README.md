# TraceLens

AI-powered wireless trace analysis — Flask/Python backend, vanilla JS frontend.

This is a full-stack port of `traffic_llm_analysis` with a UI modeled after **Packet Navigator**.
The Python pipeline runs on the server (Flask), so all the heavy pandas/numpy processing
stays in Python. The browser handles rendering: charts via Chart.js, markdown via Marked.js.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."          # required for AI plan + narrative
export OPENAI_MODEL="gpt-5.4"           # optional, default is gpt-4o
flask --app app run --debug --port 5000
```

Then open http://localhost:5000

## What to upload

A .zip file containing session folders, each with Teledyne LeCroy CSV exports:

```
my_captures.zip
├── 02_04_1/
│   ├── 02_04_tap.csv
│   ├── 02_04_mac.csv
│   ├── 02_04_error.csv
│   └── 02_04_udp.csv  (optional)
└── 02_04_2/
    ├── ...
```

You can also point it at the `dataset/` folder from `traffic_llm_analysis` by zipping it first:
```bash
cd traffic_llm_analysis-main
zip -r dataset.zip dataset/
```

## Architecture

```
Browser (JS)
  │
  ├─ POST /api/upload   → Flask reads zip, runs pandas merge pipeline, stores sessions in memory
  ├─ POST /api/analyze  → Flask calls OpenAI for plan JSON, runs analysis engine, calls OpenAI for narrative
  └─ POST /api/clear    → clears session store
```

All sessions are stored in-memory per browser session token (a UUID sent as a header).
Restarting Flask clears all sessions.

## Analysis tasks supported

- `burst_detection` — z-score burst flagging per time bucket
- `delay_jitter_analysis` — p95 IAT, std IAT, jitter score per time bucket
- `error_spike_analysis` — bad FCS, error frame, retry rate spikes
- `top_entities` — top talkers/MACs/channels by packet count or bytes
- `timeline_summary` — aggregate metrics over time buckets
- `custom_filter_aggregate` — flexible filter + group + aggregate
- Cross-session comparison — ranks all sessions by a chosen metric
