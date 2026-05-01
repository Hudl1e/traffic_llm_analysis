# Traffic LLM Analysis

Traffic LLM Analysis is an AI-assisted wireless traffic analysis project for exploring packet-level WiFi traces exported from Teledyne LeCroy/WPS. The project combines a Python data-processing pipeline with an LLM planning and explanation layer, allowing users to ask natural-language questions about wireless trace behavior such as bursty traffic, jitter, retry rates, bad FCS errors, packet-rate changes, and cross-session differences.

The main goal is to make low-level traffic traces easier to inspect and explain. Instead of manually writing a new pandas script for every question, the system converts a user question into a structured analysis plan, runs deterministic local analysis over the trace tables, and then uses an LLM to summarize the result in readable language.

## What this project does

This project supports analysis questions such as:

- Which session has the most bursty traffic?
- Which session has the highest retry rate?
- Is the traffic stable or bursty over time?
- Which time windows have high jitter?
- Are there spikes in bad FCS or error frames?
- Which MAC addresses, channels, or flows dominate during burst periods?
- In 02_04_1, show whether there is high jitter over time?

The core pipeline performs local computation using Python, pandas, and numpy. The LLM is used mainly for:

1. Translating natural-language questions into an executable analysis plan.
2. Choosing appropriate metrics such as packet count, inter-arrival time, jitter score, retry rate, or bad FCS rate.
3. Generating a human-readable explanation of the computed results.

The actual metric computation is done locally, so the results are grounded in the uploaded CSV trace data.

## Repository structure

```text
TRAFFIC_LLM_ANALYSIS/
├── dataset/                    # Example/session trace CSV folders
├── tracelens/                  # Flask web application
│   ├── app.py                  # Backend server
│   ├── templates/              # Frontend HTML templates
│   ├── README.md               # Web app setup instructions
│   └── requirements.txt        # Web app dependencies
├── traffic_llm_pipeline.py     # Main Python analysis pipeline
└── README.md
```

## Running the command-line LLM pipeline

You can also run the LLM analysis directly from the terminal without using the website.
First, set your OpenAI API key. Then run the pipeline with a dataset folder and a natural-language question

```bash
export OPENAI_API_KEY=""
python traffic_llm_pipeline.py --root dataset --question "Which session has the strongest bursty traffic pattern?"
```

Or start the TraceLens website:

```bash
cd tracelens
pip install -r requirements.txt
export OPENAI_API_KEY=""
flask --app app run --debug --port 5000
```