import math
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS


# Ensure the app's own directory is on the path so traffic_llm_pipeline is importable.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# The shared pipeline imports Matplotlib for CLI plot saving. Give it a writable
# cache path when the Flask app imports the module in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tracelens-matplotlib"))

import traffic_llm_pipeline as pipeline


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "tracelens-dev-secret")
CORS(app)

# In-memory session store (maps session_token -> {session_id -> DataFrame})
_SESSION_STORE: Dict[str, Dict[str, pd.DataFrame]] = {}


def discover_and_load_zip(zip_path: str) -> Dict[str, pd.DataFrame]:
    sessions: Dict[str, pd.DataFrame] = {}
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        root = Path(tmp)
        for session_dir in sorted(root.rglob("*")):
            if not session_dir.is_dir():
                continue

            files = {}
            for f in session_dir.glob("*.csv"):
                name = f.name.lower()
                if "tap" in name:
                    files["tap"] = f
                elif "mac" in name:
                    files["mac"] = f
                elif "error" in name:
                    files["error"] = f
                elif "udp" in name:
                    files["udp"] = f

            if not files:
                continue

            try:
                sessions[session_dir.name] = pipeline.merge_session_tables(files)
            except Exception as e:
                print(f"Skipping session {session_dir.name}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return sessions


def build_plan(question: str, session_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    session_ids = sorted(session_summaries.keys())

    if pipeline.should_run_local_anomaly_detection(question):
        return pipeline.build_local_anomaly_plan(question, session_ids)

    if pipeline.should_run_local_burst_dominance(question):
        return pipeline.build_local_burst_dominance_plan(question, session_ids)

    local_comparison_plan = pipeline.build_local_comparison_plan(question, session_ids)
    if local_comparison_plan is not None:
        return local_comparison_plan

    return pipeline.ask_llm_for_plan(question, session_summaries)


def execute_plan(
    sessions: Dict[str, pd.DataFrame],
    plan: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any], bool]:
    is_cross_session = plan.get("session_scope") == "all_sessions"

    if is_cross_session:
        plan["session_id"] = "ALL_SESSIONS"
        result, meta = pipeline.compare_sessions(sessions, plan)
        meta.setdefault("rows_after_filter", int(sum(len(df) for df in sessions.values())))
        meta.setdefault("result_rows", int(len(result)))
        return result, meta, True

    default_session = pipeline.choose_default_session(list(sessions.keys()))
    session_id = plan.get("session_id") or default_session
    if session_id not in sessions:
        session_id = default_session
    plan["session_id"] = session_id
    plan["session_scope"] = "single_session"
    plan["comparison_mode"] = "none"
    plan["comparison_metric"] = None

    result, meta = pipeline.run_plan(sessions[session_id], plan)
    return result, meta, False


def summarize_result(question: str, plan: Dict[str, Any], result: pd.DataFrame, meta: Dict[str, Any]) -> str:
    force_local_summary = (
        meta.get("comparison_metric") == "retry_rate"
        and meta.get("comparison_ascending")
        and len(meta.get("best_tied_sessions") or []) > 1
    )

    try:
        if force_local_summary:
            return pipeline.summarize_locally(question, plan, result, meta)
        return pipeline.summarize_with_llm(question, plan, result, meta)
    except Exception as e:
        print(f"[WARN] LLM summary failed; using local summary instead: {e}")
        return pipeline.summarize_locally(question, plan, result, meta)


def df_to_json(df: pd.DataFrame, max_rows: int = 500) -> List[Dict[str, Any]]:
    df2 = df.head(max_rows).copy()

    for col in df2.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
        df2[col] = df2[col].astype(str)

    for col in df2.columns:
        df2[col] = df2[col].where(df2[col].notna(), None)

    rows = df2.to_dict(orient="records")
    clean = []
    for row in rows:
        clean_row = {}
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                clean_row[key] = None
            elif hasattr(value, "item"):
                clean_row[key] = value.item()
            else:
                clean_row[key] = value
        clean.append(clean_row)
    return clean


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename.endswith(".zip"):
        return jsonify({"error": "Expected a .zip file"}), 400

    token = request.headers.get("X-Session-Token", "default")

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        uploaded_file.save(tmp_path)
        sessions = discover_and_load_zip(tmp_path)
        if not sessions:
            return jsonify({"error": "No valid sessions found. Expected folders containing tap/mac/error/udp CSVs."}), 400

        _SESSION_STORE[token] = sessions
        info = [
            {"id": session_id, "rows": len(df), "columns": list(df.columns)}
            for session_id, df in sessions.items()
        ]
        return jsonify({"sessions": info})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/api/analyze", methods=["POST"])
def analyze():
    token = request.headers.get("X-Session-Token", "default")
    sessions = _SESSION_STORE.get(token)
    if not sessions:
        return jsonify({"error": "No sessions loaded. Please upload a ZIP first."}), 400

    body = request.get_json()
    question = (body or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        summaries = {
            session_id: pipeline.summarize_dataframe_schema(df)
            for session_id, df in sessions.items()
        }
        plan = build_plan(question, summaries)
        result_df, meta, is_cross_session = execute_plan(sessions, plan)
        narrative = summarize_result(question, plan, result_df, meta)

        return jsonify({
            "plan": plan,
            "result": df_to_json(result_df),
            "meta": meta,
            "narrative": narrative,
            "is_cross_session": is_cross_session,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    token = request.headers.get("X-Session-Token", "default")
    _SESSION_STORE.pop(token, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000)