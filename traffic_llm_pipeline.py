import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openai import OpenAI

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None


# ============================================================
# Configuration
# ============================================================

MODEL_PLAN = "gpt-5.4"
MODEL_SUMMARY = "gpt-5.4"

DEFAULT_ROOT = "dataset"
OUTPUT_DIR = "outputs"

client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global client
    if client is None:
        client = OpenAI()
    return client


# ============================================================
# Utility helpers
# ============================================================

def snake_case(name: str) -> str:
    name = name.strip().replace("#", " num")
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(str(c)) for c in df.columns]
    return df


def clean_excel_style_string(s: pd.Series) -> pd.Series:
    out = s.astype("string")
    return (
        out
        .str.replace('^="', '', regex=True)
        .str.replace('"$', '', regex=True)
        .str.replace('^=$', '', regex=True)
        .str.strip()
    )


def parse_numeric_series(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_duration_to_us(s: pd.Series) -> pd.Series:
    """
    Handles examples like:
    57 µs, 57 us, 1.2 ms, 0.004 s
    """
    s = clean_excel_style_string(s)
    vals = []
    for x in s.fillna(""):
        x = str(x).strip().lower()
        if not x:
            vals.append(np.nan)
            continue

        m = re.search(r"(-?\d+(?:\.\d+)?)", x)
        if not m:
            vals.append(np.nan)
            continue

        val = float(m.group(1))
        if "ms" in x:
            vals.append(val * 1000.0)
        elif "µs" in x or "us" in x:
            vals.append(val)
        elif re.search(r"(^|\s)s($|\s)", x):
            vals.append(val * 1_000_000.0)
        else:
            # fallback: assume microseconds
            vals.append(val)

    return pd.Series(vals, index=s.index, dtype="float64")


def parse_timestamp_series(s: pd.Series) -> pd.Series:
    s = clean_excel_style_string(s)
    parsed = pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S.%f %p", errors="coerce")
    missing = parsed.isna() & s.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s.loc[missing], errors="coerce")
    return parsed


def read_wps_csv(path: Path) -> pd.DataFrame:
    """
    Teledyne WPS CSV exports in this dataset contain Latin-1 microsecond bytes.
    Try UTF-8 first for portability, then fall back to Latin-1.
    """
    for encoding in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="latin1")


MAC_RE = re.compile(r"([0-9a-f]{2}(?::[0-9a-f]{2}){5})", re.IGNORECASE)


def extract_mac(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    m = MAC_RE.search(str(value))
    return m.group(1).lower() if m else None


def extract_labeled_mac(row: pd.Series, label: str) -> Optional[str]:
    wanted = f"({label.upper()})"
    for col in ["addresses", "receive_addr", "transmit_addr", "address_3"]:
        if col not in row.index or pd.isna(row[col]):
            continue
        text = str(row[col])
        if wanted in text.upper():
            return extract_mac(text)
    return None


def derive_mac_fields(df: pd.DataFrame) -> pd.DataFrame:
    addr_cols = [c for c in ["receive_addr", "transmit_addr", "address_3"] if c in df.columns]
    for c in addr_cols:
        df[f"{c}_mac"] = df[c].map(extract_mac)

    if addr_cols:
        df["source_addr"] = df.apply(lambda r: extract_labeled_mac(r, "SA"), axis=1)
        df["destination_addr"] = df.apply(lambda r: extract_labeled_mac(r, "DA"), axis=1)
        df["bssid"] = df.apply(lambda r: extract_labeled_mac(r, "BSSID"), axis=1)

        if "transmit_addr_mac" in df.columns:
            df["source_addr"] = df["source_addr"].fillna(df["transmit_addr_mac"])
        if "receive_addr_mac" in df.columns:
            df["destination_addr"] = df["destination_addr"].fillna(df["receive_addr_mac"])
        if "address_3_mac" in df.columns:
            df["bssid"] = df["bssid"].fillna(df["address_3_mac"])

    return df


def load_csv_clean(path: Path) -> pd.DataFrame:
    df = read_wps_csv(path)
    df = normalize_columns(df)

    # Standard cleanup
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed_")], errors="ignore")

    # Common fields
    if "frame_num" in df.columns:
        df["frame_num"] = pd.to_numeric(df["frame_num"], errors="coerce").astype("Int64")
    elif "framenum" in df.columns:
        df["frame_num"] = pd.to_numeric(df["framenum"], errors="coerce").astype("Int64")
        df = df.drop(columns=["framenum"])
    elif "frame" in df.columns:
        df["frame_num"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")

    if "timestamp" in df.columns:
        df["timestamp"] = parse_timestamp_series(df["timestamp"])

    if "delta" in df.columns:
        df["delta"] = parse_numeric_series(df["delta"])

    if "duration" in df.columns:
        df["duration_us"] = parse_duration_to_us(df["duration"])

    # Numeric columns commonly seen in your exports
    for col in [
        "data_rate_mb_s",
        "frame_size",
        "channel",
        "freq",
        "bookmark",
        "ssi",
        "seq_num",
        "tods",
        "fromds",
        "frag",
        "retry",
        "more",
        "protected",
        "source_port",
        "destination_port",
        "length",
    ]:
        if col in df.columns:
            df[col] = parse_numeric_series(df[col])

    # Example: antenna column may actually contain RSSI like "-85 dBm"
    if "antenna" in df.columns:
        df["signal_dbm"] = parse_numeric_series(df["antenna"])

    if "bad_fcs" in df.columns:
        df["bad_fcs"] = (
            df["bad_fcs"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )

    df = derive_mac_fields(df)

    return df


# ============================================================
# Dataset discovery
# ============================================================

def discover_sessions(root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Expects folders like:
      dataset/
        02_04_1/
          02_04_tap.csv
          02_04_mac.csv
          02_04_error.csv
          02_04_udp.csv
    """
    sessions: Dict[str, Dict[str, Path]] = {}

    for session_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
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
        if files:
            sessions[session_dir.name] = files

    return sessions


# ============================================================
# Merge logic
# ============================================================

def prefix_nonkey_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    new_cols = {}
    for c in df.columns:
        if c == "frame_num":
            continue
        new_cols[c] = f"{prefix}_{c}"
    return df.rename(columns=new_cols)


def first_available(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    available = [c for c in candidates if c in df.columns]
    if not available:
        return None
    out = df[available[0]]
    for c in available[1:]:
        out = out.combine_first(df[c])
    return out


def coalesce_to_column(df: pd.DataFrame, target: str, candidates: List[str]) -> None:
    values = first_available(df, candidates)
    if values is not None:
        df[target] = values


def flag_series_as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).eq(1)

    text = s.astype("string").str.strip().str.lower()
    return (
        text.isin(["true", "t", "yes", "y", "1"])
        | text.str.contains("retry|retrans", na=False)
    )


def add_mac_retry_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "mac_present" in out.columns:
        mac_present = out["mac_present"].eq(True)
    elif "mac_retry" in out.columns:
        mac_present = out["mac_retry"].notna()
    elif "retry" in out.columns:
        mac_present = out["retry"].notna()
    else:
        mac_present = pd.Series(False, index=out.index)

    retry_col = "mac_retry" if "mac_retry" in out.columns else "retry" if "retry" in out.columns else None
    if retry_col:
        retry_values = out[retry_col]
        retry_has_info = retry_values.notna()
        mac_retry_flag = flag_series_as_bool(retry_values) & mac_present & retry_has_info
        mac_row_flag = mac_present & retry_has_info
    else:
        mac_retry_flag = pd.Series(False, index=out.index)
        mac_row_flag = pd.Series(False, index=out.index)

    out["mac_row_flag"] = mac_row_flag.fillna(False).astype(bool)
    out["mac_retry_flag"] = mac_retry_flag.fillna(False).astype(bool)
    out["retry_flag"] = out["mac_retry_flag"]
    return out


def build_flow_key(df: pd.DataFrame) -> pd.Series:
    if {"source_addr", "destination_addr"}.issubset(df.columns):
        base = df["source_addr"].fillna("?") + ">" + df["destination_addr"].fillna("?")
    else:
        base = pd.Series("?", index=df.index, dtype="string")

    if {"udp_source_port", "udp_destination_port"}.issubset(df.columns):
        ports = (
            df["udp_source_port"].astype("Int64").astype("string").fillna("?")
            + ">"
            + df["udp_destination_port"].astype("Int64").astype("string").fillna("?")
        )
        has_ports = df["udp_source_port"].notna() | df["udp_destination_port"].notna()
        base = base.where(~has_ports, base + " udp:" + ports)

    return base.replace({"?>?": pd.NA, "?>? udp:?>?": pd.NA})


def add_unified_wireless_fields(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()

    coalesce_to_column(merged, "timestamp", ["tap_timestamp", "mac_timestamp", "udp_timestamp", "error_timestamp"])
    coalesce_to_column(merged, "frame_size_bytes", ["tap_frame_size", "mac_frame_size", "udp_frame_size", "error_frame_size"])
    coalesce_to_column(merged, "duration_us", ["tap_duration_us"])
    coalesce_to_column(merged, "signal_dbm", ["tap_signal_dbm"])
    coalesce_to_column(merged, "data_rate_mbps", ["tap_data_rate_mb_s"])
    coalesce_to_column(merged, "channel", ["tap_channel"])
    coalesce_to_column(merged, "freq_mhz", ["tap_freq"])
    coalesce_to_column(merged, "mac_type", ["mac_type", "error_type"])
    coalesce_to_column(merged, "mac_subtype", ["mac_subtype", "error_subtype"])
    coalesce_to_column(merged, "source_addr", ["mac_source_addr", "error_source_addr"])
    coalesce_to_column(merged, "destination_addr", ["mac_destination_addr", "error_destination_addr"])
    coalesce_to_column(merged, "bssid", ["mac_bssid", "error_bssid"])

    if "tap_bad_fcs" in merged.columns:
        merged["bad_fcs_flag"] = merged["tap_bad_fcs"].eq(True)
    else:
        merged["bad_fcs_flag"] = False

    for key in ["tap", "mac", "error", "udp"]:
        present_col = f"{key}_present"
        if present_col in merged.columns:
            merged[present_col] = merged[present_col].eq(True)

    merged["has_error_row"] = merged.get("error_present", pd.Series(False, index=merged.index)).eq(True)
    merged["has_udp_row"] = merged.get("udp_present", pd.Series(False, index=merged.index)).eq(True)

    merged = add_mac_retry_metric_columns(merged)

    if "mac_type" in merged.columns:
        merged["data_frame_flag"] = merged["mac_type"].astype("string").str.lower().eq("data")
    else:
        merged["data_frame_flag"] = False

    if {"source_addr", "destination_addr"}.issubset(merged.columns):
        merged["flow_key"] = build_flow_key(merged)

    merged = merged.sort_values(["timestamp", "frame_num"], na_position="last").reset_index(drop=True)

    if "timestamp" in merged.columns:
        merged["iat_s"] = merged["timestamp"].diff().dt.total_seconds()
        if "flow_key" in merged.columns:
            valid_flow = merged["flow_key"].notna()
            merged["flow_iat_s"] = np.nan
            merged.loc[valid_flow, "flow_iat_s"] = (
                merged.loc[valid_flow].groupby("flow_key")["timestamp"].diff().dt.total_seconds()
            )
            merged["analysis_iat_s"] = merged["flow_iat_s"].combine_first(merged["iat_s"])
        else:
            merged["analysis_iat_s"] = merged["iat_s"]

    return merged


def merge_session_tables(session_files: Dict[str, Path]) -> pd.DataFrame:
    tables = {}
    for key, path in session_files.items():
        df = load_csv_clean(path)
        if "frame_num" not in df.columns:
            raise ValueError(f"{path} does not have a frame/frame_num column after cleaning.")
        df["present"] = True
        tables[key] = prefix_nonkey_columns(df, key)

    merged: Optional[pd.DataFrame] = None
    for key in ["tap", "mac", "error", "udp"]:
        if key not in tables:
            continue
        if merged is None:
            merged = tables[key]
        else:
            merged = merged.merge(tables[key], on="frame_num", how="outer")

    if merged is None:
        raise ValueError("No usable tables found for session.")

    return add_unified_wireless_fields(merged)


# ============================================================
# Session summary for prompting
# ============================================================

def summarize_dataframe_schema(df: pd.DataFrame, max_cols: int = 80) -> Dict[str, Any]:
    cols = []
    for c in df.columns[:max_cols]:
        cols.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "nonnull": int(df[c].notna().sum())
        })

    time_min = None
    time_max = None
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        time_min = str(df["timestamp"].min())
        time_max = str(df["timestamp"].max())

    return {
        "n_rows": int(len(df)),
        "time_min": time_min,
        "time_max": time_max,
        "columns": cols,
    }


# ============================================================
# LLM planning
# ============================================================

PLAN_SCHEMA = {
    "name": "traffic_analysis_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "task_type": {
                "type": "string",
                "enum": [
                    "burst_detection",
                    "delay_jitter_analysis",
                    "error_spike_analysis",
                    "anomaly_detection",
                    "top_entities",
                    "timeline_summary",
                    "custom_filter_aggregate"
                ]
            },
            "time_bucket": {
                "type": "string",
                "enum": ["1s", "5s", "10s", "30s", "60s"]
            },
            "group_by": {
                "type": "array",
                "items": {"type": "string"}
            },
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "op": {
                            "type": "string",
                            "enum": ["eq", "ne", "gt", "ge", "lt", "le", "contains", "in", "notnull"]
                        },
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "integer"},
                                {"type": "boolean"},
                                {
                                    "type": "array",
                                    "items": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {"type": "number"},
                                            {"type": "integer"},
                                            {"type": "boolean"}
                                        ]
                                    }
                                },
                                {"type": "null"}
                            ]
                        }
                    },
                    "required": ["column", "op", "value"],
                    "additionalProperties": False
                }
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "packet_count",
                        "bytes_sum",
                        "mean_iat",
                        "std_iat",
                        "p95_iat",
                        "mean_duration_us",
                        "bad_fcs_count",
                        "bad_fcs_rate",
                        "error_frame_count",
                        "error_frame_rate",
                        "retry_count",
                        "retry_rate",
                        "data_frame_count",
                        "mean_signal_dbm",
                        "mean_rate_mbps"
                    ]
                }
            },
            "top_k": {"type": "integer"},
            "plot": {
                "type": "string",
                "enum": ["line", "bar", "none"]
            },
            "question_rephrased": {"type": "string"},
            "explanation_focus": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "bursty_traffic",
                        "delay",
                        "jitter",
                        "errors",
                        "link_quality",
                        "protocol_behavior"
                    ]
                }
            }
        },
        "required": [
            "session_id",
            "task_type",
            "time_bucket",
            "group_by",
            "filters",
            "metrics",
            "top_k",
            "plot",
            "question_rephrased",
            "explanation_focus"
        ],
        "additionalProperties": False
    }
}


def choose_default_session(session_ids: List[str]) -> str:
    return sorted(session_ids)[-1]


def choose_session_for_question(question: str, session_ids: List[str]) -> str:
    q = question.lower()
    for session_id in sorted(session_ids, key=len, reverse=True):
        if session_id.lower() in q:
            return session_id
    return choose_default_session(session_ids)


def infer_cross_session_mode(question: str) -> Optional[str]:
    q = question.lower()
    mentions_session_compare = any(
        phrase in q for phrase in [
            "which session",
            "across sessions",
            "compare sessions",
            "among sessions",
            "all sessions",
        ]
    )
    if not mentions_session_compare:
        return None

    if "stable" in q or "stability" in q:
        return "weakest"
    if any(word in q for word in ["strongest", "highest", "most", "largest", "worst", "max"]):
        return "strongest"
    if any(word in q for word in ["weakest", "lowest", "least", "smallest", "best", "min"]):
        return "weakest"
    return "rank"


def infer_comparison_metric(question: str, task_type: Optional[str] = None) -> Optional[str]:
    q = question.lower()

    if "burst" in q or "bursty" in q:
        return "peak_burst_z"

    if ("stable" in q or "stability" in q) and any(term in q for term in ["traffic", "pattern", "packet rate"]):
        return "packet_rate_variance"

    if (
        "longest inter-arrival gap" in q
        or "longest interarrival gap" in q
        or "longest iat" in q
        or ("inter-arrival" in q and any(term in q for term in ["longest", "largest", "max", "maximum"]))
        or ("interarrival" in q and any(term in q for term in ["longest", "largest", "max", "maximum"]))
    ):
        return "max_iat_s"

    if "retry rate" in q:
        return "retry_rate"

    avg_packet_size_phrases = [
        "average packet size",
        "avg packet size",
        "mean packet size",
        "average frame size",
        "avg frame size",
        "mean frame size",
    ]
    if any(phrase in q for phrase in avg_packet_size_phrases):
        return "avg_packet_size"

    if any(phrase in q for phrase in ["packets per second", "packet rate", "pps"]):
        return "packets_per_second"

    if any(phrase in q for phrase in ["bytes per second", "byte rate"]):
        return "bytes_per_second"

    if "bad fcs rate" in q or "fcs rate" in q:
        return "bad_fcs_rate"

    if "jitter score" in q:
        return "jitter_score"

    task_defaults = {
        "burst_detection": "peak_burst_z",
        "delay_jitter_analysis": "jitter_score",
        "error_spike_analysis": "peak_error_z",
    }
    return task_defaults.get(task_type or "")


def comparison_ascending(question: str, metric: Optional[str], mode: str) -> bool:
    q = question.lower()

    if metric == "packet_rate_variance":
        return True
    if metric == "max_iat_s":
        return False
    if metric == "peak_burst_z":
        return mode == "weakest" or any(word in q for word in ["weakest", "lowest", "least", "smallest"])
    if metric == "retry_rate" and any(word in q for word in ["lowest", "least", "minimum", "min", "best"]):
        return True

    if any(word in q for word in ["lowest", "least", "smallest", "minimum", "min"]):
        return True
    if any(word in q for word in ["longest", "highest", "largest", "maximum", "max", "most", "strongest", "worst"]):
        return False
    return mode == "weakest"


def should_run_local_anomaly_detection(question: str, known_start: Optional[str] = None, known_end: Optional[str] = None) -> bool:
    q = question.lower()
    anomaly_terms = ["anomaly", "anomalies", "detector", "detectors", "isolationforest", "isolation forest"]
    return bool(known_start or known_end or any(term in q for term in anomaly_terms))


def should_run_local_burst_dominance(question: str) -> bool:
    q = question.lower()
    dominance_terms = ["dominate", "dominant", "top", "which channels", "which devices", "devices", "channel"]
    return "burst" in q and any(term in q for term in dominance_terms)


def build_local_anomaly_plan(question: str, session_ids: List[str]) -> Dict[str, Any]:
    return {
        "session_id": choose_session_for_question(question, session_ids),
        "task_type": "anomaly_detection",
        "time_bucket": "1s",
        "group_by": [],
        "filters": [],
        "metrics": ANOMALY_FEATURE_COLUMNS,
        "top_k": 20,
        "plot": "line",
        "question_rephrased": question,
        "explanation_focus": ["bursty_traffic", "delay", "jitter", "errors", "link_quality"],
        "session_scope": "single_session",
        "comparison_mode": "none",
        "comparison_metric": None,
    }


def build_local_burst_dominance_plan(question: str, session_ids: List[str]) -> Dict[str, Any]:
    return {
        "session_id": choose_session_for_question(question, session_ids),
        "task_type": "burst_detection",
        "time_bucket": "1s",
        "group_by": [],
        "filters": [],
        "metrics": ["packet_count", "bytes_sum"],
        "top_k": 20,
        "plot": "line",
        "question_rephrased": question,
        "explanation_focus": ["bursty_traffic", "protocol_behavior", "link_quality"],
        "session_scope": "single_session",
        "comparison_mode": "none",
        "comparison_metric": None,
    }


def build_local_comparison_plan(question: str, session_ids: List[str]) -> Optional[Dict[str, Any]]:
    comparison_mode = infer_cross_session_mode(question)
    if not comparison_mode:
        return None

    metric = infer_comparison_metric(question)
    if metric not in {"packet_rate_variance", "max_iat_s", "retry_rate", "peak_burst_z"}:
        return None

    ascending = comparison_ascending(question, metric, comparison_mode)
    task_type = "burst_detection" if metric == "peak_burst_z" else "timeline_summary"
    metrics = ["packet_count", "bytes_sum"] if metric == "peak_burst_z" else ["packet_count", "retry_rate"]
    return {
        "session_id": "ALL_SESSIONS",
        "task_type": task_type,
        "time_bucket": "1s",
        "group_by": [],
        "filters": [],
        "metrics": metrics,
        "top_k": 20,
        "plot": "bar",
        "question_rephrased": question,
        "explanation_focus": ["bursty_traffic", "delay", "jitter", "errors", "protocol_behavior"],
        "session_scope": "all_sessions",
        "comparison_mode": "weakest" if ascending else "strongest",
        "comparison_metric": metric,
    }


def ask_llm_for_plan(question: str, session_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    available_sessions = sorted(session_summaries.keys())
    default_session = choose_default_session(available_sessions)
    comparison_mode = infer_cross_session_mode(question)

    prompt = f"""
You are helping analyze wireless trace tables that were already exported from Teledyne LeCroy.
Convert the user's question into a conservative JSON analysis plan.

Rules:
- Pick one session_id from: {available_sessions}
- If the user did not specify a session, use "{default_session}"
- Only use columns that plausibly exist in these session summaries
- Prefer packet_count, bytes_sum, mean_iat, std_iat, p95_iat for burst/delay/jitter questions
- Prefer bad_fcs_count, bad_fcs_rate, error_frame_count, retry_rate, or error_frame_rate for error questions
- Prefer anomaly_detection for explicit anomaly-detector or known-anomaly validation questions
- For questions asking which channels or devices dominate during bursts, use burst_detection; its result includes dominant_channel plus dominant_source_addr, dominant_destination_addr, and dominant_bssid fields when those columns exist. source_addr is derived from Transmit Addr and destination_addr from Receive Addr.
- Prefer line plot for time-series questions, bar plot for top-k questions
- Keep the plan simple and executable locally with pandas
- group_by may be empty or use concrete columns like source_addr, destination_addr, bssid, mac_type, mac_subtype, channel, or UDP ports if present
- filters should be minimal and safe
- If the question compares sessions, treat this as a cross-session comparison. Still emit one valid session_id placeholder such as "{default_session}", but use task_type/metrics that support comparing every session locally.

Session summaries:
{json.dumps(session_summaries, indent=2)}

User question:
{question}
""".strip()

    resp = get_openai_client().responses.create(
        model=MODEL_PLAN,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": PLAN_SCHEMA["name"],
                "strict": PLAN_SCHEMA["strict"],
                "schema": PLAN_SCHEMA["schema"],
            }
        },
    )

    plan = json.loads(resp.output_text)
    if comparison_mode:
        plan["session_scope"] = "all_sessions"
        plan["comparison_mode"] = comparison_mode
        plan["comparison_metric"] = infer_comparison_metric(question, plan.get("task_type"))
        plan["plot"] = "bar"
    else:
        plan["session_scope"] = "single_session"
        plan["comparison_mode"] = "none"
        plan["comparison_metric"] = None
    return plan


# ============================================================
# Local execution engine
# ============================================================

def apply_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()

    for f in filters:
        col = f["column"]
        op = f["op"]
        value = f["value"]

        if col not in out.columns:
            continue

        s = out[col]

        if op == "eq":
            out = out[s == value]
        elif op == "ne":
            out = out[s != value]
        elif op == "gt":
            out = out[pd.to_numeric(s, errors="coerce") > float(value)]
        elif op == "ge":
            out = out[pd.to_numeric(s, errors="coerce") >= float(value)]
        elif op == "lt":
            out = out[pd.to_numeric(s, errors="coerce") < float(value)]
        elif op == "le":
            out = out[pd.to_numeric(s, errors="coerce") <= float(value)]
        elif op == "contains":
            out = out[s.astype(str).str.contains(str(value), case=False, na=False)]
        elif op == "in":
            if not isinstance(value, list):
                value = [value]
            out = out[s.isin(value)]
        elif op == "notnull":
            out = out[s.notna()]

    return out


def ensure_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = add_mac_retry_metric_columns(df)

    if "bad_fcs_flag" not in out.columns:
        out["bad_fcs_flag"] = False
    if "has_error_row" not in out.columns:
        out["has_error_row"] = False
    if "data_frame_flag" not in out.columns:
        out["data_frame_flag"] = False
    if "frame_size_bytes" not in out.columns:
        out["frame_size_bytes"] = np.nan
    if "iat_s" not in out.columns:
        out["iat_s"] = np.nan
    if "analysis_iat_s" not in out.columns:
        out["analysis_iat_s"] = out["iat_s"]
    if "duration_us" not in out.columns and "tap_duration_us" in out.columns:
        out["duration_us"] = pd.to_numeric(out["tap_duration_us"], errors="coerce")
    if "signal_dbm" not in out.columns:
        out["signal_dbm"] = np.nan
    if "data_rate_mbps" not in out.columns:
        out["data_rate_mbps"] = np.nan

    return out


def aggregate_metrics(df: pd.DataFrame, metrics: List[str], group_cols: List[str]) -> pd.DataFrame:
    df = ensure_metric_columns(df)

    agg_map = {}
    if "packet_count" in metrics:
        agg_map["frame_num"] = ("frame_num", "count")
    if "bytes_sum" in metrics:
        agg_map["bytes_sum"] = ("frame_size_bytes", "sum")
    if "mean_iat" in metrics:
        agg_map["mean_iat"] = ("analysis_iat_s", "mean")
    if "std_iat" in metrics:
        agg_map["std_iat"] = ("analysis_iat_s", "std")
    if "p95_iat" in metrics:
        agg_map["p95_iat"] = ("analysis_iat_s", lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan)
    if "mean_duration_us" in metrics:
        if "duration_us" in df.columns:
            agg_map["mean_duration_us"] = ("duration_us", "mean")
        elif "tap_duration_us" in df.columns:
            agg_map["mean_duration_us"] = ("tap_duration_us", "mean")
    if "bad_fcs_count" in metrics:
        agg_map["bad_fcs_count"] = ("bad_fcs_flag", "sum")
    if "bad_fcs_rate" in metrics:
        agg_map["bad_fcs_rate"] = ("bad_fcs_flag", "mean")
    if "error_frame_count" in metrics:
        agg_map["error_frame_count"] = ("has_error_row", "sum")
    if "error_frame_rate" in metrics:
        agg_map["error_frame_rate"] = ("has_error_row", "mean")
    needs_retry_metrics = any(m in metrics for m in ["mac_rows", "retry_count", "retry_rate"])
    if needs_retry_metrics:
        agg_map["mac_rows"] = ("mac_row_flag", "sum")
        agg_map["retry_count"] = ("mac_retry_flag", "sum")
    if "data_frame_count" in metrics:
        agg_map["data_frame_count"] = ("data_frame_flag", "sum")
    if "mean_signal_dbm" in metrics:
        agg_map["mean_signal_dbm"] = ("signal_dbm", "mean")
    if "mean_rate_mbps" in metrics:
        agg_map["mean_rate_mbps"] = ("data_rate_mbps", "mean")

    if not group_cols:
        row = {}
        if "packet_count" in metrics:
            row["packet_count"] = int(df["frame_num"].count())
        if "bytes_sum" in metrics:
            row["bytes_sum"] = float(df["frame_size_bytes"].sum(skipna=True))
        if "mean_iat" in metrics:
            row["mean_iat"] = float(df["analysis_iat_s"].mean(skipna=True))
        if "std_iat" in metrics:
            row["std_iat"] = float(df["analysis_iat_s"].std(skipna=True))
        if "p95_iat" in metrics:
            row["p95_iat"] = float(np.nanpercentile(df["analysis_iat_s"].dropna(), 95)) if df["analysis_iat_s"].notna().any() else np.nan
        if "mean_duration_us" in metrics:
            src = "duration_us" if "duration_us" in df.columns else "tap_duration_us"
            row["mean_duration_us"] = float(pd.to_numeric(df[src], errors="coerce").mean(skipna=True))
        if "bad_fcs_count" in metrics:
            row["bad_fcs_count"] = int(pd.Series(df["bad_fcs_flag"]).fillna(False).sum())
        if "bad_fcs_rate" in metrics:
            row["bad_fcs_rate"] = float(pd.Series(df["bad_fcs_flag"]).fillna(False).mean())
        if "error_frame_count" in metrics:
            row["error_frame_count"] = int(pd.Series(df["has_error_row"]).fillna(False).sum())
        if "error_frame_rate" in metrics:
            row["error_frame_rate"] = float(pd.Series(df["has_error_row"]).fillna(False).mean())
        if needs_retry_metrics:
            mac_rows = int(pd.Series(df["mac_row_flag"]).fillna(False).sum())
            retry_count = int(pd.Series(df["mac_retry_flag"]).fillna(False).sum())
            row["mac_rows"] = mac_rows
            row["retry_count"] = retry_count
            if "retry_rate" in metrics:
                row["retry_rate"] = float(retry_count / mac_rows) if mac_rows else float("nan")
        if "data_frame_count" in metrics:
            row["data_frame_count"] = int(pd.Series(df["data_frame_flag"]).fillna(False).sum())
        if "mean_signal_dbm" in metrics:
            row["mean_signal_dbm"] = float(df["signal_dbm"].mean(skipna=True))
        if "mean_rate_mbps" in metrics:
            row["mean_rate_mbps"] = float(df["data_rate_mbps"].mean(skipna=True))
        return pd.DataFrame([row])

    grouped = df.groupby(group_cols, dropna=False).agg(**agg_map).reset_index()
    if "packet_count" not in grouped.columns and "frame_num" in grouped.columns:
        grouped = grouped.rename(columns={"frame_num": "packet_count"})
    if needs_retry_metrics and "retry_rate" in metrics:
        grouped["retry_rate"] = grouped["retry_count"] / grouped["mac_rows"].replace(0, np.nan)
    return grouped


def bucket_time(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns or out["timestamp"].isna().all():
        raise ValueError("No usable timestamp column found for time bucketing.")
    out["time_bucket"] = out["timestamp"].dt.floor(bucket)
    return out


def bucket_seconds(bucket: str) -> float:
    m = re.fullmatch(r"(\d+)\s*s", bucket.strip().lower())
    return float(m.group(1)) if m else 1.0


def order_existing_columns(df: pd.DataFrame, preferred_order: List[str]) -> pd.DataFrame:
    ordered = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    return df[ordered + remaining]


def top_value_stats(group: pd.DataFrame, column: str) -> Dict[str, Any]:
    values = group[column].dropna()
    if values.empty:
        return {
            f"{column}_known_count": 0,
            f"dominant_{column}": None,
            f"dominant_{column}_packet_count": 0,
            f"dominant_{column}_packet_share": np.nan,
        }

    counts = values.value_counts(dropna=True)
    top_value = counts.index[0]
    if hasattr(top_value, "item"):
        top_value = top_value.item()
    top_count = int(counts.iloc[0])
    return {
        f"{column}_known_count": int(len(values)),
        f"dominant_{column}": top_value,
        f"dominant_{column}_packet_count": top_count,
        f"dominant_{column}_packet_share": float(top_count / len(values)) if len(values) else np.nan,
    }


def add_dominant_traffic_fields(bucketed: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    dominant_columns = [
        c for c in ["channel", "source_addr", "destination_addr", "bssid"]
        if c in bucketed.columns
    ]
    if not dominant_columns:
        return result

    rows = []
    for time_bucket, group in bucketed.groupby("time_bucket", dropna=False):
        row: Dict[str, Any] = {"time_bucket": time_bucket}
        for column in dominant_columns:
            row.update(top_value_stats(group, column))
        rows.append(row)

    dominant = pd.DataFrame(rows)
    return result.merge(dominant, on="time_bucket", how="left")


def detect_bursts(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    b = bucket_time(ensure_metric_columns(df), bucket)
    g = b.groupby("time_bucket", dropna=False).agg(
        packet_count=("frame_num", "count"),
        mac_rows=("mac_row_flag", "sum"),
        data_frame_count=("data_frame_flag", "sum"),
        bytes_sum=("frame_size_bytes", "sum"),
        bad_fcs_count=("bad_fcs_flag", lambda s: pd.Series(s).fillna(False).sum()),
        error_frame_count=("has_error_row", lambda s: pd.Series(s).fillna(False).sum()),
        retry_count=("mac_retry_flag", lambda s: pd.Series(s).fillna(False).sum()),
        mean_signal_dbm=("signal_dbm", "mean"),
        mean_rate_mbps=("data_rate_mbps", "mean"),
    ).reset_index()

    g["packets_per_s"] = g["packet_count"] / bucket_seconds(bucket)
    g["bad_fcs_rate"] = g["bad_fcs_count"] / g["packet_count"].replace(0, np.nan)
    g["error_frame_rate"] = g["error_frame_count"] / g["packet_count"].replace(0, np.nan)
    g["retry_rate"] = g["retry_count"] / g["mac_rows"].replace(0, np.nan)

    # z-score burst indicator
    count_mean = g["packet_count"].mean()
    count_std = g["packet_count"].std(ddof=0)
    if pd.notna(count_std) and count_std > 0:
        g["burst_z"] = (g["packet_count"] - count_mean) / count_std
    else:
        g["burst_z"] = 0.0

    g["is_burst"] = g["burst_z"] >= 2.0
    g = add_dominant_traffic_fields(b, g)
    return g


def detect_delay_jitter(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    b = bucket_time(ensure_metric_columns(df), bucket)
    g = b.groupby("time_bucket", dropna=False).agg(
        packet_count=("frame_num", "count"),
        mac_rows=("mac_row_flag", "sum"),
        data_frame_count=("data_frame_flag", "sum"),
        mean_iat=("analysis_iat_s", "mean"),
        std_iat=("analysis_iat_s", "std"),
        p95_iat=("analysis_iat_s", lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan),
        mean_global_iat=("iat_s", "mean"),
        mean_flow_iat=("flow_iat_s", "mean") if "flow_iat_s" in b.columns else ("analysis_iat_s", "mean"),
        mean_duration_us=("tap_duration_us", "mean") if "tap_duration_us" in b.columns else ("duration_us", "mean"),
        retry_count=("mac_retry_flag", lambda s: pd.Series(s).fillna(False).sum()),
        bad_fcs_count=("bad_fcs_flag", lambda s: pd.Series(s).fillna(False).sum()),
        error_frame_count=("has_error_row", lambda s: pd.Series(s).fillna(False).sum()),
        mean_signal_dbm=("signal_dbm", "mean"),
        mean_rate_mbps=("data_rate_mbps", "mean"),
    ).reset_index()

    # simple jitter score: normalized std_iat
    eps = 1e-9
    g["jitter_score"] = g["std_iat"] / (g["mean_iat"] + eps)
    g["retry_rate"] = g["retry_count"] / g["mac_rows"].replace(0, np.nan)
    g["bad_fcs_rate"] = g["bad_fcs_count"] / g["packet_count"].replace(0, np.nan)
    g["error_frame_rate"] = g["error_frame_count"] / g["packet_count"].replace(0, np.nan)
    return order_existing_columns(g, [
        "time_bucket",
        "packet_count",
        "std_iat",
        "jitter_score",
        "p95_iat",
        "mean_iat",
        "mean_flow_iat",
        "mean_global_iat",
        "mean_duration_us",
        "mean_signal_dbm",
        "mean_rate_mbps",
        "error_frame_count",
        "error_frame_rate",
        "bad_fcs_count",
        "bad_fcs_rate",
        "retry_count",
        "retry_rate",
        "mac_rows",
        "data_frame_count",
    ])


def detect_error_spikes(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    b = bucket_time(ensure_metric_columns(df), bucket)
    g = b.groupby("time_bucket", dropna=False).agg(
        packet_count=("frame_num", "count"),
        mac_rows=("mac_row_flag", "sum"),
        data_frame_count=("data_frame_flag", "sum"),
        bad_fcs_count=("bad_fcs_flag", lambda s: pd.Series(s).fillna(False).sum()),
        error_frame_count=("has_error_row", lambda s: pd.Series(s).fillna(False).sum()),
        retry_count=("mac_retry_flag", lambda s: pd.Series(s).fillna(False).sum()),
        mean_signal_dbm=("signal_dbm", "mean"),
        mean_rate_mbps=("data_rate_mbps", "mean"),
    ).reset_index()
    g["bad_fcs_rate"] = g["bad_fcs_count"] / g["packet_count"].replace(0, np.nan)
    g["error_frame_rate"] = g["error_frame_count"] / g["packet_count"].replace(0, np.nan)
    g["retry_rate"] = g["retry_count"] / g["mac_rows"].replace(0, np.nan)

    count_mean = g["error_frame_count"].mean()
    count_std = g["error_frame_count"].std(ddof=0)
    if pd.notna(count_std) and count_std > 0:
        g["error_z"] = (g["error_frame_count"] - count_mean) / count_std
    else:
        g["error_z"] = 0.0
    g["is_error_spike"] = (g["error_z"] >= 2.0) | (g["bad_fcs_rate"] >= 0.2)
    return g


ANOMALY_FEATURE_COLUMNS = [
    "packet_count",
    "bytes_sum",
    "packets_per_s",
    "mean_iat",
    "std_iat",
    "p95_iat",
    "jitter_score",
    "bad_fcs_rate",
    "error_frame_rate",
    "retry_rate",
    "mean_signal_dbm",
    "mean_rate_mbps",
]


def build_anomaly_feature_table(df: pd.DataFrame, bucket: str = "1s") -> pd.DataFrame:
    """
    Build one row per time bucket with deterministic local features for anomaly detectors.
    """
    b = bucket_time(ensure_metric_columns(df), bucket)
    g = b.groupby("time_bucket", dropna=False).agg(
        packet_count=("frame_num", "count"),
        mac_rows=("mac_row_flag", "sum"),
        bytes_sum=("frame_size_bytes", "sum"),
        mean_iat=("analysis_iat_s", "mean"),
        std_iat=("analysis_iat_s", "std"),
        p95_iat=("analysis_iat_s", lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan),
        bad_fcs_count=("bad_fcs_flag", lambda s: pd.Series(s).fillna(False).sum()),
        error_frame_count=("has_error_row", lambda s: pd.Series(s).fillna(False).sum()),
        retry_count=("mac_retry_flag", lambda s: pd.Series(s).fillna(False).sum()),
        mean_signal_dbm=("signal_dbm", "mean"),
        mean_rate_mbps=("data_rate_mbps", "mean"),
    ).reset_index()

    seconds = bucket_seconds(bucket)
    g["packets_per_s"] = g["packet_count"] / seconds
    g["bad_fcs_rate"] = g["bad_fcs_count"] / g["packet_count"].replace(0, np.nan)
    g["error_frame_rate"] = g["error_frame_count"] / g["packet_count"].replace(0, np.nan)
    g["retry_rate"] = g["retry_count"] / g["mac_rows"].replace(0, np.nan)

    # Jitter score follows the existing local metric: inter-arrival std normalized by mean IAT.
    eps = 1e-9
    g["jitter_score"] = g["std_iat"] / (g["mean_iat"] + eps)

    ordered = ["time_bucket"] + ANOMALY_FEATURE_COLUMNS + [
        "bad_fcs_count",
        "error_frame_count",
        "mac_rows",
        "retry_count",
    ]
    return g[[c for c in ordered if c in g.columns]]


def detect_robust_zscore(
    features: pd.DataFrame,
    columns: List[str],
    threshold: float = 3.5,
) -> pd.DataFrame:
    out = features.copy()
    z_cols = []

    for col in columns:
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce")
        median = x.median(skipna=True)
        mad = (x - median).abs().median(skipna=True)
        z_col = f"{col}_robust_z"

        if pd.notna(mad) and mad > 0:
            # Robust z-score = 0.6745 * (x - median) / MAD.
            out[z_col] = 0.6745 * (x - median) / mad
        else:
            out[z_col] = 0.0
        z_cols.append(z_col)

    if z_cols:
        out["robust_zscore_score"] = out[z_cols].abs().max(axis=1).fillna(0.0)
    else:
        out["robust_zscore_score"] = 0.0
    out["robust_zscore_anomaly"] = out["robust_zscore_score"] >= threshold
    return out


def detect_rolling_shift(
    features: pd.DataFrame,
    column: str,
    baseline_window: int = 5,
    current_window: int = 3,
    threshold: float = 3.0,
) -> pd.DataFrame:
    out = features.copy()
    if column not in out.columns:
        out["rolling_shift_directional_score"] = 0.0
        out["rolling_shift_score"] = 0.0
        out["rolling_shift_anomaly"] = False
        return out

    x = pd.to_numeric(out[column], errors="coerce")
    current_mean = x.rolling(current_window, min_periods=current_window).mean()
    previous = x.shift(current_window)
    baseline_mean = previous.rolling(baseline_window, min_periods=baseline_window).mean()
    baseline_std = previous.rolling(baseline_window, min_periods=baseline_window).std(ddof=0)

    # Rolling shift = difference between current rolling mean and previous baseline,
    # normalized by baseline rolling std.
    directional = (current_mean - baseline_mean) / baseline_std.replace(0, np.nan)
    out["rolling_shift_directional_score"] = directional.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["rolling_shift_score"] = out["rolling_shift_directional_score"].abs()
    out["rolling_shift_anomaly"] = out["rolling_shift_score"] >= threshold
    return out


def detect_isolation_forest(
    features: pd.DataFrame,
    columns: List[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    out = features.copy()
    available = [c for c in columns if c in out.columns]
    if not available:
        out["isolation_forest_score"] = 0.0
        out["isolation_forest_anomaly"] = False
        return out
    if IsolationForest is None:
        raise ImportError("scikit-learn is required for IsolationForest anomaly detection.")

    x = out[available].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    if len(x) < 2:
        out["isolation_forest_score"] = 0.0
        out["isolation_forest_anomaly"] = False
        return out

    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    safe_contamination = min(max(float(contamination), 1.0 / max(len(x), 2)), 0.5)

    # IsolationForest gives an unsupervised multivariate anomaly score.
    model = IsolationForest(contamination=safe_contamination, random_state=random_state)
    labels = model.fit_predict(x)
    out["isolation_forest_score"] = -model.decision_function(x)
    out["isolation_forest_anomaly"] = labels == -1
    return out


def _format_metric_value(value: Any) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(val):
        return "nan"
    if abs(val) >= 100:
        return f"{val:.1f}"
    if abs(val) >= 1:
        return f"{val:.3f}"
    return f"{val:.4f}"


def add_combined_anomaly_output(features: pd.DataFrame, detector_columns: List[str]) -> pd.DataFrame:
    out = features.copy()
    flags = [c for c in detector_columns if c in out.columns]
    out["num_detectors_flagged"] = out[flags].fillna(False).astype(bool).sum(axis=1) if flags else 0
    out["any_anomaly"] = out["num_detectors_flagged"] > 0

    robust_z_cols = [c for c in out.columns if c.endswith("_robust_z")]
    base_feature_cols = [c for c in ANOMALY_FEATURE_COLUMNS if c in out.columns]

    def reason(row: pd.Series) -> str:
        if not bool(row.get("any_anomaly", False)):
            return "No detector flagged this bucket."

        parts = []
        if bool(row.get("robust_zscore_anomaly", False)) and robust_z_cols:
            z_values = row[robust_z_cols].abs()
            if z_values.notna().any():
                z_col = str(z_values.idxmax())
                metric = z_col.removesuffix("_robust_z")
                parts.append(f"robust z-score on {metric}={_format_metric_value(row.get(metric))}")

        if bool(row.get("rolling_shift_anomaly", False)):
            parts.append(
                "rolling packet_count shift "
                f"score={_format_metric_value(row.get('rolling_shift_directional_score'))}"
            )

        if bool(row.get("isolation_forest_anomaly", False)):
            deviations = []
            for col in base_feature_cols:
                series = pd.to_numeric(out[col], errors="coerce")
                std = series.std(ddof=0)
                if pd.notna(std) and std > 0:
                    median = series.median(skipna=True)
                    deviations.append((abs((row.get(col, np.nan) - median) / std), col))
            deviations = sorted(deviations, reverse=True)[:2]
            metrics = ", ".join(f"{col}={_format_metric_value(row.get(col))}" for _, col in deviations)
            if metrics:
                parts.append(f"IsolationForest multivariate outlier ({metrics})")
            else:
                parts.append("IsolationForest multivariate outlier")

        return "; ".join(parts[:3]) if parts else "Detector threshold exceeded."

    out["anomaly_reason"] = out.apply(reason, axis=1)
    return out


def apply_known_anomaly_window(
    result: pd.DataFrame,
    known_anomaly_start: Optional[str] = None,
    known_anomaly_end: Optional[str] = None,
) -> pd.DataFrame:
    out = result.copy()
    out["known_anomaly"] = False
    if not known_anomaly_start or not known_anomaly_end:
        return out

    if "time_bucket" not in out.columns:
        raise ValueError("Known anomaly validation requires a time_bucket column.")

    start = pd.to_datetime(known_anomaly_start, errors="raise")
    end = pd.to_datetime(known_anomaly_end, errors="raise")
    if end < start:
        raise ValueError("--known-anomaly-end must be greater than or equal to --known-anomaly-start")

    buckets = pd.to_datetime(out["time_bucket"], errors="coerce")
    out["known_anomaly"] = (buckets >= start) & (buckets <= end)
    return out


def compute_validation_metrics(result: pd.DataFrame) -> Dict[str, Any]:
    if "known_anomaly" not in result.columns or "any_anomaly" not in result.columns:
        return {}

    known = result["known_anomaly"].fillna(False).astype(bool)
    predicted = result["any_anomaly"].fillna(False).astype(bool)
    tp = int((predicted & known).sum())
    fp = int((predicted & ~known).sum())
    fn = int((~predicted & known).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "known_anomaly_buckets": int(known.sum()),
        "predicted_anomaly_buckets": int(predicted.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def run_anomaly_detection(
    df: pd.DataFrame,
    bucket: str = "1s",
    known_anomaly_start: Optional[str] = None,
    known_anomaly_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    features = build_anomaly_feature_table(df, bucket=bucket)
    detector_columns = [c for c in ANOMALY_FEATURE_COLUMNS if c in features.columns]

    result = detect_robust_zscore(features, detector_columns, threshold=3.5)
    result = detect_rolling_shift(result, column="packet_count", baseline_window=5, current_window=3, threshold=3.0)
    result = detect_isolation_forest(result, detector_columns, contamination=0.05, random_state=42)
    result = add_combined_anomaly_output(
        result,
        detector_columns=[
            "robust_zscore_anomaly",
            "rolling_shift_anomaly",
            "isolation_forest_anomaly",
        ],
    )
    result = apply_known_anomaly_window(result, known_anomaly_start, known_anomaly_end)
    validation = compute_validation_metrics(result)

    meta = {
        "feature_rows": int(len(features)),
        "anomaly_rows": int(result["any_anomaly"].fillna(False).sum()) if "any_anomaly" in result.columns else 0,
        "detectors": ["robust_zscore", "rolling_shift", "isolation_forest"],
        "validation": validation,
    }
    return result, meta


def top_entities(df: pd.DataFrame, group_cols: List[str], metrics: List[str], top_k: int) -> pd.DataFrame:
    if not group_cols:
        # try to pick something reasonable
        candidates = [
            "source_addr",
            "destination_addr",
            "bssid",
            "mac_type",
            "mac_subtype",
            "mac_transmit_addr_mac",
            "mac_receive_addr_mac",
            "udp_source_port",
            "udp_destination_port",
            "channel",
        ]
        group_cols = [c for c in candidates if c in df.columns][:1]

    out = aggregate_metrics(df, metrics=metrics, group_cols=group_cols)
    sort_col = "packet_count" if "packet_count" in out.columns else out.columns[-1]
    return out.sort_values(sort_col, ascending=False).head(top_k)


def run_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
    known_anomaly_start: Optional[str] = None,
    known_anomaly_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filtered = apply_filters(df, plan["filters"])

    task_type = plan["task_type"]
    bucket = plan["time_bucket"]
    metrics = plan["metrics"]
    group_by = [g for g in plan["group_by"] if g in filtered.columns]
    top_k = int(plan["top_k"])

    if task_type == "burst_detection":
        result = detect_bursts(filtered, bucket=bucket)
    elif task_type == "delay_jitter_analysis":
        result = detect_delay_jitter(filtered, bucket=bucket)
    elif task_type == "error_spike_analysis":
        result = detect_error_spikes(filtered, bucket=bucket)
    elif task_type == "anomaly_detection":
        result, anomaly_meta = run_anomaly_detection(
            filtered,
            bucket=bucket,
            known_anomaly_start=known_anomaly_start,
            known_anomaly_end=known_anomaly_end,
        )
    elif task_type == "top_entities":
        result = top_entities(filtered, group_cols=group_by, metrics=metrics or ["packet_count"], top_k=top_k)
    elif task_type == "timeline_summary":
        tmp = bucket_time(filtered, bucket)
        group_cols = ["time_bucket"] + group_by
        result = aggregate_metrics(tmp, metrics=metrics, group_cols=group_cols)
    elif task_type == "custom_filter_aggregate":
        result = aggregate_metrics(filtered, metrics=metrics, group_cols=group_by)
        if len(result) > top_k:
            sort_col = "packet_count" if "packet_count" in result.columns else result.columns[-1]
            result = result.sort_values(sort_col, ascending=False).head(top_k)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    meta = {
        "rows_after_filter": int(len(filtered)),
        "result_rows": int(len(result)),
    }
    if task_type == "anomaly_detection":
        meta.update(anomaly_meta)
    return result, meta


def _safe_max(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    return float(series.max()) if series.notna().any() else float("nan")


def _safe_mean(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    return float(series.mean()) if series.notna().any() else float("nan")


def _safe_sum(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    return float(series.sum()) if series.notna().any() else float("nan")


def _divide_or_nan(numerator: Any, denominator: Any) -> float:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return float("nan")
    return num / den


def retry_flag_as_bool(df: pd.DataFrame) -> pd.Series:
    if "mac_retry_flag" in df.columns:
        return pd.Series(df["mac_retry_flag"], index=df.index).fillna(False).astype(bool)
    if "retry_flag" not in df.columns:
        return pd.Series(False, index=df.index)

    retry = df["retry_flag"]
    if pd.api.types.is_bool_dtype(retry):
        return retry.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(retry):
        return pd.to_numeric(retry, errors="coerce").fillna(0).ne(0)

    text = retry.astype("string").str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"])


def packet_rate_stats(df: pd.DataFrame, bucket: str = "1s") -> Tuple[float, float]:
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return float("nan"), float("nan")

    tmp = df[df["timestamp"].notna()].copy()
    if tmp.empty:
        return float("nan"), float("nan")

    tmp["time_bucket"] = tmp["timestamp"].dt.floor(bucket)
    counts = tmp.groupby("time_bucket")["frame_num"].count().sort_index()
    if counts.empty:
        return float("nan"), float("nan")

    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq=bucket)
    counts = counts.reindex(full_index, fill_value=0)
    packet_rate = counts.astype(float) / bucket_seconds(bucket)
    return float(packet_rate.var(ddof=0)), float(packet_rate.std(ddof=0))


def max_inter_arrival_gap_s(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return float("nan")

    ts = df["timestamp"].dropna().sort_values()
    if len(ts) < 2:
        return float("nan")
    return float(ts.diff().dt.total_seconds().max())


def jitter_summary_stats(df: pd.DataFrame, bucket: str = "1s") -> Dict[str, float]:
    try:
        jitter = detect_delay_jitter(df, bucket=bucket)
    except Exception:
        return {
            "jitter_score": float("nan"),
            "peak_jitter_score": float("nan"),
            "peak_std_iat": float("nan"),
            "peak_p95_iat": float("nan"),
            "mean_std_iat": float("nan"),
        }

    return {
        "jitter_score": _safe_max(jitter["jitter_score"]) if "jitter_score" in jitter.columns else float("nan"),
        "peak_jitter_score": _safe_max(jitter["jitter_score"]) if "jitter_score" in jitter.columns else float("nan"),
        "peak_std_iat": _safe_max(jitter["std_iat"]) if "std_iat" in jitter.columns else float("nan"),
        "peak_p95_iat": _safe_max(jitter["p95_iat"]) if "p95_iat" in jitter.columns else float("nan"),
        "mean_std_iat": _safe_mean(jitter["std_iat"]) if "std_iat" in jitter.columns else float("nan"),
    }


def choose_comparison_metric(plan: Dict[str, Any], summary: Dict[str, Any]) -> str:
    requested = plan.get("comparison_metric")
    if requested in summary and pd.notna(summary[requested]):
        return requested

    fallbacks = [
        infer_comparison_metric(plan.get("question_rephrased", ""), plan.get("task_type")),
        "comparison_score",
        "packet_rate_variance",
        "max_iat_s",
        "retry_rate",
        "avg_packet_size",
        "packets_per_second",
        "bytes_per_second",
        "bad_fcs_rate",
        "jitter_score",
        "peak_burst_z",
        "peak_error_z",
        "packet_count",
        "bytes_sum",
    ]
    for metric in fallbacks:
        if metric in summary and pd.notna(summary[metric]):
            return metric
    return "comparison_score"


def summarize_session_result(session_id: str, session_df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    filtered_df = apply_filters(session_df, plan["filters"])
    filtered_df = ensure_metric_columns(filtered_df)
    result, meta = run_plan(session_df, plan)
    task_type = plan["task_type"]

    ts = filtered_df["timestamp"] if "timestamp" in filtered_df.columns else pd.Series(dtype="datetime64[ns]")
    session_duration_s = (
        float((ts.max() - ts.min()).total_seconds())
        if not ts.empty and ts.notna().any()
        else float("nan")
    )
    packet_count = int(filtered_df["frame_num"].count()) if "frame_num" in filtered_df.columns else int(meta["rows_after_filter"])
    bytes_sum = _safe_sum(filtered_df["frame_size_bytes"]) if "frame_size_bytes" in filtered_df.columns else float("nan")
    avg_packet_size = _divide_or_nan(bytes_sum, packet_count)
    packets_per_second = _divide_or_nan(packet_count, session_duration_s)
    bytes_per_second = _divide_or_nan(bytes_sum, session_duration_s)
    packet_rate_variance, packet_rate_std = packet_rate_stats(filtered_df, bucket=plan.get("time_bucket", "1s"))
    max_iat_s = max_inter_arrival_gap_s(filtered_df)
    jitter_stats = jitter_summary_stats(filtered_df, bucket=plan.get("time_bucket", "1s"))
    bad_fcs_rate = (
        float(filtered_df["bad_fcs_flag"].fillna(False).mean())
        if "bad_fcs_flag" in filtered_df.columns and len(filtered_df)
        else float("nan")
    )
    mac_rows = int(filtered_df["mac_row_flag"].fillna(False).sum()) if "mac_row_flag" in filtered_df.columns else 0
    retry_count = int(filtered_df["mac_retry_flag"].fillna(False).sum()) if "mac_retry_flag" in filtered_df.columns else 0
    retry_rate = _divide_or_nan(retry_count, mac_rows)

    summary: Dict[str, Any] = {
        "session_id": session_id,
        "task_type": task_type,
        "rows_after_filter": meta["rows_after_filter"],
        "result_rows": meta["result_rows"],
        "session_duration_s": session_duration_s,
        "packet_count": packet_count,
        "bytes_sum": bytes_sum,
        "avg_packet_size": avg_packet_size,
        "packets_per_second": packets_per_second,
        "bytes_per_second": bytes_per_second,
        "packet_rate_variance": packet_rate_variance,
        "packet_rate_std": packet_rate_std,
        "max_iat_s": max_iat_s,
        **jitter_stats,
        "bad_fcs_rate": bad_fcs_rate,
        "mac_rows": mac_rows,
        "retry_count": retry_count,
        "retry_rate": retry_rate,
    }

    if task_type == "burst_detection":
        summary.update({
            "peak_burst_z": _safe_max(result["burst_z"]) if "burst_z" in result.columns else float("nan"),
            "peak_packets_per_s": _safe_max(result["packets_per_s"]) if "packets_per_s" in result.columns else float("nan"),
            "burst_bucket_count": int(result["is_burst"].fillna(False).sum()) if "is_burst" in result.columns else 0,
            "burst_bucket_fraction": float(result["is_burst"].fillna(False).mean()) if "is_burst" in result.columns and len(result) else 0.0,
            "mean_bad_fcs_rate": _safe_mean(result["bad_fcs_rate"]) if "bad_fcs_rate" in result.columns else float("nan"),
            "mean_retry_rate": _safe_mean(result["retry_rate"]) if "retry_rate" in result.columns else float("nan"),
        })
        summary["peak_error_z"] = float("nan")
    elif task_type == "delay_jitter_analysis":
        summary.update({
            "peak_jitter_score": _safe_max(result["jitter_score"]) if "jitter_score" in result.columns else float("nan"),
            "peak_std_iat": _safe_max(result["std_iat"]) if "std_iat" in result.columns else float("nan"),
            "peak_p95_iat": _safe_max(result["p95_iat"]) if "p95_iat" in result.columns else float("nan"),
            "peak_mean_iat": _safe_max(result["mean_iat"]) if "mean_iat" in result.columns else float("nan"),
            "mean_bad_fcs_rate": _safe_mean(result["bad_fcs_rate"]) if "bad_fcs_rate" in result.columns else float("nan"),
            "mean_retry_rate": _safe_mean(result["retry_rate"]) if "retry_rate" in result.columns else float("nan"),
        })
        summary["jitter_score"] = summary["peak_jitter_score"]
        summary["peak_burst_z"] = float("nan")
        summary["peak_error_z"] = float("nan")
    elif task_type == "error_spike_analysis":
        summary.update({
            "peak_error_z": _safe_max(result["error_z"]) if "error_z" in result.columns else float("nan"),
            "peak_error_frame_rate": _safe_max(result["error_frame_rate"]) if "error_frame_rate" in result.columns else float("nan"),
            "peak_bad_fcs_rate": _safe_max(result["bad_fcs_rate"]) if "bad_fcs_rate" in result.columns else float("nan"),
            "peak_retry_rate": _safe_max(result["retry_rate"]) if "retry_rate" in result.columns else float("nan"),
            "error_spike_bucket_count": int(result["is_error_spike"].fillna(False).sum()) if "is_error_spike" in result.columns else 0,
        })
        summary["jitter_score"] = float("nan")
        summary["peak_burst_z"] = float("nan")
    else:
        if "packet_count" in result.columns:
            summary["packet_count"] = _safe_sum(result["packet_count"])
            summary["avg_packet_size"] = _divide_or_nan(summary["bytes_sum"], summary["packet_count"])
            summary["packets_per_second"] = _divide_or_nan(summary["packet_count"], session_duration_s)
        if "bytes_sum" in result.columns:
            summary["bytes_sum"] = _safe_sum(result["bytes_sum"])
            summary["avg_packet_size"] = _divide_or_nan(summary["bytes_sum"], summary["packet_count"])
            summary["bytes_per_second"] = _divide_or_nan(summary["bytes_sum"], session_duration_s)

    primary_metric = choose_comparison_metric(plan, summary)
    summary["primary_metric"] = primary_metric
    summary["comparison_score"] = summary.get(primary_metric, float("nan"))

    return summary


def validate_retry_rate_ground_truth(result: pd.DataFrame) -> Dict[str, Any]:
    expected_zero_sessions = {
        "02_03_1",
        "02_04_1",
        "02_04_2",
        "02_05_1",
        "02_05_2",
        "02_06_1",
        "02_06_2",
    }
    expected_nonzero_session = "02_03_2"
    expected_nonzero_rate = 0.1143986910

    if not {"session_id", "retry_rate", "is_best_tie"}.issubset(result.columns):
        return {"checked": False, "reason": "retry-rate comparison columns are missing"}

    rates = pd.to_numeric(result["retry_rate"], errors="coerce")
    lowest = float(rates.min()) if rates.notna().any() else float("nan")
    zero_sessions = set(result.loc[np.isclose(rates, 0.0, rtol=1e-9, atol=1e-12), "session_id"].astype(str))
    tied_sessions = set(result.loc[result["is_best_tie"].fillna(False), "session_id"].astype(str))
    nonzero_sessions = set(result.loc[rates.fillna(0).ne(0), "session_id"].astype(str))

    nonzero_rate = float("nan")
    match = result["session_id"].astype(str).eq(expected_nonzero_session)
    if match.any():
        nonzero_rate = float(pd.to_numeric(result.loc[match, "retry_rate"], errors="coerce").iloc[0])

    checks = {
        "lowest_retry_rate_is_zero": bool(math.isfinite(lowest) and np.isclose(lowest, 0.0, rtol=1e-9, atol=1e-12)),
        "all_zero_sessions_marked_tied": expected_zero_sessions.issubset(tied_sessions),
        "zero_sessions_match_expected": zero_sessions == expected_zero_sessions,
        "only_02_03_2_nonzero": nonzero_sessions == {expected_nonzero_session},
        "02_03_2_rate_around_0_1144": bool(
            math.isfinite(nonzero_rate)
            and np.isclose(nonzero_rate, expected_nonzero_rate, rtol=1e-4, atol=1e-6)
        ),
    }
    return {
        "checked": True,
        "passed": all(checks.values()),
        "checks": checks,
        "lowest_retry_rate": lowest,
        "zero_retry_sessions": sorted(zero_sessions),
        "best_tied_sessions": sorted(tied_sessions),
        "nonzero_retry_sessions": sorted(nonzero_sessions),
        "02_03_2_retry_rate": nonzero_rate,
    }


def compare_sessions(sessions: Dict[str, pd.DataFrame], plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = [summarize_session_result(session_id, df, plan) for session_id, df in sessions.items()]
    out = pd.DataFrame(rows)

    mode = plan.get("comparison_mode", "rank")
    primary_metric = (
        plan.get("comparison_metric")
        or (out["primary_metric"].dropna().iloc[0] if "primary_metric" in out.columns and out["primary_metric"].notna().any() else None)
    )
    ascending = comparison_ascending(plan.get("question_rephrased", ""), primary_metric, mode)

    tie_breakers = [c for c in [
        "comparison_score",
        "packet_rate_variance",
        "max_iat_s",
        "retry_rate",
        "burst_bucket_fraction",
        "burst_bucket_count",
        "peak_packets_per_s",
        "peak_p95_iat",
        "peak_error_frame_rate",
        "rows_after_filter",
    ] if c in out.columns]
    if not tie_breakers:
        raise ValueError("No comparison columns available for cross-session ranking.")

    sort_ascending = [ascending] + [ascending] * (len(tie_breakers) - 1)
    out = out.sort_values(tie_breakers, ascending=sort_ascending, na_position="last").reset_index(drop=True)

    best_score = pd.to_numeric(out["comparison_score"], errors="coerce").dropna()
    if not best_score.empty:
        target = float(best_score.iloc[0])
        scores = pd.to_numeric(out["comparison_score"], errors="coerce")
        out["is_best_tie"] = np.isclose(scores, target, rtol=1e-9, atol=1e-12)
    else:
        out["is_best_tie"] = False

    rank_col = "best_rank" if ascending else "strongest_rank"
    out.insert(0, rank_col, range(1, len(out) + 1))

    preferred_order = [
        rank_col,
        "session_id",
        "packet_count",
        "packet_rate_variance",
        "packet_rate_std",
        "max_iat_s",
        "jitter_score",
        "peak_jitter_score",
        "peak_std_iat",
        "peak_p95_iat",
        "mean_std_iat",
        "mac_rows",
        "retry_count",
        "retry_rate",
        "bytes_sum",
        "avg_packet_size",
        "packets_per_second",
        "bytes_per_second",
        "bad_fcs_rate",
        "comparison_score",
        "primary_metric",
        "is_best_tie",
    ]
    ordered_cols = [c for c in preferred_order if c in out.columns]
    remaining_cols = [c for c in out.columns if c not in ordered_cols]
    out = out[ordered_cols + remaining_cols]

    best_tied_sessions = (
        out.loc[out["is_best_tie"].fillna(False), "session_id"].astype(str).tolist()
        if "is_best_tie" in out.columns and "session_id" in out.columns
        else []
    )
    meta = {
        "session_scope": "all_sessions",
        "comparison_mode": mode,
        "comparison_ascending": bool(ascending),
        "sessions_compared": int(len(out)),
        "comparison_metric": out["primary_metric"].iloc[0] if "primary_metric" in out.columns and not out.empty else None,
        "best_tied_sessions": best_tied_sessions,
    }
    if meta["comparison_metric"] == "retry_rate":
        meta["retry_rate_validation"] = validate_retry_rate_ground_truth(out)
    return out, meta


# ============================================================
# Plotting
# ============================================================

def save_plot(result: pd.DataFrame, plan: Dict[str, Any], out_prefix: Path) -> Optional[Path]:
    plot_kind = plan["plot"]
    if plot_kind == "none" or result.empty:
        return None

    plt.figure(figsize=(10, 5))

    if plot_kind == "line":
        x_col = "time_bucket" if "time_bucket" in result.columns else result.columns[0]
        if plan["task_type"] == "anomaly_detection":
            preferred = ["num_detectors_flagged", "packet_count", "packets_per_s", "jitter_score", "bad_fcs_rate", "retry_rate"]
        elif plan["task_type"] == "error_spike_analysis":
            preferred = ["error_frame_count", "bad_fcs_count", "error_frame_rate", "bad_fcs_rate", "retry_rate"]
        elif plan["task_type"] == "delay_jitter_analysis":
            preferred = ["std_iat", "jitter_score", "p95_iat", "mean_iat", "mean_duration_us"]
        else:
            preferred = ["packet_count", "packets_per_s", "bytes_sum", "data_frame_count"]
        y_candidates = [c for c in preferred if c in result.columns]
        if not y_candidates:
            return None
        y_col = y_candidates[0]
        plt.plot(result[x_col], result[y_col], marker="o")
        plt.xticks(rotation=30, ha="right")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(plan["question_rephrased"])

    elif plot_kind == "bar":
        x_col = "session_id" if "session_id" in result.columns else result.columns[0]
        preferred = [
            plan.get("comparison_metric"),
            "comparison_score", "avg_packet_size", "packets_per_second", "bytes_per_second",
            "packet_count", "bytes_sum", "error_frame_count", "bad_fcs_count",
            "retry_count", "bad_fcs_rate", "error_frame_rate", "retry_rate"
        ]
        y_candidates = [c for c in preferred if c and c in result.columns]
        if not y_candidates:
            return None
        y_col = y_candidates[0]
        plt.bar(result[x_col].astype(str), result[y_col])
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(plan["question_rephrased"])

    else:
        return None

    plt.tight_layout()
    out_path = out_prefix.with_suffix(".png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


# ============================================================
# Narrative summary
# ============================================================

def df_preview_for_llm(df: pd.DataFrame, max_rows: int = 20) -> List[Dict[str, Any]]:
    preview = df.head(max_rows).copy()
    for c in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[c]):
            preview[c] = preview[c].astype(str)
    return preview.to_dict(orient="records")


def summarize_with_llm(
    question: str,
    plan: Dict[str, Any],
    result: pd.DataFrame,
    meta: Dict[str, Any],
) -> str:
    preview_df = result
    if plan.get("task_type") == "burst_detection" and {"is_burst", "burst_z"}.issubset(result.columns):
        preview_df = result.sort_values(["is_burst", "burst_z"], ascending=[False, False])
    elif plan.get("task_type") == "error_spike_analysis" and {"is_error_spike", "error_z"}.issubset(result.columns):
        preview_df = result.sort_values(["is_error_spike", "error_z"], ascending=[False, False])
    elif plan.get("task_type") == "delay_jitter_analysis" and "std_iat" in result.columns:
        preview_df = result.sort_values("std_iat", ascending=False, na_position="last")

    prompt = f"""
You are writing a short network-traffic analysis summary for a class project.

User question:
{question}

Executed plan:
{json.dumps(plan, indent=2)}

Execution metadata:
{json.dumps(meta, indent=2)}

Top result rows:
{json.dumps(df_preview_for_llm(preview_df, max_rows=20), indent=2)}

Write:
1. A short answer to the question
2. A brief interpretation of whether there is bursty traffic, high delay/jitter, or other abnormal patterns
3. If anomalies appear, add a compact incident-style explanation with likely causes
4. Do not invent fields that are not present
5. Keep it under 250 words
6. If this is a cross-session comparison, explicitly name the top-ranked session and mention the comparison_metric / primary_metric basis from the result rows. Prefer the named metric column, such as peak_burst_z, over the generic comparison_score alias.
7. If this is anomaly_detection, treat any_anomaly and detector flags as already-computed local labels; do not relabel buckets
8. If result rows include is_best_tie=True for multiple sessions, mention all tied sessions, especially for lowest retry rate
9. If burst_detection rows include dominant_* fields, use them to identify dominant channels/devices; source_addr comes from Transmit Addr and destination_addr comes from Receive Addr.
10. If this is error_spike_analysis, base the answer on the highest error_z / is_error_spike rows first, then discuss whether their mean_signal_dbm or mean_rate_mbps values support the user's association question.
11. If this is delay_jitter_analysis, use std_iat as the primary "jitter over time" metric because the UI chart plots std_iat; use jitter_score and p95_iat only as supporting context.
""".strip()

    resp = get_openai_client().responses.create(
        model=MODEL_SUMMARY,
        input=prompt,
    )
    return resp.output_text.strip()


def summarize_locally(question: str, plan: Dict[str, Any], result: pd.DataFrame, meta: Dict[str, Any]) -> str:
    if plan.get("task_type") == "anomaly_detection":
        anomaly_count = int(result["any_anomaly"].fillna(False).sum()) if "any_anomaly" in result.columns else 0
        total = int(len(result))
        validation = meta.get("validation") or {}
        lines = [f"Local anomaly detectors flagged {anomaly_count} of {total} time buckets."]
        if validation:
            lines.append(
                "Validation: "
                f"TP={validation.get('true_positives')}, "
                f"FP={validation.get('false_positives')}, "
                f"FN={validation.get('false_negatives')}, "
                f"precision={_format_metric_value(validation.get('precision'))}, "
                f"recall={_format_metric_value(validation.get('recall'))}, "
                f"F1={_format_metric_value(validation.get('f1'))}."
            )
        if anomaly_count:
            top = result[result["any_anomaly"].fillna(False)].head(3)
            examples = [
                f"{row.get('time_bucket')}: {row.get('anomaly_reason')}"
                for _, row in top.iterrows()
            ]
            lines.append("First flagged buckets: " + " | ".join(examples))
        return " ".join(lines)

    if meta.get("session_scope") == "all_sessions" and not result.empty:
        metric = meta.get("comparison_metric") or (result["primary_metric"].iloc[0] if "primary_metric" in result.columns else "comparison_score")
        tied = meta.get("best_tied_sessions") or []
        direction = "lowest" if meta.get("comparison_ascending") else "highest"
        if metric == "retry_rate" and direction == "lowest" and len(tied) > 1:
            return (
                f"Multiple sessions tie for lowest retry rate: {', '.join(tied)}. "
                f"Lowest retry_rate={_format_metric_value(result['comparison_score'].iloc[0])}."
            )
        if tied:
            return (
                f"Best session(s) by {direction} {metric}: {', '.join(tied)}. "
                f"Top comparison_score={_format_metric_value(result['comparison_score'].iloc[0])}."
            )
        return f"Top session by {direction} {metric}: {result['session_id'].iloc[0]}."

    return f"Completed local execution for: {question}. Result rows: {len(result)}."


# ============================================================
# Main app
# ============================================================

def build_all_sessions(root: Path) -> Dict[str, pd.DataFrame]:
    session_files = discover_sessions(root)
    if not session_files:
        raise FileNotFoundError(f"No session folders found under {root}")

    sessions = {}
    for sid, files in session_files.items():
        try:
            sessions[sid] = merge_session_tables(files)
            print(f"[OK] Loaded {sid}: {len(sessions[sid])} merged rows")
        except Exception as e:
            print(f"[WARN] Failed to load {sid}: {e}")
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Wireless trace LLM pipeline with local pandas execution")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Root dataset directory")
    parser.add_argument("--question", type=str, required=True, help="Natural-language analysis question")
    parser.add_argument("--session", type=str, default=None, help="Optional session override, e.g. 02_04_1")
    parser.add_argument("--save-csv", action="store_true", help="Save result CSV")
    parser.add_argument("--known-anomaly-start", type=str, default=None, help="Known anomaly window start timestamp")
    parser.add_argument("--known-anomaly-end", type=str, default=None, help="Known anomaly window end timestamp")
    args = parser.parse_args()

    if bool(args.known_anomaly_start) != bool(args.known_anomaly_end):
        raise ValueError("Provide both --known-anomaly-start and --known-anomaly-end, or neither.")

    root = Path(args.root)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = build_all_sessions(root)
    if not sessions:
        raise RuntimeError("No sessions were successfully loaded.")

    session_summaries = {sid: summarize_dataframe_schema(df) for sid, df in sessions.items()}

    if should_run_local_anomaly_detection(args.question, args.known_anomaly_start, args.known_anomaly_end):
        plan = build_local_anomaly_plan(args.question, list(session_summaries.keys()))
    elif should_run_local_burst_dominance(args.question):
        plan = build_local_burst_dominance_plan(args.question, list(session_summaries.keys()))
    else:
        plan = build_local_comparison_plan(args.question, list(session_summaries.keys()))
        if plan is None:
            plan = ask_llm_for_plan(args.question, session_summaries)

    if args.session:
        plan["session_id"] = args.session
        plan["session_scope"] = "single_session"
        plan["comparison_mode"] = "none"

    if plan.get("session_scope") == "all_sessions" and not args.session:
        plan["session_id"] = "ALL_SESSIONS"
        result, meta = compare_sessions(sessions, plan)
        stem_base = "all_sessions"
    else:
        session_id = plan["session_id"]
        if session_id not in sessions:
            raise ValueError(f"LLM selected unknown session_id={session_id}. Available: {sorted(sessions.keys())}")
        result, meta = run_plan(
            sessions[session_id],
            plan,
            known_anomaly_start=args.known_anomaly_start,
            known_anomaly_end=args.known_anomaly_end,
        )
        stem_base = session_id

    stem = re.sub(r"[^\w\-]+", "_", f"{stem_base}_{plan['task_type']}")
    out_prefix = out_dir / stem

    plot_path = save_plot(result, plan, out_prefix)

    if args.save_csv:
        csv_path = out_prefix.with_suffix(".csv")
        result.to_csv(csv_path, index=False)
        print(f"[OK] Saved result CSV to {csv_path}")

    force_local_summary = (
        meta.get("comparison_metric") == "retry_rate"
        and meta.get("comparison_ascending")
        and len(meta.get("best_tied_sessions") or []) > 1
    )
    try:
        if force_local_summary:
            summary = summarize_locally(args.question, plan, result, meta)
        else:
            summary = summarize_with_llm(args.question, plan, result, meta)
    except Exception as e:
        print(f"[WARN] LLM summary failed; using local summary instead: {e}")
        summary = summarize_locally(args.question, plan, result, meta)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(args.question)

    print("\n" + "=" * 80)
    print("PLAN")
    print("=" * 80)
    print(json.dumps(plan, indent=2))

    print("\n" + "=" * 80)
    print("RESULT PREVIEW")
    print("=" * 80)
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(result.head(20))

    if meta.get("validation"):
        print("\n" + "=" * 80)
        print("VALIDATION")
        print("=" * 80)
        print(json.dumps(meta["validation"], indent=2))

    if meta.get("retry_rate_validation"):
        print("\n" + "=" * 80)
        print("RETRY RATE VALIDATION")
        print("=" * 80)
        print(json.dumps(meta["retry_rate_validation"], indent=2))

    if plot_path:
        print(f"\n[OK] Plot saved to: {plot_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary)


if __name__ == "__main__":
    main()
