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
    coalesce_to_column(merged, "retry_flag", ["mac_retry", "error_retry"])

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

    if "retry_flag" in merged.columns:
        merged["retry_flag"] = pd.to_numeric(merged["retry_flag"], errors="coerce").fillna(0).astype(int).astype(bool)
    else:
        merged["retry_flag"] = False

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

    if any(word in q for word in ["strongest", "highest", "most", "largest", "worst", "max"]):
        return "strongest"
    if any(word in q for word in ["weakest", "lowest", "least", "smallest", "best", "min"]):
        return "weakest"
    return "rank"


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
        plan["plot"] = "bar"
    else:
        plan["session_scope"] = "single_session"
        plan["comparison_mode"] = "none"
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
    out = df.copy()

    if "bad_fcs_flag" not in out.columns:
        out["bad_fcs_flag"] = False
    if "has_error_row" not in out.columns:
        out["has_error_row"] = False
    if "retry_flag" not in out.columns:
        out["retry_flag"] = False
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
    if "retry_count" in metrics:
        agg_map["retry_count"] = ("retry_flag", "sum")
    if "retry_rate" in metrics:
        agg_map["retry_rate"] = ("retry_flag", "mean")
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
        if "retry_count" in metrics:
            row["retry_count"] = int(pd.Series(df["retry_flag"]).fillna(False).sum())
        if "retry_rate" in metrics:
            row["retry_rate"] = float(pd.Series(df["retry_flag"]).fillna(False).mean())
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


def detect_bursts(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    b = bucket_time(ensure_metric_columns(df), bucket)
    g = b.groupby("time_bucket", dropna=False).agg(
        packet_count=("frame_num", "count"),
        data_frame_count=("data_frame_flag", "sum"),
        bytes_sum=("frame_size_bytes", "sum"),
        bad_fcs_count=("bad_fcs_flag", lambda s: pd.Series(s).fillna(False).sum()),
        error_frame_count=("has_error_row", lambda s: pd.Series(s).fillna(False).sum()),
        retry_count=("retry_flag", lambda s: pd.Series(s).fillna(False).sum()),
        mean_signal_dbm=("signal_dbm", "mean"),
        mean_rate_mbps=("data_rate_mbps", "mean"),
    ).reset_index()

    g["packets_per_s"] = g["packet_count"] / bucket_seconds(bucket)
    g["bad_fcs_rate"] = g["bad_fcs_count"] / g["packet_count"].replace(0, np.nan)
    g["error_frame_rate"] = g["error_frame_count"] / g["packet_count"].replace(0, np.nan)
    g["retry_rate"] = g["retry_count"] / g["packet_count"].replace(0, np.nan)

    # z-score burst indicator
    count_mean = g["packet_count"].mean()
    count_std = g["packet_count"].std(ddof=0)
    if pd.notna(count_std) and count_std > 0:
        g["burst_z"] = (g["packet_count"] - count_mean) / count_std
    else:
        g["burst_z"] = 0.0

    g["is_burst"] = g["burst_z"] >= 2.0
    return g


def detect_delay_jitter(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    b = bucket_time(ensure_metric_columns(df), bucket)
    g = b.groupby("time_bucket", dropna=False).agg(
        packet_count=("frame_num", "count"),
        data_frame_count=("data_frame_flag", "sum"),
        mean_iat=("analysis_iat_s", "mean"),
        std_iat=("analysis_iat_s", "std"),
        p95_iat=("analysis_iat_s", lambda s: np.nanpercentile(s.dropna(), 95) if s.notna().any() else np.nan),
        mean_global_iat=("iat_s", "mean"),
        mean_flow_iat=("flow_iat_s", "mean") if "flow_iat_s" in b.columns else ("analysis_iat_s", "mean"),
        mean_duration_us=("tap_duration_us", "mean") if "tap_duration_us" in b.columns else ("duration_us", "mean"),
        retry_count=("retry_flag", lambda s: pd.Series(s).fillna(False).sum()),
        bad_fcs_count=("bad_fcs_flag", lambda s: pd.Series(s).fillna(False).sum()),
        error_frame_count=("has_error_row", lambda s: pd.Series(s).fillna(False).sum()),
        mean_signal_dbm=("signal_dbm", "mean"),
        mean_rate_mbps=("data_rate_mbps", "mean"),
    ).reset_index()

    # simple jitter score: normalized std_iat
    eps = 1e-9
    g["jitter_score"] = g["std_iat"] / (g["mean_iat"] + eps)
    g["retry_rate"] = g["retry_count"] / g["packet_count"].replace(0, np.nan)
    g["bad_fcs_rate"] = g["bad_fcs_count"] / g["packet_count"].replace(0, np.nan)
    g["error_frame_rate"] = g["error_frame_count"] / g["packet_count"].replace(0, np.nan)
    return g


def detect_error_spikes(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    b = bucket_time(ensure_metric_columns(df), bucket)
    g = b.groupby("time_bucket", dropna=False).agg(
        packet_count=("frame_num", "count"),
        data_frame_count=("data_frame_flag", "sum"),
        bad_fcs_count=("bad_fcs_flag", lambda s: pd.Series(s).fillna(False).sum()),
        error_frame_count=("has_error_row", lambda s: pd.Series(s).fillna(False).sum()),
        retry_count=("retry_flag", lambda s: pd.Series(s).fillna(False).sum()),
        mean_signal_dbm=("signal_dbm", "mean"),
        mean_rate_mbps=("data_rate_mbps", "mean"),
    ).reset_index()
    g["bad_fcs_rate"] = g["bad_fcs_count"] / g["packet_count"].replace(0, np.nan)
    g["error_frame_rate"] = g["error_frame_count"] / g["packet_count"].replace(0, np.nan)
    g["retry_rate"] = g["retry_count"] / g["packet_count"].replace(0, np.nan)

    count_mean = g["error_frame_count"].mean()
    count_std = g["error_frame_count"].std(ddof=0)
    if pd.notna(count_std) and count_std > 0:
        g["error_z"] = (g["error_frame_count"] - count_mean) / count_std
    else:
        g["error_z"] = 0.0
    g["is_error_spike"] = (g["error_z"] >= 2.0) | (g["bad_fcs_rate"] >= 0.2)
    return g


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


def run_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
    return result, meta


def _safe_max(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    return float(series.max()) if series.notna().any() else float("nan")


def _safe_mean(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce")
    return float(series.mean()) if series.notna().any() else float("nan")


def summarize_session_result(session_id: str, session_df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
    result, meta = run_plan(session_df, plan)
    task_type = plan["task_type"]

    ts = session_df["timestamp"] if "timestamp" in session_df.columns else pd.Series(dtype="datetime64[ns]")
    session_duration_s = (
        float((ts.max() - ts.min()).total_seconds())
        if not ts.empty and ts.notna().any()
        else float("nan")
    )

    summary: Dict[str, Any] = {
        "session_id": session_id,
        "task_type": task_type,
        "rows_after_filter": meta["rows_after_filter"],
        "result_rows": meta["result_rows"],
        "session_duration_s": session_duration_s,
    }

    if task_type == "burst_detection":
        summary.update({
            "comparison_score": _safe_max(result["burst_z"]) if "burst_z" in result.columns else float("nan"),
            "peak_burst_z": _safe_max(result["burst_z"]) if "burst_z" in result.columns else float("nan"),
            "peak_packets_per_s": _safe_max(result["packets_per_s"]) if "packets_per_s" in result.columns else float("nan"),
            "burst_bucket_count": int(result["is_burst"].fillna(False).sum()) if "is_burst" in result.columns else 0,
            "burst_bucket_fraction": float(result["is_burst"].fillna(False).mean()) if "is_burst" in result.columns and len(result) else 0.0,
            "mean_bad_fcs_rate": _safe_mean(result["bad_fcs_rate"]) if "bad_fcs_rate" in result.columns else float("nan"),
            "mean_retry_rate": _safe_mean(result["retry_rate"]) if "retry_rate" in result.columns else float("nan"),
        })
    elif task_type == "delay_jitter_analysis":
        summary.update({
            "comparison_score": _safe_max(result["jitter_score"]) if "jitter_score" in result.columns else float("nan"),
            "peak_jitter_score": _safe_max(result["jitter_score"]) if "jitter_score" in result.columns else float("nan"),
            "peak_p95_iat": _safe_max(result["p95_iat"]) if "p95_iat" in result.columns else float("nan"),
            "peak_mean_iat": _safe_max(result["mean_iat"]) if "mean_iat" in result.columns else float("nan"),
            "mean_bad_fcs_rate": _safe_mean(result["bad_fcs_rate"]) if "bad_fcs_rate" in result.columns else float("nan"),
            "mean_retry_rate": _safe_mean(result["retry_rate"]) if "retry_rate" in result.columns else float("nan"),
        })
    elif task_type == "error_spike_analysis":
        summary.update({
            "comparison_score": _safe_max(result["error_z"]) if "error_z" in result.columns else float("nan"),
            "peak_error_z": _safe_max(result["error_z"]) if "error_z" in result.columns else float("nan"),
            "peak_error_frame_rate": _safe_max(result["error_frame_rate"]) if "error_frame_rate" in result.columns else float("nan"),
            "peak_bad_fcs_rate": _safe_max(result["bad_fcs_rate"]) if "bad_fcs_rate" in result.columns else float("nan"),
            "peak_retry_rate": _safe_max(result["retry_rate"]) if "retry_rate" in result.columns else float("nan"),
            "error_spike_bucket_count": int(result["is_error_spike"].fillna(False).sum()) if "is_error_spike" in result.columns else 0,
        })
    else:
        sort_col = "packet_count" if "packet_count" in result.columns else result.columns[-1]
        summary.update({
            "comparison_score": _safe_max(result[sort_col]),
            "primary_metric": sort_col,
        })

    return summary


def compare_sessions(sessions: Dict[str, pd.DataFrame], plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = [summarize_session_result(session_id, df, plan) for session_id, df in sessions.items()]
    out = pd.DataFrame(rows)

    mode = plan.get("comparison_mode", "rank")
    ascending = mode == "weakest"

    tie_breakers = [c for c in [
        "comparison_score",
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

    rank_col = "weakest_rank" if ascending else "strongest_rank"
    out.insert(0, rank_col, range(1, len(out) + 1))

    meta = {
        "session_scope": "all_sessions",
        "comparison_mode": mode,
        "sessions_compared": int(len(out)),
    }
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
        if plan["task_type"] == "error_spike_analysis":
            preferred = ["error_frame_count", "bad_fcs_count", "error_frame_rate", "bad_fcs_rate", "retry_rate"]
        elif plan["task_type"] == "delay_jitter_analysis":
            preferred = ["p95_iat", "std_iat", "jitter_score", "mean_iat", "mean_duration_us"]
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
        y_candidates = [c for c in [
            "comparison_score", "packet_count", "bytes_sum", "error_frame_count", "bad_fcs_count",
            "retry_count", "bad_fcs_rate", "error_frame_rate", "retry_rate"
        ] if c in result.columns]
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
    prompt = f"""
You are writing a short network-traffic analysis summary for a class project.

User question:
{question}

Executed plan:
{json.dumps(plan, indent=2)}

Execution metadata:
{json.dumps(meta, indent=2)}

Top result rows:
{json.dumps(df_preview_for_llm(result, max_rows=20), indent=2)}

Write:
1. A short answer to the question
2. A brief interpretation of whether there is bursty traffic, high delay/jitter, or other abnormal patterns
3. If anomalies appear, add a compact incident-style explanation with likely causes
4. Do not invent fields that are not present
5. Keep it under 250 words
6. If this is a cross-session comparison, explicitly name the top-ranked session and mention the comparison_score basis from the result rows
""".strip()

    resp = get_openai_client().responses.create(
        model=MODEL_SUMMARY,
        input=prompt,
    )
    return resp.output_text.strip()


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
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = build_all_sessions(root)
    if not sessions:
        raise RuntimeError("No sessions were successfully loaded.")

    session_summaries = {sid: summarize_dataframe_schema(df) for sid, df in sessions.items()}

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
        result, meta = run_plan(sessions[session_id], plan)
        stem_base = session_id

    stem = re.sub(r"[^\w\-]+", "_", f"{stem_base}_{plan['task_type']}")
    out_prefix = out_dir / stem

    plot_path = save_plot(result, plan, out_prefix)

    if args.save_csv:
        csv_path = out_prefix.with_suffix(".csv")
        result.to_csv(csv_path, index=False)
        print(f"[OK] Saved result CSV to {csv_path}")

    summary = summarize_with_llm(args.question, plan, result, meta)

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

    if plot_path:
        print(f"\n[OK] Plot saved to: {plot_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary)


if __name__ == "__main__":
    main()
