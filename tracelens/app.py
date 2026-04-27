import os
import re
import json
import math
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "tracelens-dev-secret")
CORS(app)

# In-memory session store (maps session_token -> {session_id -> DataFrame})
_SESSION_STORE: Dict[str, Dict[str, pd.DataFrame]] = {}

# ============================================================
# Utility helpers (ported from traffic_llm_pipeline.py)
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
    s = clean_excel_style_string(s)
    vals = []
    for x in s.fillna(""):
        x = str(x).strip().lower()
        if not x:
            vals.append(np.nan); continue
        m = re.search(r"(-?\d+(?:\.\d+)?)", x)
        if not m:
            vals.append(np.nan); continue
        val = float(m.group(1))
        if "ms" in x: vals.append(val * 1000.0)
        elif "µs" in x or "us" in x: vals.append(val)
        elif re.search(r"(^|\s)s($|\s)", x): vals.append(val * 1_000_000.0)
        else: vals.append(val)
    return pd.Series(vals, index=s.index, dtype="float64")

def parse_timestamp_series(s: pd.Series) -> pd.Series:
    s = clean_excel_style_string(s)
    parsed = pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S.%f %p", errors="coerce")
    missing = parsed.isna() & s.notna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(s.loc[missing], errors="coerce")
    return parsed

MAC_RE = re.compile(r"([0-9a-f]{2}(?::[0-9a-f]{2}){5})", re.IGNORECASE)

def extract_mac(value: Any) -> Optional[str]:
    if pd.isna(value): return None
    m = MAC_RE.search(str(value))
    return m.group(1).lower() if m else None

def extract_labeled_mac(row: pd.Series, label: str) -> Optional[str]:
    wanted = f"({label.upper()})"
    for col in ["addresses", "receive_addr", "transmit_addr", "address_3"]:
        if col not in row.index or pd.isna(row[col]): continue
        if wanted in str(row[col]).upper():
            return extract_mac(row[col])
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

def read_wps_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="latin1")

def load_csv_clean(path: Path) -> pd.DataFrame:
    df = read_wps_csv(path)
    df = normalize_columns(df)
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed_")], errors="ignore")
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
    for col in ["data_rate_mb_s","frame_size","channel","freq","bookmark","ssi",
                "seq_num","tods","fromds","frag","retry","more","protected",
                "source_port","destination_port","length"]:
        if col in df.columns:
            df[col] = parse_numeric_series(df[col])
    if "antenna" in df.columns:
        df["signal_dbm"] = parse_numeric_series(df["antenna"])
    if "bad_fcs" in df.columns:
        df["bad_fcs"] = (df["bad_fcs"].astype(str).str.strip().str.lower()
                         .map({"true": True, "false": False}))
    df = derive_mac_fields(df)
    return df

# ============================================================
# Merge / session loading
# ============================================================

def prefix_nonkey_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    new_cols = {c: f"{prefix}_{c}" for c in df.columns if c != "frame_num"}
    return df.rename(columns=new_cols)

def first_available(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    available = [c for c in candidates if c in df.columns]
    if not available: return None
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
        ports = (df["udp_source_port"].astype("Int64").astype("string").fillna("?")
                 + ">" + df["udp_destination_port"].astype("Int64").astype("string").fillna("?"))
        has_ports = df["udp_source_port"].notna() | df["udp_destination_port"].notna()
        base = base.where(~has_ports, base + " udp:" + ports)
    return base.replace({"?>?": pd.NA, "?>? udp:?>?": pd.NA})

def add_unified_wireless_fields(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    coalesce_to_column(merged, "timestamp", ["tap_timestamp","mac_timestamp","udp_timestamp","error_timestamp"])
    coalesce_to_column(merged, "frame_size_bytes", ["tap_frame_size","mac_frame_size","udp_frame_size","error_frame_size"])
    coalesce_to_column(merged, "duration_us", ["tap_duration_us"])
    coalesce_to_column(merged, "signal_dbm", ["tap_signal_dbm"])
    coalesce_to_column(merged, "data_rate_mbps", ["tap_data_rate_mb_s"])
    coalesce_to_column(merged, "channel", ["tap_channel"])
    coalesce_to_column(merged, "freq_mhz", ["tap_freq"])
    coalesce_to_column(merged, "mac_type", ["mac_type","error_type"])
    coalesce_to_column(merged, "mac_subtype", ["mac_subtype","error_subtype"])
    coalesce_to_column(merged, "source_addr", ["mac_source_addr","error_source_addr"])
    coalesce_to_column(merged, "destination_addr", ["mac_destination_addr","error_destination_addr"])
    coalesce_to_column(merged, "bssid", ["mac_bssid","error_bssid"])
    coalesce_to_column(merged, "retry_flag", ["mac_retry","error_retry"])
    merged["bad_fcs_flag"] = merged.get("tap_bad_fcs", pd.Series(False, index=merged.index)).eq(True)
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
    if {"source_addr","destination_addr"}.issubset(merged.columns):
        merged["flow_key"] = build_flow_key(merged)
    merged = merged.sort_values(["timestamp","frame_num"], na_position="last").reset_index(drop=True)
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
            continue
        df["present"] = True
        tables[key] = prefix_nonkey_columns(df, key)
    merged = None
    for key in ["tap","mac","error","udp"]:
        if key not in tables: continue
        if merged is None:
            merged = tables[key]
        else:
            merged = merged.merge(tables[key], on="frame_num", how="outer")
    if merged is None:
        raise ValueError("No usable tables found for session.")
    return add_unified_wireless_fields(merged)

def discover_and_load_zip(zip_path: str) -> Dict[str, pd.DataFrame]:
    sessions = {}
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)
        root = Path(tmp)
        # find all session directories (dirs containing at least one csv)
        for session_dir in sorted(root.rglob("*")):
            if not session_dir.is_dir(): continue
            csvs = list(session_dir.glob("*.csv"))
            if not csvs: continue
            files = {}
            for f in csvs:
                name = f.name.lower()
                if "tap" in name: files["tap"] = f
                elif "mac" in name: files["mac"] = f
                elif "error" in name: files["error"] = f
                elif "udp" in name: files["udp"] = f
            if files:
                sid = session_dir.name
                try:
                    sessions[sid] = merge_session_tables(files)
                except Exception as e:
                    print(f"Skipping session {sid}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return sessions

# ============================================================
# Schema summarization
# ============================================================

def summarize_schema(df: pd.DataFrame, max_cols: int = 60) -> Dict:
    cols = []
    for c in list(df.columns)[:max_cols]:
        cols.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "nonnull": int(df[c].notna().sum())
        })
    time_min = time_max = None
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
# Analysis engine
# ============================================================

def _safe_max(s: pd.Series) -> float:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return float(s2.max()) if len(s2) else float("nan")

def _divide_or_nan(a: Any, b: Any) -> float:
    try:
        a, b = float(a), float(b)
        return a / b if b != 0 and math.isfinite(b) else float("nan")
    except:
        return float("nan")

def bucket_seconds(bucket: str) -> float:
    m = re.match(r"(\d+)s", bucket.strip().lower())
    return float(m.group(1)) if m else 1.0

def apply_filters(df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
    for f in filters:
        col, op, val = f["column"], f["op"], f["value"]
        if col not in df.columns: continue
        if op == "eq": df = df[df[col] == val]
        elif op == "ne": df = df[df[col] != val]
        elif op == "gt": df = df[pd.to_numeric(df[col], errors="coerce") > float(val)]
        elif op == "ge": df = df[pd.to_numeric(df[col], errors="coerce") >= float(val)]
        elif op == "lt": df = df[pd.to_numeric(df[col], errors="coerce") < float(val)]
        elif op == "le": df = df[pd.to_numeric(df[col], errors="coerce") <= float(val)]
        elif op == "contains": df = df[df[col].astype(str).str.lower().str.contains(str(val).lower(), na=False)]
        elif op == "in": df = df[df[col].isin(val if isinstance(val, list) else [val])]
        elif op == "notnull": df = df[df[col].notna()]
    return df

def bucket_time(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    df = df.copy()
    sec = bucket_seconds(bucket)
    freq = f"{int(sec)}s"
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df["time_bucket"] = df["timestamp"].dt.floor(freq)
    return df

METRIC_MAP = {
    "packet_count": None,
    "bytes_sum": ("frame_size_bytes", "sum"),
    "mean_iat": ("analysis_iat_s", "mean"),
    "std_iat": ("analysis_iat_s", "std"),
    "p95_iat": ("analysis_iat_s", lambda x: x.quantile(0.95)),
    "mean_duration_us": ("duration_us", "mean"),
    "bad_fcs_count": ("bad_fcs_flag", "sum"),
    "bad_fcs_rate": ("bad_fcs_flag", "mean"),
    "error_frame_count": ("has_error_row", "sum"),
    "error_frame_rate": ("has_error_row", "mean"),
    "retry_count": ("retry_flag", "sum"),
    "retry_rate": ("retry_flag", "mean"),
    "data_frame_count": ("data_frame_flag", "sum"),
    "mean_signal_dbm": ("signal_dbm", "mean"),
    "mean_rate_mbps": ("data_rate_mbps", "mean"),
}

def aggregate_metrics(df: pd.DataFrame, metrics: List[str], group_cols: List[str]) -> pd.DataFrame:
    if not metrics:
        metrics = ["packet_count"]
    agg_cols = [c for c in group_cols if c in df.columns]
    agg_specs = {}
    for m in metrics:
        if m == "packet_count":
            continue
        if m in METRIC_MAP and METRIC_MAP[m]:
            col, func = METRIC_MAP[m]
            if col in df.columns:
                agg_specs[m] = pd.NamedAgg(column=col, aggfunc=func)
    if not agg_cols:
        row = {"packet_count": len(df)}
        for m, spec in agg_specs.items():
            s = df[spec.column]
            fn = spec.aggfunc
            try:
                row[m] = float(s.agg(fn)) if callable(fn) else float(s.agg(fn))
            except:
                row[m] = float("nan")
        return pd.DataFrame([row])
    grp = df.groupby(agg_cols, dropna=False)
    result = grp.size().rename("packet_count").reset_index()
    for m, spec in agg_specs.items():
        result[m] = grp[spec.column].agg(spec.aggfunc).values
    return result

def detect_bursts(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    sec = bucket_seconds(bucket)
    bucketed = bucket_time(df, bucket)
    grp = bucketed.groupby("time_bucket", dropna=False)
    result = grp.size().rename("packet_count").reset_index()
    result["packets_per_s"] = result["packet_count"] / sec
    for col, name in [("frame_size_bytes","bytes_sum"),("bad_fcs_flag","bad_fcs_count"),
                       ("has_error_row","error_frame_count"),("retry_flag","retry_count")]:
        if col in bucketed.columns:
            result[name] = grp[col].sum().values
    for col, name in [("bad_fcs_flag","bad_fcs_rate"),("has_error_row","error_frame_rate"),("retry_flag","retry_rate")]:
        if col in bucketed.columns:
            result[name] = grp[col].mean().values
    if "signal_dbm" in bucketed.columns:
        result["mean_signal_dbm"] = grp["signal_dbm"].mean().values
    if "data_rate_mbps" in bucketed.columns:
        result["mean_rate_mbps"] = grp["data_rate_mbps"].mean().values
    counts = result["packet_count"]
    mu, sigma = counts.mean(), counts.std()
    result["burst_z"] = (counts - mu) / sigma if sigma > 0 else 0.0
    result["is_burst"] = result["burst_z"] >= 2.0
    result = result.sort_values("time_bucket").reset_index(drop=True)
    return result

def detect_delay_jitter(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    bucketed = bucket_time(df, bucket)
    grp = bucketed.groupby("time_bucket", dropna=False)
    result = grp.size().rename("packet_count").reset_index()
    if "analysis_iat_s" in bucketed.columns:
        result["mean_iat"] = grp["analysis_iat_s"].mean().values
        result["std_iat"] = grp["analysis_iat_s"].std().values
        result["p95_iat"] = grp["analysis_iat_s"].quantile(0.95).values
        result["jitter_score"] = result["std_iat"] / (result["mean_iat"] + 1e-9)
    if "iat_s" in bucketed.columns:
        result["mean_global_iat"] = grp["iat_s"].mean().values
    if "flow_iat_s" in bucketed.columns:
        result["mean_flow_iat"] = grp["flow_iat_s"].mean().values
    for col, name in [("bad_fcs_flag","bad_fcs_count"),("has_error_row","error_frame_count"),("retry_flag","retry_count")]:
        if col in bucketed.columns:
            result[name] = grp[col].sum().values
    for col, name in [("bad_fcs_flag","bad_fcs_rate"),("has_error_row","error_frame_rate"),("retry_flag","retry_rate")]:
        if col in bucketed.columns:
            result[name] = grp[col].mean().values
    if "signal_dbm" in bucketed.columns:
        result["mean_signal_dbm"] = grp["signal_dbm"].mean().values
    result = result.sort_values("time_bucket").reset_index(drop=True)
    return result

def detect_error_spikes(df: pd.DataFrame, bucket: str) -> pd.DataFrame:
    bucketed = bucket_time(df, bucket)
    grp = bucketed.groupby("time_bucket", dropna=False)
    result = grp.size().rename("packet_count").reset_index()
    for col, name in [("bad_fcs_flag","bad_fcs_count"),("has_error_row","error_frame_count"),("retry_flag","retry_count")]:
        if col in bucketed.columns:
            result[name] = grp[col].sum().values
    for col, name in [("bad_fcs_flag","bad_fcs_rate"),("has_error_row","error_frame_rate"),("retry_flag","retry_rate")]:
        if col in bucketed.columns:
            result[name] = grp[col].mean().values
    if "signal_dbm" in bucketed.columns:
        result["mean_signal_dbm"] = grp["signal_dbm"].mean().values
    err = result.get("error_frame_count", pd.Series(0, index=result.index))
    mu, sigma = err.mean(), err.std()
    result["error_z"] = (err - mu) / sigma if sigma > 0 else 0.0
    bad_fcs_rate = result.get("bad_fcs_rate", pd.Series(0.0, index=result.index))
    result["is_error_spike"] = (result["error_z"] >= 2.0) | (bad_fcs_rate >= 0.2)
    result = result.sort_values("time_bucket").reset_index(drop=True)
    return result

def top_entities(df: pd.DataFrame, group_cols: List[str], metrics: List[str], top_k: int) -> pd.DataFrame:
    if not group_cols:
        candidates = ["source_addr","destination_addr","bssid","mac_type","mac_subtype","channel"]
        found = next((c for c in candidates if c in df.columns), None)
        if found: group_cols = [found]
    result = aggregate_metrics(df, metrics or ["packet_count"], group_cols)
    sort_col = "packet_count" if "packet_count" in result.columns else result.columns[-1]
    result = result.sort_values(sort_col, ascending=False)
    return result.head(top_k)

def run_plan(df: pd.DataFrame, plan: Dict) -> Tuple[pd.DataFrame, Dict]:
    filtered = apply_filters(df, plan.get("filters", []))
    group_by = [g for g in plan.get("group_by", []) if g in filtered.columns or g == "time_bucket"]
    task = plan["task_type"]
    bucket = plan.get("time_bucket", "5s")
    metrics = plan.get("metrics", [])
    top_k = plan.get("top_k", 10)
    if task == "burst_detection":
        result = detect_bursts(filtered, bucket)
    elif task == "delay_jitter_analysis":
        result = detect_delay_jitter(filtered, bucket)
    elif task == "error_spike_analysis":
        result = detect_error_spikes(filtered, bucket)
    elif task == "top_entities":
        result = top_entities(filtered, group_by, metrics, top_k)
    elif task == "timeline_summary":
        bucketed = bucket_time(filtered, bucket)
        result = aggregate_metrics(bucketed, metrics, ["time_bucket"] + group_by)
        result = result.sort_values("time_bucket").reset_index(drop=True)
    else:  # custom_filter_aggregate
        result = aggregate_metrics(filtered, metrics, group_by)
        sort_col = "packet_count" if "packet_count" in result.columns else result.columns[-1]
        result = result.sort_values(sort_col, ascending=False).head(top_k)
    return result, {"rows_after_filter": len(filtered), "result_rows": len(result)}

def summarize_session_result(session_id: str, df: pd.DataFrame, plan: Dict) -> Dict:
    filtered = apply_filters(df, plan.get("filters", []))
    result, meta = run_plan(df, plan)
    ts = filtered["timestamp"].dropna() if "timestamp" in filtered.columns else pd.Series([], dtype="datetime64[ns]")
    duration = (ts.max() - ts.min()).total_seconds() if len(ts) > 1 else float("nan")
    packet_count = len(filtered)
    bytes_sum = float(filtered["frame_size_bytes"].sum()) if "frame_size_bytes" in filtered.columns else float("nan")
    task = plan["task_type"]
    summary = {
        "session_id": session_id,
        "task_type": task,
        "rows_after_filter": meta["rows_after_filter"],
        "result_rows": meta["result_rows"],
        "session_duration_s": duration,
        "packet_count": packet_count,
        "bytes_sum": bytes_sum,
        "avg_packet_size": _divide_or_nan(bytes_sum, packet_count),
        "packets_per_second": _divide_or_nan(packet_count, duration),
        "bytes_per_second": _divide_or_nan(bytes_sum, duration),
        "bad_fcs_rate": float(filtered["bad_fcs_flag"].mean()) if "bad_fcs_flag" in filtered.columns else float("nan"),
        "retry_rate": float(filtered["retry_flag"].mean()) if "retry_flag" in filtered.columns else float("nan"),
    }
    if task == "burst_detection" and "burst_z" in result.columns:
        summary["peak_burst_z"] = _safe_max(result["burst_z"])
        summary["peak_packets_per_s"] = _safe_max(result.get("packets_per_s", pd.Series()))
        summary["burst_bucket_count"] = int(result.get("is_burst", pd.Series()).sum())
        summary["jitter_score"] = float("nan")
        summary["peak_error_z"] = float("nan")
    elif task == "delay_jitter_analysis" and "jitter_score" in result.columns:
        summary["jitter_score"] = _safe_max(result["jitter_score"])
        summary["peak_p95_iat"] = _safe_max(result.get("p95_iat", pd.Series()))
        summary["peak_burst_z"] = float("nan")
        summary["peak_error_z"] = float("nan")
    elif task == "error_spike_analysis" and "error_z" in result.columns:
        summary["peak_error_z"] = _safe_max(result["error_z"])
        summary["peak_error_frame_rate"] = _safe_max(result.get("error_frame_rate", pd.Series()))
        summary["jitter_score"] = float("nan")
        summary["peak_burst_z"] = float("nan")
    # pick primary metric
    fallbacks = ["avg_packet_size","packets_per_second","bytes_per_second","bad_fcs_rate",
                 "jitter_score","peak_burst_z","peak_error_z","packet_count"]
    primary = "packet_count"
    for m in fallbacks:
        v = summary.get(m, float("nan"))
        if v is not None and not (isinstance(v, float) and math.isnan(v)) and math.isfinite(float(v)):
            primary = m; break
    summary["primary_metric"] = primary
    summary["comparison_score"] = summary.get(primary, float("nan"))
    return summary

def compare_sessions(sessions: Dict[str, pd.DataFrame], plan: Dict) -> Tuple[pd.DataFrame, Dict]:
    rows = [summarize_session_result(sid, df, plan) for sid, df in sessions.items()]
    mode = plan.get("comparison_mode", "rank")
    ascending = mode == "weakest"
    rows.sort(key=lambda r: (math.isnan(float(r.get("comparison_score", float("nan")))),
                              float(r.get("comparison_score", float("nan"))) * (-1 if not ascending else 1)))
    rank_col = "weakest_rank" if ascending else "strongest_rank"
    for i, r in enumerate(rows):
        r[rank_col] = i + 1
    result = pd.DataFrame(rows)
    return result, {
        "session_scope": "all_sessions",
        "comparison_mode": mode,
        "sessions_compared": len(rows),
        "comparison_metric": rows[0].get("primary_metric") if rows else None,
    }

def infer_cross_session_mode(question: str) -> Optional[str]:
    q = question.lower()
    phrases = ["which session","across sessions","compare sessions","among sessions","all sessions"]
    if not any(p in q for p in phrases): return None
    if any(w in q for w in ["strongest","highest","most","largest","worst","max"]): return "strongest"
    if any(w in q for w in ["weakest","lowest","least","smallest","best","min"]): return "weakest"
    return "rank"

def infer_comparison_metric(question: str, task_type: str = None) -> Optional[str]:
    q = question.lower()
    if any(p in q for p in ["average packet size","avg packet size","mean packet size"]): return "avg_packet_size"
    if any(p in q for p in ["packets per second","packet rate","pps"]): return "packets_per_second"
    if any(p in q for p in ["bytes per second","byte rate"]): return "bytes_per_second"
    if "bad fcs rate" in q or "fcs rate" in q: return "bad_fcs_rate"
    if "jitter score" in q: return "jitter_score"
    task_map = {"burst_detection":"peak_burst_z","delay_jitter_analysis":"jitter_score","error_spike_analysis":"peak_error_z"}
    return task_map.get(task_type) if task_type else None

# ============================================================
# LLM planning
# ============================================================

PLAN_SCHEMA = {
    "name": "traffic_analysis_plan",
    "description": "Create a wireless traffic analysis plan.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "task_type": {"type": "string", "enum": ["burst_detection","delay_jitter_analysis","error_spike_analysis","top_entities","timeline_summary","custom_filter_aggregate"]},
            "time_bucket": {"type": "string", "enum": ["1s","5s","10s","30s","60s"]},
            "group_by": {"type": "array", "items": {"type": "string"}},
            "filters": {"type": "array", "items": {"type": "object", "properties": {"column": {"type": "string"},"op": {"type": "string","enum": ["eq","ne","gt","ge","lt","le","contains","in","notnull"]},"value": {"anyOf": [{"type":"string"},{"type":"number"},{"type":"integer"},{"type":"boolean"},{"type":"array","items":{"anyOf":[{"type":"string"},{"type":"number"},{"type":"boolean"}]}},{"type":"null"}]}}, "required": ["column","op","value"], "additionalProperties": False}},
            "metrics": {"type": "array", "items": {"type": "string", "enum": ["packet_count","bytes_sum","mean_iat","std_iat","p95_iat","mean_duration_us","bad_fcs_count","bad_fcs_rate","error_frame_count","error_frame_rate","retry_count","retry_rate","data_frame_count","mean_signal_dbm","mean_rate_mbps"]}},
            "top_k": {"type": "integer"},
            "plot": {"type": "string", "enum": ["line","bar","none"]},
            "question_rephrased": {"type": "string"},
            "explanation_focus": {"type": "array", "items": {"type": "string", "enum": ["bursty_traffic","delay","jitter","errors","link_quality","protocol_behavior"]}}
        },
        "required": ["session_id","task_type","time_bucket","group_by","filters","metrics","top_k","plot","question_rephrased","explanation_focus"],
        "additionalProperties": False
    }
}

def ask_llm_for_plan(question: str, session_summaries: Dict) -> Dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    session_ids = list(session_summaries.keys())
    default = sorted(session_ids)[-1] if session_ids else ""
    prompt = f"""You are helping analyze wireless trace tables exported from Teledyne LeCroy.
Convert the user's question into a conservative JSON analysis plan.

Rules:
- Pick one session_id from: {json.dumps(session_ids)}
- If user did not specify a session, use "{default}"
- Only use columns that plausibly exist in these summaries
- Prefer packet_count, bytes_sum, mean_iat, std_iat, p95_iat for burst/delay/jitter questions
- Prefer bad_fcs_count, bad_fcs_rate, error_frame_count, retry_rate for error questions
- Prefer line plot for time-series, bar for top-k or comparison
- Keep plan simple and minimal; group_by may be empty

Session summaries:
{json.dumps(session_summaries, indent=2, default=str)}

User question:
{question}

Call the traffic_analysis_plan tool with the plan."""
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "function", "function": PLAN_SCHEMA}],
        tool_choice={"type": "function", "function": {"name": "traffic_analysis_plan"}},
    )
    args = resp.choices[0].message.tool_calls[0].function.arguments
    return json.loads(args)

def summarize_with_llm(question: str, plan: Dict, meta: Dict, preview: List[Dict]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "(OpenAI API key not configured — set OPENAI_API_KEY to enable AI summaries)"
    client = OpenAI(api_key=api_key)
    prompt = f"""You are writing a short network-traffic analysis summary.

User question:
{question}

Executed plan:
{json.dumps(plan, indent=2)}

Execution metadata:
{json.dumps(meta, indent=2)}

Top result rows:
{json.dumps(preview, indent=2, default=str)}

Write:
1. A short answer to the question
2. Brief interpretation (bursty? high jitter? errors?)
3. If anomalies appear, an incident-style explanation with likely causes
4. Do not invent fields not present
5. Under 250 words
6. If cross-session, name the top-ranked session and the comparison_score basis"""
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# ============================================================
# Serialization helpers
# ============================================================

def df_to_json(df: pd.DataFrame, max_rows: int = 500) -> List[Dict]:
    df2 = df.head(max_rows).copy()
    for col in df2.select_dtypes(include=["datetime64[ns, UTC]", "datetime64[ns]"]).columns:
        df2[col] = df2[col].astype(str)
    for col in df2.columns:
        df2[col] = df2[col].where(df2[col].notna(), None)
    rows = df2.to_dict(orient="records")
    clean = []
    for r in rows:
        cr = {}
        for k, v in r.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                cr[k] = None
            elif hasattr(v, "item"):
                cr[k] = v.item()
            else:
                cr[k] = v
        clean.append(cr)
    return clean

# ============================================================
# Flask routes
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".zip"):
        return jsonify({"error": "Expected a .zip file"}), 400
    token = request.headers.get("X-Session-Token", "default")

    # Windows fix: close the temp file handle before saving/reading/deleting it.
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        f.save(tmp_path)
        sessions = discover_and_load_zip(tmp_path)
        if not sessions:
            return jsonify({"error": "No valid sessions found. Expected folders containing tap/mac/error/udp CSVs."}), 400
        _SESSION_STORE[token] = sessions
        info = [{"id": sid, "rows": len(df), "columns": list(df.columns)} for sid, df in sessions.items()]
        return jsonify({"sessions": info})
    except Exception as e:
        import traceback; traceback.print_exc()
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
        summaries = {sid: summarize_schema(df) for sid, df in sessions.items()}
        plan = ask_llm_for_plan(question, summaries)
        # cross-session inference
        mode = infer_cross_session_mode(question)
        is_cross = bool(mode)
        if is_cross:
            plan["session_scope"] = "all_sessions"
            plan["comparison_mode"] = mode
            plan["comparison_metric"] = infer_comparison_metric(question, plan.get("task_type"))
            plan["plot"] = "bar"
            result_df, meta = compare_sessions(sessions, plan)
        else:
            plan["session_scope"] = "single_session"
            plan["comparison_mode"] = "none"
            plan["comparison_metric"] = None
            default = sorted(sessions.keys())[-1]
            target_id = plan.get("session_id", default)
            if target_id not in sessions:
                target_id = default
            plan["session_id"] = target_id
            result_df, meta = run_plan(sessions[target_id], plan)
        preview = df_to_json(result_df, max_rows=20)
        narrative = summarize_with_llm(question, plan, meta, preview)
        result_rows = df_to_json(result_df)
        return jsonify({
            "plan": plan,
            "result": result_rows,
            "meta": meta,
            "narrative": narrative,
            "is_cross_session": is_cross,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/clear", methods=["POST"])
def clear():
    token = request.headers.get("X-Session-Token", "default")
    _SESSION_STORE.pop(token, None)
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
