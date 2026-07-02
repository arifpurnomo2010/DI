"""
Country-Segment Insight Browser
===============================

Standalone viewer for:
- Top selector: all Country|Segment pairs from Model.csv
- Left panel: sentiment summary by model for selected pair
- Right panel: expandable details tree
  Model > Sentiment > Category > Parameter > Phrase (Original + English)

Run:
    shiny run app_insight_browser.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path
import pandas as pd
from shiny import App, reactive, render, ui

BASE_DIR = Path(__file__).resolve().parent
MODEL_CSV = BASE_DIR / "Model.csv"
OUTPUT_DIR = BASE_DIR / "Output"

SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
NON_SUBSTANTIAL_FLAGS = {"API_FAILED", "LONG_NONSUBSTANTIAL"}
COUNTRY_MAP = {"VIETNAM": "Vietnam", "INDONESIA": "Indonesia", "PHILIPPINES": "Philippines"}
COUNTRY_CODE = {"Vietnam": "VN", "Indonesia": "ID", "Philippines": "PH"}
FOLDER_COUNTRY_MAP = {"VN": "Vietnam", "ID": "Indonesia", "PH": "Philippines"}


def _read_csv_flex(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except Exception:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(s or "").lower()).strip()


def _split_multi(v: str) -> list[str]:
    s = str(v or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def _is_substantial(review_flag: str, category: str) -> bool:
    flags = {f.strip() for f in str(review_flag or "").split("|") if f.strip()}
    if flags & NON_SUBSTANTIAL_FLAGS:
        return False
    cats = {c.strip() for c in str(category or "").split(";") if c.strip()}
    if not cats or cats <= {"Other"}:
        return False
    return True


def parse_model_segment_map(df_model: pd.DataFrame | None) -> pd.DataFrame:
    if df_model is None or df_model.empty:
        return pd.DataFrame(columns=["country", "segment", "model_name", "model_key"])

    raw = df_model.copy().fillna("").astype(str)
    rows: list[dict] = []

    current_country: str | None = None
    segments_by_col: dict[int, str] = {}

    def _extract_segment_columns(seg_vals: list[str]) -> dict[int, str]:
        """
        Read only the left segment block from the segment header row.
        The file also contains a right-side metadata table (Country/Model/Launch_*)
        that must never be interpreted as segment names.
        """
        out: dict[int, str] = {}
        started = False
        for c, seg in enumerate(seg_vals):
            seg = str(seg or "").strip()
            if not seg:
                if started:
                    # Stop once the first segment block ends.
                    break
                continue
            started = True
            out[c] = seg
        return out
    i = 0
    while i < len(raw):
        vals = [str(x).strip() for x in raw.iloc[i].tolist()]
        first = vals[0].upper() if vals else ""

        if first in COUNTRY_MAP:
            current_country = COUNTRY_MAP[first]
            segments_by_col = {}
            if i + 1 < len(raw):
                seg_vals = [str(x).strip() for x in raw.iloc[i + 1].tolist()]
                segments_by_col = _extract_segment_columns(seg_vals)
            i += 2
            continue

        if current_country and segments_by_col:
            for c, segment in segments_by_col.items():
                if c >= len(vals):
                    continue
                cell = vals[c].strip()
                if not cell:
                    continue
                for name in re.split(r",\s*", cell):
                    model_name = name.strip().strip('"')
                    if not model_name:
                        continue
                    rows.append(
                        {
                            "country": current_country,
                            "segment": segment,
                            "model_name": model_name,
                            "model_key": _norm(model_name),
                        }
                    )
        i += 1

    out = pd.DataFrame(rows).drop_duplicates(subset=["country", "segment", "model_key"])
    return out


def _explode_step5_fallback(df: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in df.iterrows():
        cats = _split_multi(r.get("category", ""))
        params = _split_multi(r.get("parameter", ""))
        sents = _split_multi(r.get("sentiment", "")) or [str(r.get("sentiment", "Neutral")).strip() or "Neutral"]
        conf = pd.to_numeric(r.get("confidence"), errors="coerce")
        n = max(1, len(cats), len(params), len(sents))
        for idx in range(n):
            cat = cats[idx] if idx < len(cats) else (cats[-1] if cats else "Other")
            prm = params[idx] if idx < len(params) else (params[-1] if params else "General")
            sent = sents[idx] if idx < len(sents) else sents[-1]
            review_flag = str(r.get("review_flag", "OK"))
            rows.append(
                {
                    "country": country or r.get("country", ""),
                    "record_id": r.get("record_id"),
                    "focus_model": r.get("focus_model"),
                    "aspect_index": idx,
                    "category": cat,
                    "parameter": prm,
                    "sentiment": sent,
                    "confidence": conf,
                    "review_flag": review_flag,
                    "is_substantial": _is_substantial(review_flag, cat),
                }
            )
    return pd.DataFrame(rows)


def _guess_country_from_name(name: str) -> str:
    up = str(name or "").upper()
    for code, country in FOLDER_COUNTRY_MAP.items():
        if f"_{code}_" in f"_{up}_" or f"/{code}/" in up.replace("\\", "/"):
            return country
    return ""


def load_output_data(
    aspects_files: list[pd.DataFrame],
    pipeline_files: list[pd.DataFrame],
    step5_files: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_aspects: list[pd.DataFrame] = []
    all_pipeline: list[pd.DataFrame] = []

    for pipe in pipeline_files:
        if pipe is None or pipe.empty:
            continue
        p = pipe.copy()
        if "country" not in p.columns:
            p["country"] = ""
        all_pipeline.append(p)

    for asp in aspects_files:
        if asp is None or asp.empty:
            continue
        a = asp.copy()
        if "is_substantial" not in a.columns:
            a["is_substantial"] = a.apply(
                lambda r: _is_substantial(r.get("review_flag", "OK"), r.get("category", "Other")), axis=1
            )
        a = a[a["is_substantial"] == True].copy()
        a = a[a["category"].astype(str).str.strip() != "Other"].copy()
        if "country" not in a.columns:
            a["country"] = ""
        all_aspects.append(a)

    for step5 in step5_files:
        if step5 is None or step5.empty:
            continue
        s = _explode_step5_fallback(step5)
        if s.empty:
            continue
        s = s[s["is_substantial"] == True].copy()
        s = s[s["category"].astype(str).str.strip() != "Other"].copy()
        all_aspects.append(s)

    aspects_df = pd.concat(all_aspects, ignore_index=True) if all_aspects else pd.DataFrame()
    pipeline_df = pd.concat(all_pipeline, ignore_index=True) if all_pipeline else pd.DataFrame()
    return aspects_df, pipeline_df


def build_joined_facts(model_map: pd.DataFrame, aspects: pd.DataFrame, pipeline: pd.DataFrame) -> pd.DataFrame:
    if aspects is None or aspects.empty:
        return pd.DataFrame()

    facts = aspects.copy()
    if "country" not in facts.columns:
        facts["country"] = ""
    facts["country"] = facts["country"].astype(str).str.strip()
    facts["focus_model"] = facts["focus_model"].astype(str).str.strip()

    if pipeline is not None and not pipeline.empty:
        keep_cols = [
            c for c in [
                "country", "record_id", "focus_model",
                "content_original", "content_english",
                "phrase_original", "phrase_english",
                "wow_factor_score",
            ] if c in pipeline.columns
        ]
        phrase = pipeline[keep_cols].copy()
        # Primary join by country+record_id+focus_model when available;
        # fallback to record_id+focus_model so country can be backfilled from pipeline.
        if all(k in phrase.columns and k in facts.columns for k in ["record_id", "focus_model", "country"]):
            join_keys = ["country", "record_id", "focus_model"]
        elif all(k in phrase.columns and k in facts.columns for k in ["record_id", "focus_model"]):
            join_keys = ["record_id", "focus_model"]
        else:
            join_keys = []

        if join_keys:
            phrase = phrase.drop_duplicates(subset=join_keys)
            facts = facts.merge(phrase, on=join_keys, how="left", suffixes=("", "_pipe"))
            if "country_pipe" in facts.columns:
                facts["country"] = facts["country"].replace("", pd.NA).fillna(facts["country_pipe"]).fillna("")
                facts = facts.drop(columns=["country_pipe"], errors="ignore")
            # If aspects already has wow_factor_score (often empty), backfill from pipeline.
            if "wow_factor_score_pipe" in facts.columns:
                if "wow_factor_score" in facts.columns:
                    facts["wow_factor_score"] = facts["wow_factor_score"].replace("", pd.NA).fillna(
                        facts["wow_factor_score_pipe"]
                    )
                else:
                    facts["wow_factor_score"] = facts["wow_factor_score_pipe"]
                facts = facts.drop(columns=["wow_factor_score_pipe"], errors="ignore")
        else:
            facts["content_original"] = facts.get("content_original", "")
            facts["content_english"] = facts.get("content_english", "")
            facts["phrase_original"] = facts.get("phrase_original", "")
            facts["phrase_english"] = facts.get("phrase_english", "")
    else:
        facts["content_original"] = ""
        facts["content_english"] = ""
        facts["phrase_original"] = ""
        facts["phrase_english"] = ""

    # Backward compatibility: historical outputs may not include phrase columns.
    for col in ["phrase_original", "phrase_english"]:
        if col not in facts.columns:
            facts[col] = ""

    facts["country"] = facts["country"].astype(str).str.strip().replace("", "Unknown")

    facts["model_key"] = facts["focus_model"].apply(_norm)
    seg_map = model_map[["country", "segment", "model_key", "model_name"]].drop_duplicates()
    facts = facts.merge(seg_map, on=["country", "model_key"], how="left")
    facts["display_model"] = facts["model_name"].fillna(facts["focus_model"])

    facts["sentiment"] = (
        facts["sentiment"].astype(str).str.strip().str.capitalize().where(lambda s: s.isin(SENTIMENT_ORDER), "Neutral")
    )
    for col in ["content_original", "content_english", "phrase_original", "phrase_english"]:
        if col in facts.columns:
            facts[col] = facts[col].fillna("")
    if "wow_factor_score" not in facts.columns:
        facts["wow_factor_score"] = pd.NA
    return facts


def load_output_data_auto(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_aspects: list[pd.DataFrame] = []
    all_pipeline: list[pd.DataFrame] = []

    if not output_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        country = FOLDER_COUNTRY_MAP.get(d.name.upper(), d.name.title())

        p_aspects = d / "output_step5_aspects.csv"
        p_step5 = d / "output_step5.csv"
        p_pipeline = d / "pipeline_summary.csv"

        aspects = pd.DataFrame()
        if p_aspects.exists():
            aspects = _read_csv_flex(p_aspects)
            if "country" not in aspects.columns:
                aspects["country"] = country
        elif p_step5.exists():
            step5 = _read_csv_flex(p_step5)
            aspects = _explode_step5_fallback(step5, country)

        if not aspects.empty:
            if "country" not in aspects.columns:
                aspects["country"] = country
            if "is_substantial" not in aspects.columns:
                aspects["is_substantial"] = aspects.apply(
                    lambda r: _is_substantial(r.get("review_flag", "OK"), r.get("category", "Other")), axis=1
                )
            aspects = aspects[aspects["is_substantial"] == True].copy()
            aspects = aspects[aspects["category"].astype(str).str.strip() != "Other"].copy()
            all_aspects.append(aspects)

        if p_pipeline.exists():
            pipe = _read_csv_flex(p_pipeline)
            if "country" not in pipe.columns:
                pipe["country"] = country
            all_pipeline.append(pipe)

    aspects_df = pd.concat(all_aspects, ignore_index=True) if all_aspects else pd.DataFrame()
    pipeline_df = pd.concat(all_pipeline, ignore_index=True) if all_pipeline else pd.DataFrame()
    return aspects_df, pipeline_df


def _build_segment_choices(model_map: pd.DataFrame) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    if model_map is None or model_map.empty:
        return choices
    combos = model_map[["country", "segment"]].drop_duplicates().sort_values(["country", "segment"])
    for _, r in combos.iterrows():
        country = str(r["country"])
        segment = str(r["segment"])
        key = f"{country}|||{segment}"
        label = f"{COUNTRY_CODE.get(country, country[:2].upper())}|{segment}"
        choices.append((key, label))
    return choices


def _load_model_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Model.csv not found at {path} — segment selector will be empty.")
        return pd.DataFrame()
    try:
        return _read_csv_flex(path, header=None, dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[WARN] Failed to read Model.csv at {path}: {e}")
        return pd.DataFrame()


MODEL_MAP_DF = parse_model_segment_map(_load_model_csv_safe(MODEL_CSV))
ASPECTS_DF, PIPELINE_DF = load_output_data_auto(OUTPUT_DIR)
FACTS_DF = build_joined_facts(MODEL_MAP_DF, ASPECTS_DF, PIPELINE_DF)
SEGMENT_CHOICES = _build_segment_choices(MODEL_MAP_DF)
CHOICES_DICT = {k: v for k, v in SEGMENT_CHOICES}
DEFAULT_SEGMENT = SEGMENT_CHOICES[0][0] if SEGMENT_CHOICES else "__none__"


def _compute_summary(models: list[str], facts: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"Model": models})
    if not models:
        return base
    if facts is None or facts.empty:
        out = base.copy()
        for s in SENTIMENT_ORDER:
            out[s] = 0.0
        return out

    cnt = facts.groupby(["display_model", "sentiment"]).size().reset_index(name="n")
    piv = cnt.pivot_table(index="display_model", columns="sentiment", values="n", fill_value=0).reset_index()
    piv = piv.rename(columns={"display_model": "Model"})
    for s in SENTIMENT_ORDER:
        if s not in piv.columns:
            piv[s] = 0
    out = base.merge(piv[["Model", *SENTIMENT_ORDER]], on="Model", how="left").fillna(0)
    out[SENTIMENT_ORDER] = out[SENTIMENT_ORDER].astype(float)
    tot = out[SENTIMENT_ORDER].sum(axis=1).replace(0, 1)
    out["Positive"] = (out["Positive"] / tot * 100).round(1)
    out["Neutral"] = (out["Neutral"] / tot * 100).round(1)
    out["Negative"] = (out["Negative"] / tot * 100).round(1)
    return out


def _render_vertical_stacked_chart(summary_pct: pd.DataFrame) -> str:
    if summary_pct is None or summary_pct.empty:
        return "<div style='color:#94A3B8'>No data.</div>"

    palette = {"Positive": "#4CAF50", "Negative": "#E91E63", "Neutral": "#F4C542"}
    legend = (
        "<div style='display:flex;gap:14px;margin-bottom:8px;font-size:12px'>"
        "<span><span style='display:inline-block;width:10px;height:10px;background:#4CAF50;margin-right:4px'></span>Positive</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;background:#E91E63;margin-right:4px'></span>Negative</span>"
        "<span><span style='display:inline-block;width:10px;height:10px;background:#F4C542;margin-right:4px'></span>Neutral</span>"
        "</div>"
    )

    bars: list[str] = []
    for _, r in summary_pct.iterrows():
        model = html.escape(str(r.get("Model", "")))
        p = float(r.get("Positive", 0))
        n = float(r.get("Negative", 0))
        u = float(r.get("Neutral", 0))
        segs = [("Positive", p), ("Negative", n), ("Neutral", u)]

        stack = ""
        for name, val in segs:
            if val <= 0:
                continue
            label = f"{val:.0f}%" if val >= 10 else ""
            stack += (
                f"<div title='{name}: {val:.1f}%' style='height:{val}%;background:{palette[name]};"
                "display:flex;align-items:center;justify-content:center;font-size:10px;color:white;font-weight:600'>"
                f"{label}</div>"
            )
        if not stack:
            stack = "<div style='height:100%;background:#F1F5F9'></div>"

        bars.append(
            "<div style='display:flex;flex-direction:column;align-items:center;width:86px'>"
            "<div style='height:220px;width:56px;border:1px solid #E2E8F0;display:flex;flex-direction:column-reverse;overflow:hidden'>"
            f"{stack}</div>"
            f"<div style='margin-top:6px;font-size:11px;text-align:center;line-height:1.15'>{model}</div>"
            "</div>"
        )

    return (
        legend
        + "<div style='display:flex;gap:12px;align-items:flex-end;overflow-x:auto;padding-bottom:6px'>"
        + "".join(bars)
        + "</div>"
        + "<div style='font-size:11px;color:#64748B;margin-top:4px'>% of mentions by model</div>"
    )


app_ui = ui.page_fluid(
    ui.h2("ASEAN Auto NLP Intelligence"),
    ui.p(
        "Consumer Voice across Indonesia, Vietnam, Philippines",
        class_="text-muted",
    ),
    ui.input_radio_buttons(
        "country_segment",
        "Country|Segment",
        choices=CHOICES_DICT if CHOICES_DICT else {"__none__": "(no segments found in Model.csv)"},
        selected=DEFAULT_SEGMENT,
        inline=True,
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Sentiment Summary by Model"),
            ui.output_ui("summary_chart"),
        ),
        ui.card(
            ui.card_header("Detailed Insights"),
            ui.output_ui("detail_tree"),
        ),
        col_widths=[5, 7],
    ),
)


def server(input, output, session):
    def _filter_confidence(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame() if df is None else df
        if "confidence" not in df.columns:
            return df
        return df[pd.to_numeric(df["confidence"], errors="coerce").fillna(0) > 0.9].copy()

    @reactive.calc
    def selected_context():
        key = input.country_segment()
        if not key or key == "__none__":
            return {"country": None, "segment": None, "models": [], "facts": pd.DataFrame()}

        country, segment = key.split("|||", 1)
        model_rows = MODEL_MAP_DF[(MODEL_MAP_DF["country"] == country) & (MODEL_MAP_DF["segment"] == segment)].copy()
        models = sorted(model_rows["model_name"].astype(str).tolist())
        if FACTS_DF.empty:
            return {"country": country, "segment": segment, "models": models, "facts": pd.DataFrame()}

        f = FACTS_DF[(FACTS_DF["country"] == country) & (FACTS_DF["segment"] == segment)].copy()
        return {"country": country, "segment": segment, "models": models, "facts": f}

    @output
    @render.ui
    def summary_chart():
        ctx = selected_context()
        models = ctx["models"]
        f = ctx["facts"]

        if MODEL_MAP_DF.empty:
            return ui.p("Model.csv not found or unreadable.", class_="text-muted")

        if not models:
            return ui.p("No models mapped for this Country|Segment in Model.csv.", class_="text-muted")

        f_summary = _filter_confidence(f)
        summary_pct = _compute_summary(models, f_summary).sort_values("Model").reset_index(drop=True)
        return ui.HTML(_render_vertical_stacked_chart(summary_pct))

    @output
    @render.ui
    def detail_tree():
        ctx = selected_context()
        country = ctx["country"]
        segment = ctx["segment"]
        models = ctx["models"]
        f = ctx["facts"]

        if not country or not segment:
            return ui.p("Select a Country|Segment first.", class_="text-muted")

        if not models:
            return ui.p("No model mapping found for this Country|Segment in Model.csv.", class_="text-muted")

        parts: list[str] = []
        parts.append(f"<div style='font-size:0.85rem;color:#64748B;margin-bottom:8px'>{html.escape(country)} | {html.escape(segment)}</div>")

        for model in models:
            mf = f[f["display_model"] == model].copy() if not f.empty else pd.DataFrame()
            mf = _filter_confidence(mf)
            model_n = len(mf)
            parts.append(f"<details><summary><b>{html.escape(model)}</b> ({model_n} insights)</summary>")

            if mf.empty:
                parts.append("<div style='margin:6px 0 8px 16px;color:#94A3B8'>No insights available for this model.</div></details>")
                continue

            for sentiment in SENTIMENT_ORDER:
                sf = mf[mf["sentiment"] == sentiment].copy()
                if sf.empty:
                    continue
                parts.append(f"<details style='margin-left:14px'><summary>{html.escape(sentiment)} ({len(sf)})</summary>")
                cat_counts = sf["category"].astype(str).value_counts()
                for cat in cat_counts.index.tolist():
                    cf = sf[sf["category"].astype(str) == cat].copy()
                    parts.append(f"<details style='margin-left:14px'><summary>{html.escape(cat)} ({len(cf)})</summary>")
                    param_counts = cf["parameter"].astype(str).value_counts()
                    for param in param_counts.index.tolist():
                        pf = cf[cf["parameter"].astype(str) == param].copy()
                        parts.append(
                            f"<details style='margin-left:14px'><summary>{html.escape(param)} ({len(pf)})</summary>"
                        )
                        leaves = pf[
                            [
                                c
                                for c in [
                                    "record_id",
                                    "phrase_original",
                                    "phrase_english",
                                    "content_original",
                                    "content_english",
                                    "wow_factor_score",
                                    "mention_target",
                                ]
                                if c in pf.columns
                            ]
                        ].drop_duplicates()
                        if "wow_factor_score" in leaves.columns:
                            leaves["wow_factor_score_num"] = pd.to_numeric(
                                leaves["wow_factor_score"], errors="coerce"
                            )
                            leaves = leaves.sort_values(
                                "wow_factor_score_num", ascending=False, na_position="last"
                            )
                        if leaves.empty:
                            parts.append(
                                "<div style='margin:6px 0 8px 16px;color:#94A3B8'>No phrase available.</div>"
                            )
                        else:
                            parts.append("<ul style='margin:6px 0 8px 16px'>")
                            for _, r in leaves.iterrows():
                                rec_id = html.escape(str(r.get("record_id", "")))
                                phrase_orig_raw = str(r.get("phrase_original", "")).strip()
                                phrase_eng_raw = str(r.get("phrase_english", "")).strip()
                                content_orig_raw = str(r.get("content_original", "")).strip()
                                content_eng_raw = str(r.get("content_english", "")).strip()
                                orig = html.escape(phrase_orig_raw or content_orig_raw) or "-"
                                eng = html.escape(phrase_eng_raw or content_eng_raw or phrase_orig_raw or content_orig_raw) or "-"

                                # Badge mention_target
                                mention_target = str(r.get("mention_target", "") or "").strip().lower()
                                if mention_target == "competitor":
                                    badge_html = (
                                        "<span style='"
                                        "display:inline-block;margin-left:6px;padding:1px 6px;"
                                        "background:#FFF3CD;color:#92400E;border:1px solid #F59E0B;"
                                        "border-radius:10px;font-size:10px;font-weight:600;"
                                        "vertical-align:middle;cursor:default"
                                        "' title='This phrase is about a competitor vehicle'>"
                                        "⚠ Competitor"
                                        "</span>"
                                    )
                                elif mention_target == "general":
                                    badge_html = (
                                        "<span style='"
                                        "display:inline-block;margin-left:6px;padding:1px 6px;"
                                        "background:#F1F5F9;color:#475569;border:1px solid #CBD5E1;"
                                        "border-radius:10px;font-size:10px;font-weight:600;"
                                        "vertical-align:middle;cursor:default"
                                        "' title='General market observation - not specific to this vehicle'>"
                                        "ℹ General"
                                        "</span>"
                                    )
                                else:
                                    badge_html = ""

                                # Bangun tooltip text: prioritas full comment, fallback ke phrase.
                                full_orig = html.escape(content_orig_raw) if content_orig_raw else ""
                                full_eng = html.escape(content_eng_raw) if content_eng_raw else ""
                                phrase_orig_tip = html.escape(phrase_orig_raw) if phrase_orig_raw else ""
                                phrase_eng_tip = html.escape(phrase_eng_raw) if phrase_eng_raw else ""

                                tooltip_orig = full_orig or phrase_orig_tip
                                tooltip_eng = full_eng or phrase_eng_tip
                                if tooltip_orig and tooltip_eng:
                                    tooltip_text = (
                                        f"DETAIL&#10;Original: {tooltip_orig[:400]}&#10;English: {tooltip_eng[:400]}"
                                    )
                                elif tooltip_eng:
                                    tooltip_text = f"DETAIL&#10;{tooltip_eng[:400]}"
                                elif tooltip_orig:
                                    tooltip_text = f"DETAIL&#10;{tooltip_orig[:400]}"
                                else:
                                    tooltip_text = "DETAIL&#10;No extra context."

                                # Render English phrase — dengan tooltip jika tersedia, plain jika tidak
                                eng_display = (
                                    f"<span title='{tooltip_text}' style='cursor:help;border-bottom:1px dotted #94A3B8'>"
                                    f"{eng[:240]}{'...' if len(eng) > 240 else ''}</span>"
                                    if tooltip_text
                                    else f"{eng[:240]}{'...' if len(eng) > 240 else ''}"
                                )
                                parts.append(
                                    "<li style='margin-bottom:8px'>"
                                    f"<div><b>record_id:</b> {rec_id}</div>"
                                    f"<div><b>Original:</b> {orig[:240]}{'...' if len(orig) > 240 else ''}</div>"
                                    f"<div><b>English:</b> {eng_display}{badge_html}</div>"
                                    "</li>"
                                )
                            parts.append("</ul>")
                        parts.append("</details>")
                    parts.append("</details>")
                parts.append("</details>")
            parts.append("</details>")

        return ui.HTML("".join(parts))


app = App(app_ui, server)

