"""
streamlit_app.py
────────────────
AI CEO — Strategic Intelligence Dashboard (the only frontend).

Reads the pipeline outputs (data/analysis.json + data/sentiment.json + the
collected corpus) and renders the board briefing, plus a live CEO Q&A chat that
runs the tool-calling agent's grounded Q&A on your local model.

Run locally:
    streamlit run streamlit_app.py

Prereqs (run the pipeline first so there's data to show):
    python pipeline.py
And have Ollama running with the configured model (config.LLM_MODEL) for the
live Q&A chat:
    ollama serve

Sections:
    1. Header / company banner + key metrics
    2. Market Intelligence  (recent docs · trends · competitor set)
    3. Opportunity Monitor
    4. Risk Monitor
    5. Sentiment Analysis    (Plotly gauge + per-source bars)
    6. Strategic Recommendations
    7. CEO Briefing
    +  Live CEO Q&A (chat)
"""
import glob
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ───────────────────────────────────────────────────────────── data loading
def _load(path, default):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return default


def load_analysis():
    return _load(config.ANALYSIS_CACHE, {})


def load_sentiment():
    return _load(os.path.join(config.DATA_DIR, "sentiment.json"), {})


def load_corpus_docs():
    docs, skip = [], {"clean_corpus.json", "analysis.json", "sentiment.json"}
    for path in glob.glob(os.path.join(config.DATA_DIR, "*.json")):
        if os.path.basename(path) in skip:
            continue
        data = _load(path, [])
        if isinstance(data, list):
            docs.extend(d for d in data if isinstance(d, dict) and d.get("title"))
    docs.sort(key=lambda d: d.get("published", ""), reverse=True)
    return docs


# ───────────────────────────────────────────────────────────── small helpers
_LEVEL_COLOR = {"high": "#ef4444", "medium": "#f59e0b", "low": "#22c55e"}


def chip(level):
    lv = (level or "").lower()
    c = _LEVEL_COLOR.get(lv, "#64748b")
    return (f"<span style='background:{c}22;color:{c};border:1px solid {c}55;"
            f"border-radius:999px;padding:1px 8px;font-size:.72rem;font-weight:600;"
            f"text-transform:capitalize'>{level or '—'}</span>")


def neutral_chip(text):
    return (f"<span style='background:#1d2942;color:#9fb3d1;border:1px solid #1f2a3d;"
            f"border-radius:999px;padding:1px 8px;font-size:.72rem'>{text}</span>")


def evidence_md(items):
    if not items:
        return "<span style='color:#5d748f;font-size:.74rem'>no evidence linked</span>"
    out = []
    for e in items:
        if isinstance(e, dict):
            title, src, url = e.get("title", ""), e.get("source", "?"), e.get("url", "")
        else:
            title, src, url = str(e), "?", ""
        label = f"({src}) {title[:80]}"
        if url:
            out.append(f"<a href='{url}' target='_blank' style='color:#9fb3d1;font-size:.74rem;"
                       f"border:1px solid #1f2a3d;border-radius:6px;padding:1px 6px;"
                       f"margin-right:4px;text-decoration:none'>{label}</a>")
        else:
            out.append(f"<span style='color:#9fb3d1;font-size:.74rem;border:1px solid #1f2a3d;"
                       f"border-radius:6px;padding:1px 6px;margin-right:4px'>{label}</span>")
    return "".join(out)


def section_header(idx, title, note=""):
    note_html = f"<span style='margin-left:auto;color:#8aa0bd;font-size:.85rem'>{note}</span>" if note else ""
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.6rem;margin:1.4rem 0 .6rem;"
        f"border-bottom:1px solid #1f2a3d;padding-bottom:.5rem'>"
        f"<span style='color:#4f8cff;font-weight:700;background:#13203a;border:1px solid #1f2a3d;"
        f"border-radius:8px;padding:2px 8px;font-size:.85rem'>{idx:02d}</span>"
        f"<span style='font-size:1.15rem;font-weight:600'>{title}</span>{note_html}</div>",
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────── charts
def sentiment_gauge(overall):
    import plotly.graph_objects as go
    color = "#22c55e" if overall > 0.05 else "#ef4444" if overall < -0.05 else "#94a3b8"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall,
        number={"font": {"size": 34, "color": color}},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "#33415c"},
            "bar": {"color": color},
            "bgcolor": "#0c1320",
            "borderwidth": 1, "bordercolor": "#1f2a3d",
            "steps": [
                {"range": [-1, -0.05], "color": "#3b1d22"},
                {"range": [-0.05, 0.05], "color": "#23292e"},
                {"range": [0.05, 1], "color": "#173028"},
            ],
        },
        title={"text": "Overall compound", "font": {"size": 13, "color": "#8aa0bd"}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf7"})
    return fig


def sentiment_by_source_bar(by_source):
    import plotly.graph_objects as go
    srcs = sorted(by_source.keys())
    avgs = [by_source[s].get("avg_compound", 0) for s in srcs]
    colors = ["#22c55e" if a > 0.05 else "#ef4444" if a < -0.05 else "#94a3b8" for a in avgs]
    fig = go.Figure(go.Bar(x=avgs, y=srcs, orientation="h",
                           marker_color=colors,
                           text=[f"{a:+.2f}" for a in avgs], textposition="auto"))
    fig.update_layout(height=max(220, 44 * len(srcs)),
                      margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#e6edf7"},
                      xaxis={"range": [-1, 1], "zeroline": True, "zerolinecolor": "#33415c",
                             "gridcolor": "#1f2a3d"},
                      yaxis={"gridcolor": "#1f2a3d"})
    return fig


# ───────────────────────────────────────────────────────────── page
def main():
    st.set_page_config(page_title=f"AI CEO · {config.COMPANY}",
                       page_icon="🧭", layout="wide")

    analysis = load_analysis()
    sentiment = load_sentiment()
    docs = load_corpus_docs()

    company = analysis.get("company", config.COMPANY)
    industry = analysis.get("industry", config.INDUSTRY)
    engine = analysis.get("_engine", "")
    opps = analysis.get("opportunities", [])
    risks = analysis.get("risks", [])
    trends = analysis.get("trends", [])
    recs = analysis.get("recommendations", [])
    brief = analysis.get("ceo_briefing", {})

    # 1 ── Banner
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#16213b,#0d1322);"
        f"border:1px solid #1f2a3d;border-radius:16px;padding:1.4rem 1.6rem;margin-bottom:.6rem'>"
        f"<div style='color:#4f8cff;font-size:.78rem;text-transform:uppercase;letter-spacing:.12em;"
        f"font-weight:600'>AI CEO · Strategic Intelligence</div>"
        f"<div style='font-size:2rem;font-weight:700;letter-spacing:-.02em'>{company}</div>"
        f"<div style='color:#8aa0bd'>{industry}"
        + (f" &nbsp;·&nbsp; <span style='font-size:.8rem'>engine: {engine}</span>" if engine else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )

    if not analysis:
        st.warning("No analysis found. Run `python pipeline.py` first to generate "
                   "data/analysis.json, then refresh.")

    # key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Documents", len(docs))
    c2.metric("Opportunities", len(opps))
    c3.metric("Risks", len(risks))
    c4.metric("Recommendations", len(recs))
    overall = sentiment.get("overall_compound", 0.0)
    c5.metric("Sentiment", f"{overall:+.2f}")

    # 2 ── Market intelligence
    section_header(2, "Market Intelligence", f"{len(docs)} documents")
    mleft, mright = st.columns([3, 2])
    with mleft:
        st.markdown("**Recent News & Announcements**")
        if docs:
            for d in docs[:8]:
                src = d.get("source", "?")
                title = d.get("title", "")[:110]
                url = d.get("url", "")
                line = f"<span style='color:#8aa0bd;font-size:.72rem;text-transform:uppercase'>{src}</span> &nbsp;"
                line += f"<a href='{url}' target='_blank' style='color:#4f8cff;text-decoration:none'>{title}</a>" if url else f"<span>{title}</span>"
                st.markdown(line, unsafe_allow_html=True)
        else:
            st.caption("No documents collected yet.")
    with mright:
        st.markdown("**Emerging Trends to Monitor**")
        if trends:
            for t in trends[:6]:
                st.markdown(f"▸ {t.get('title','')}", unsafe_allow_html=True)
        else:
            st.caption("No trends generated yet.")
        st.markdown("**Competitor Set**")
        st.markdown(" ".join(neutral_chip(c) for c in config.COMPETITORS),
                    unsafe_allow_html=True)

    # 3 ── Opportunities
    section_header(3, "Opportunity Monitor", f"{len(opps)} identified")
    if opps:
        for o in opps:
            with st.container(border=True):
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;gap:.8rem'>"
                    f"<span style='font-weight:650'>{o.get('title','')}</span>"
                    f"<span style='font-size:.8rem;color:#8aa0bd;white-space:nowrap'>"
                    f"impact {chip(o.get('impact'))} · conf {round(float(o.get('confidence',0)),2)}</span></div>"
                    f"<div style='color:#c4d2e6;font-size:.9rem;margin:.3rem 0'>{o.get('rationale','')}</div>"
                    f"{evidence_md(o.get('evidence', []))}",
                    unsafe_allow_html=True)
    else:
        st.caption("No opportunities generated yet.")

    # 4 ── Risks
    section_header(4, "Risk Monitor", f"{len(risks)} identified")
    if risks:
        for r in risks:
            with st.container(border=True):
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;gap:.8rem'>"
                    f"<span style='font-weight:650'>{r.get('title','')}</span>"
                    f"<span style='font-size:.8rem;color:#8aa0bd;white-space:nowrap'>"
                    f"{neutral_chip(r.get('category','—'))} sev {chip(r.get('severity'))} · "
                    f"conf {round(float(r.get('confidence',0)),2)}</span></div>"
                    f"<div style='color:#c4d2e6;font-size:.9rem;margin:.3rem 0'>{r.get('rationale','')}</div>"
                    f"{evidence_md(r.get('evidence', []))}",
                    unsafe_allow_html=True)
    else:
        st.caption("No risks generated yet.")

    # 5 ── Sentiment
    section_header(5, "Sentiment Analysis", f"engine: {sentiment.get('engine','—')}")
    if sentiment and sentiment.get("by_source"):
        sleft, sright = st.columns([2, 3])
        with sleft:
            st.plotly_chart(sentiment_gauge(overall), use_container_width=True)
        with sright:
            st.plotly_chart(sentiment_by_source_bar(sentiment["by_source"]),
                            use_container_width=True)
    else:
        st.caption("No sentiment data. Run the pipeline.")

    # 6 ── Recommendations
    section_header(6, "Strategic Recommendations", f"{len(recs)} actions")
    if recs:
        order = {"high": 0, "medium": 1, "low": 2}
        for r in sorted(recs, key=lambda x: order.get((x.get("priority") or "").lower(), 3)):
            with st.container(border=True):
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;gap:.8rem'>"
                    f"<span style='font-weight:650'>{r.get('action','')}</span>"
                    f"<span style='font-size:.8rem;color:#8aa0bd;white-space:nowrap'>"
                    f"priority {chip(r.get('priority'))} · risk {chip(r.get('risk_level'))}</span></div>"
                    f"<div style='color:#c4d2e6;font-size:.9rem;margin:.3rem 0'>"
                    f"<b>Expected impact:</b> {r.get('expected_impact','')}</div>"
                    f"{evidence_md(r.get('evidence', []))}",
                    unsafe_allow_html=True)
    else:
        st.caption("No recommendations generated yet.")

    # 7 ── CEO briefing
    section_header(7, "CEO Briefing", config.BRIEFING_QUESTION)
    if brief:
        st.markdown(f"**What happened** — {brief.get('what_happened','—')}")
        st.markdown(f"**Why it matters** — {brief.get('why_it_matters','—')}")
        st.success(f"**What to do next** — {brief.get('what_to_do_next','—')}")
    else:
        st.caption("No briefing generated yet.")

    # ── Live CEO Q&A
    section_header(8, "Live CEO Q&A", "grounded in the corpus · runs your local model")
    st.caption("Ask a strategic question. The agent retrieves evidence from the "
               "indexed corpus and answers with your local Ollama model.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    examples = [
        f"What are {config.COMPANY}'s biggest risks right now?",
        f"Where should {config.COMPANY} invest next?",
        f"How does {config.COMPANY} compare to its competitors?",
    ]
    ex_cols = st.columns(len(examples))
    clicked = None
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, use_container_width=True):
            clicked = ex

    typed = st.chat_input("Ask the AI CEO…")
    question = clicked or typed

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and reasoning…"):
                try:
                    import agents
                    answer = agents.chat(question)
                except Exception as e:
                    answer = (f"Could not run the agent: {e}\n\n"
                              "Make sure the index is built (`python pipeline.py`) and "
                              "Ollama is running (`ollama serve`).")
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
