"""
╔══════════════════════════════════════════════╗
║          SpamShield AI  v2.0                 ║
║   Hackathon-grade Spam Detection Platform    ║
╚══════════════════════════════════════════════╝
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time, io, datetime

from model import train_model, predict_message, analyze_bulk
from utils import highlight_text, risk_color, confidence_label, spam_type_emoji, format_bulk_summary

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SpamShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return train_model()

with st.spinner("🔧 Initialising SpamShield AI engine…"):
    model, vectorizer, accuracy = load_model()

# ─────────────────────────────────────────────
# THEME (dark / light via session state)
# ─────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = []
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

dark = st.session_state.dark_mode

# ─────────────────────────────────────────────
# CSS — Glassmorphism · Gradient · Animations
# ─────────────────────────────────────────────
BG         = "#0a0f1e" if dark else "#f0f4ff"
GLASS      = "rgba(255,255,255,0.06)" if dark else "rgba(255,255,255,0.72)"
GLASS_BD   = "rgba(255,255,255,0.15)" if dark else "rgba(200,210,255,0.6)"
TEXT_PRI   = "#e8eaf6" if dark else "#1a1a2e"
TEXT_SEC   = "#90a4ae" if dark else "#546e7a"
ACCENT     = "#00e5ff"
ACCENT2    = "#7c4dff"
CARD_BG    = "rgba(255,255,255,0.04)" if dark else "rgba(255,255,255,0.9)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
    color: {TEXT_PRI};
}}
.stApp {{
    background: {BG};
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,229,255,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(124,77,255,0.10) 0%, transparent 60%);
    min-height: 100vh;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {"rgba(10,15,30,0.95)" if dark else "rgba(240,244,255,0.98)"};
    border-right: 1px solid {GLASS_BD};
    backdrop-filter: blur(20px);
}}

/* ── Glass Card ── */
.glass-card {{
    background: {GLASS};
    border: 1px solid {GLASS_BD};
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.glass-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,229,255,0.12);
}}

/* ── Neon Metric ── */
.metric-box {{
    background: {CARD_BG};
    border: 1px solid {GLASS_BD};
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(12px);
}}
.metric-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}}
.metric-label {{
    font-size: 0.75rem;
    color: {TEXT_SEC};
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* ── Logo / Title ── */
.brand-title {{
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT2} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.1;
}}
.brand-sub {{
    font-size: 0.85rem;
    color: {TEXT_SEC};
    margin-top: 0.2rem;
    letter-spacing: 0.05em;
}}

/* ── Chat Bubbles ── */
.chat-wrap {{ display: flex; flex-direction: column; gap: 0.75rem; margin-bottom: 1rem; }}
.msg-user {{
    align-self: flex-end;
    background: linear-gradient(135deg, {ACCENT2}33, {ACCENT2}22);
    border: 1px solid {ACCENT2}55;
    border-radius: 18px 18px 4px 18px;
    padding: 0.8rem 1.1rem;
    max-width: 72%;
    font-size: 0.92rem;
    animation: fadeUp 0.3s ease;
}}
.msg-bot {{
    align-self: flex-start;
    background: {GLASS};
    border: 1px solid {GLASS_BD};
    border-radius: 4px 18px 18px 18px;
    padding: 0.8rem 1.1rem;
    max-width: 85%;
    font-size: 0.9rem;
    animation: fadeUp 0.3s ease;
}}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Highlighted words ── */
mark.highlight {{
    background: linear-gradient(135deg, #ff2d5544, #ff950044);
    color: #ff9500;
    border-radius: 4px;
    padding: 1px 3px;
    font-weight: 600;
}}

/* ── Risk Badge ── */
.risk-badge {{
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

/* ── Section Header ── */
.section-header {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {GLASS_BD};
}}

/* ── Animated Confidence Bar ── */
.conf-bar-wrap {{ margin: 0.5rem 0; }}
.conf-bar-label {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: {TEXT_SEC};
    margin-bottom: 0.2rem;
}}
.conf-bar-track {{
    height: 8px;
    background: {"rgba(255,255,255,0.08)" if dark else "rgba(0,0,0,0.08)"};
    border-radius: 999px;
    overflow: hidden;
}}
.conf-bar-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 0.55rem 1.5rem !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    letter-spacing: 0.02em !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(0,229,255,0.35) !important;
}}

/* ── Text Input ── */
.stTextArea textarea, .stTextInput input {{
    background: {"rgba(255,255,255,0.05)" if dark else "rgba(255,255,255,0.9)"} !important;
    border: 1px solid {GLASS_BD} !important;
    border-radius: 12px !important;
    color: {TEXT_PRI} !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,0.15) !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {GLASS} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid {GLASS_BD} !important;
    gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    color: {TEXT_SEC} !important;
    font-weight: 500 !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {ACCENT}22, {ACCENT2}22) !important;
    color: {TEXT_PRI} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* ── DataTable ── */
.stDataFrame {{ border-radius: 12px; overflow: hidden; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {GLASS_BD}; border-radius: 3px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="padding:1rem 0 1.5rem;">
        <p class="brand-title">🛡️ SpamShield</p>
        <p class="brand-sub">AI-Powered Threat Detection</p>
    </div>
    """, unsafe_allow_html=True)

    # Theme toggle
    toggle_label = "☀️ Light Mode" if dark else "🌙 Dark Mode"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.dark_mode = not dark
        st.rerun()

    st.divider()

    # Stats
    st.markdown('<p class="section-header">System Stats</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{accuracy}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{st.session_state.analysis_count}</div>
            <div class="metric-label">Analysed</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-header">Detection Signals</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; color:{TEXT_SEC}; line-height:1.8;">
    🎣 &nbsp;Phishing Intent<br>
    ⏰ &nbsp;Urgency Manipulation<br>
    📢 &nbsp;Promotional Spam<br>
    🔤 &nbsp;TF-IDF Risk Words<br>
    📊 &nbsp;Confidence Scoring<br>
    🗂️ &nbsp;Bulk CSV Analysis
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown(f'<p style="font-size:0.7rem; color:{TEXT_SEC}; text-align:center;">SpamShield AI v2.0 · Built with ❤️</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding: 1rem 0 0.5rem;">
    <h1 class="brand-title" style="font-size:2.8rem;">🛡️ SpamShield AI</h1>
    <p style="color:{TEXT_SEC}; font-size:1rem; margin-top:0.3rem;">
        Real-time intelligent spam & threat detection · Explainable · Privacy-first
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬  Chat Analyser", "📂  Bulk Upload", "📊  Analytics Dashboard"])

# ══════════════════════════════════════════════
#  TAB 1 — CHAT INTERFACE
# ══════════════════════════════════════════════
with tab1:
    col_chat, col_result = st.columns([1.1, 0.9], gap="large")

    # ── Left: Chat window ──
    with col_chat:
        st.markdown(f'<p class="section-header">Message Analyser</p>', unsafe_allow_html=True)

        # Chat history
        if st.session_state.chat_history:
            st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
            for entry in st.session_state.chat_history[-12:]:   # last 12 turns
                st.markdown(f'<div class="msg-user">👤 {entry["message"]}</div>', unsafe_allow_html=True)
                verdict_icon = "🚫" if entry["result"]["prediction"] == 1 else "✅"
                sp = entry["result"]["spam_prob"]
                rc = risk_color(entry["result"]["risk_level"])
                st.markdown(f"""
                <div class="msg-bot">
                    {verdict_icon} <strong>{"SPAM" if entry["result"]["prediction"]==1 else "SAFE"}</strong>
                    &nbsp;&nbsp;
                    <span class="risk-badge" style="background:{rc}22; color:{rc}; border:1px solid {rc}55;">
                        {entry["result"]["risk_level"]} Risk
                    </span>
                    <br><span style="font-size:0.78rem; color:{TEXT_SEC}; font-family:'JetBrains Mono',monospace;">
                        Spam prob: {sp}% · {entry["result"]["spam_type"]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align:center; padding:2rem 1rem; color:{TEXT_SEC}; font-size:0.9rem;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">💬</div>
                Type a message below to begin analysis
            </div>
            """, unsafe_allow_html=True)

        # Input
        user_text = st.text_area(
            "Message to analyse",
            placeholder="Paste any message — email, SMS, notification…",
            height=110,
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyse_btn = st.button("🔍 Analyse", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # ── Right: Results panel ──
    with col_result:
        st.markdown(f'<p class="section-header">Analysis Report</p>', unsafe_allow_html=True)

        if analyse_btn and user_text.strip():
            with st.spinner("Analysing…"):
                time.sleep(0.4)   # UX breathing room
                result = predict_message(user_text, model, vectorizer)

            st.session_state.chat_history.append({"message": user_text, "result": result})
            st.session_state.analysis_count += 1
            st.session_state.last_result = result
            st.session_state.last_text   = user_text
            st.rerun()

        if "last_result" in st.session_state:
            result    = st.session_state.last_result
            last_text = st.session_state.last_text
            rc        = risk_color(result["risk_level"])
            is_spam   = result["prediction"] == 1
            emoji     = spam_type_emoji(result["spam_type"])

            # ── Verdict card ──
            verdict_color = "#ff2d55" if is_spam else "#34c759"
            st.markdown(f"""
            <div class="glass-card" style="border-color:{verdict_color}44; text-align:center; padding:1.5rem;">
                <div style="font-size:3rem; margin-bottom:0.4rem;">{"🚫" if is_spam else "✅"}</div>
                <div style="font-size:1.5rem; font-weight:700; color:{verdict_color};">
                    {"SPAM DETECTED" if is_spam else "SAFE MESSAGE"}
                </div>
                <div style="margin-top:0.5rem;">
                    <span class="risk-badge" style="background:{rc}22; color:{rc}; border:1px solid {rc}55; font-size:0.82rem;">
                        {result["risk_level"]} Risk
                    </span>
                    &nbsp;
                    <span class="risk-badge" style="background:{GLASS}; color:{TEXT_SEC}; border:1px solid {GLASS_BD}; font-size:0.82rem;">
                        {emoji} {result["spam_type"]}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence bars ──
            st.markdown(f'<p class="section-header">Confidence Score</p>', unsafe_allow_html=True)
            sp = result["spam_prob"]
            hp = result["ham_prob"]
            st.markdown(f"""
            <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>🚫 Spam</span><span>{sp}%</span></div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill" style="width:{sp}%; background:linear-gradient(90deg,#ff2d55,#ff9500);"></div>
                </div>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar-label"><span>✅ Safe</span><span>{hp}%</span></div>
                <div class="conf-bar-track">
                    <div class="conf-bar-fill" style="width:{hp}%; background:linear-gradient(90deg,#34c759,#00e5ff);"></div>
                </div>
            </div>
            <p style="font-size:0.76rem; color:{TEXT_SEC}; margin-top:0.5rem;">
                {confidence_label(sp if is_spam else hp)}
            </p>
            """, unsafe_allow_html=True)

            # ── Plotly donut ──
            fig = go.Figure(go.Pie(
                values=[sp, hp],
                labels=["Spam", "Safe"],
                hole=0.65,
                marker_colors=["#ff2d55", "#34c759"],
                textinfo="none",
            ))
            fig.update_layout(
                showlegend=True,
                margin=dict(t=10, b=10, l=10, r=10),
                height=160,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT_PRI,
                legend=dict(font=dict(size=11)),
            )
            fig.add_annotation(
                text=f"<b>{sp}%</b><br>Spam",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color=TEXT_PRI)
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # ── Signals detected ──
            signals = []
            if result["urgency_signals"]:
                signals.append(("⏰ Urgency", result["urgency_signals"], "#ff9500"))
            if result["phishing_signals"]:
                signals.append(("🎣 Phishing", result["phishing_signals"], "#ff2d55"))
            if result["promo_signals"]:
                signals.append(("📢 Promo", result["promo_signals"], "#7c4dff"))

            if signals:
                st.markdown(f'<p class="section-header">Signals Detected</p>', unsafe_allow_html=True)
                for label, words, clr in signals:
                    tags = " ".join(
                        f'<span style="background:{clr}22; color:{clr}; border:1px solid {clr}44; border-radius:6px; padding:2px 8px; font-size:0.74rem; margin:2px; display:inline-block;">{w}</span>'
                        for w in words[:5]
                    )
                    st.markdown(f'<div style="margin-bottom:0.5rem;"><strong style="font-size:0.8rem;">{label}</strong><br>{tags}</div>', unsafe_allow_html=True)

            # ── Highlighted text ──
            st.markdown(f'<p class="section-header">Highlighted Message</p>', unsafe_allow_html=True)
            highlighted = highlight_text(last_text, result["top_words"])
            st.markdown(f"""
            <div class="glass-card" style="font-size:0.88rem; line-height:1.7; font-family:'Space Grotesk',sans-serif;">
                {highlighted}
            </div>
            """, unsafe_allow_html=True)

            # ── Explanation ──
            st.markdown(f'<p class="section-header">Why this verdict?</p>', unsafe_allow_html=True)
            for reason in result["explanation"].split(" | "):
                st.markdown(f"""
                <div style="display:flex; align-items:start; gap:0.5rem; margin-bottom:0.4rem; font-size:0.83rem; color:{TEXT_SEC};">
                    <span style="color:{ACCENT}; flex-shrink:0;">›</span> {reason}
                </div>
                """, unsafe_allow_html=True)

        else:
            if not analyse_btn:
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:center;
                            height:300px; color:{TEXT_SEC}; font-size:0.9rem; text-align:center;">
                    <div>
                        <div style="font-size:3rem; margin-bottom:0.5rem;">🔬</div>
                        Analysis report will<br>appear here
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 2 — BULK UPLOAD
# ══════════════════════════════════════════════
with tab2:
    st.markdown(f'<p class="section-header">Bulk Message Analysis</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT_SEC}; font-size:0.88rem;">Upload a CSV with a <code>message</code> column, or paste messages (one per line).</p>', unsafe_allow_html=True)

    col_up, col_paste = st.columns(2)

    with col_up:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    with col_paste:
        pasted = st.text_area("Or paste messages (one per line)", height=120, label_visibility="collapsed",
                               placeholder="Message 1\nMessage 2\n…")

    run_bulk = st.button("🚀 Run Bulk Analysis", use_container_width=False)

    if run_bulk:
        messages = []
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if "message" in df_up.columns:
                messages = df_up["message"].astype(str).tolist()
            else:
                st.error("CSV must have a 'message' column.")
        elif pasted.strip():
            messages = [m for m in pasted.strip().split("\n") if m.strip()]

        if messages:
            with st.spinner(f"Analysing {len(messages)} messages…"):
                bulk_res = analyze_bulk(messages, model, vectorizer)
                st.session_state.bulk_results = list(zip(messages, bulk_res))
                st.session_state.analysis_count += len(messages)

    if st.session_state.bulk_results:
        pairs = st.session_state.bulk_results
        summary = format_bulk_summary([r for _, r in pairs])

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl in zip(
            [m1, m2, m3, m4],
            [summary["total"], summary["spam"], summary["ham"], f"{summary['spam_rate']}%"],
            ["Total", "Spam", "Safe", "Spam Rate"]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Type breakdown chart
        if summary["type_breakdown"]:
            bd = summary["type_breakdown"]
            fig2 = px.bar(
                x=list(bd.keys()), y=list(bd.values()),
                labels={"x": "Spam Type", "y": "Count"},
                color=list(bd.keys()),
                color_discrete_sequence=["#ff2d55","#7c4dff","#ff9500","#00e5ff","#34c759"],
            )
            fig2.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT_PRI,
                margin=dict(t=20, b=20, l=20, r=20),
                height=200,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Table
        table_data = []
        for msg, res in pairs:
            icon = "🚫" if res["prediction"] == 1 else "✅"
            table_data.append({
                "Message":    msg[:60] + ("…" if len(msg) > 60 else ""),
                "Verdict":    f"{icon} {'Spam' if res['prediction']==1 else 'Safe'}",
                "Spam %":     f"{res['spam_prob']}%",
                "Risk":       res["risk_level"],
                "Type":       res["spam_type"],
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = df_table.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_bytes,
            file_name=f"spamshield_results_{datetime.date.today()}.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════
#  TAB 3 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
with tab3:
    st.markdown(f'<p class="section-header">Session Analytics</p>', unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown(f"""
        <div style="text-align:center; padding:3rem 1rem; color:{TEXT_SEC};">
            <div style="font-size:3rem; margin-bottom:0.5rem;">📊</div>
            Analyse some messages first to see your session dashboard
        </div>
        """, unsafe_allow_html=True)
    else:
        history = st.session_state.chat_history
        results = [h["result"] for h in history]

        # KPIs
        spam_count = sum(1 for r in results if r["prediction"] == 1)
        safe_count = len(results) - spam_count
        avg_prob   = round(sum(r["spam_prob"] for r in results) / len(results), 1)

        k1, k2, k3, k4 = st.columns(4)
        for col, val, lbl in zip(
            [k1, k2, k3, k4],
            [len(results), spam_count, safe_count, f"{avg_prob}%"],
            ["Messages", "Spam", "Safe", "Avg Spam Prob"]
        ):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        # Pie
        with c1:
            fig3 = go.Figure(go.Pie(
                labels=["Spam", "Safe"],
                values=[spam_count, safe_count],
                hole=0.55,
                marker_colors=["#ff2d55", "#34c759"],
            ))
            fig3.update_layout(
                title="Spam vs Safe",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT_PRI,
                margin=dict(t=40, b=10, l=10, r=10),
                height=250,
            )
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        # Line: spam prob over time
        with c2:
            probs = [r["spam_prob"] for r in results]
            fig4 = go.Figure(go.Scatter(
                y=probs, mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                marker=dict(color=ACCENT2, size=7),
                fill="tozeroy",
                fillcolor=f"rgba(0,229,255,0.08)",
            ))
            fig4.update_layout(
                title="Spam Probability Over Time",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT_PRI,
                margin=dict(t=40, b=20, l=20, r=20),
                height=250,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Message #"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Spam %", range=[0,100]),
            )
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        # Risk level breakdown
        risk_counts = {}
        for r in results:
            risk_counts[r["risk_level"]] = risk_counts.get(r["risk_level"], 0) + 1
        if risk_counts:
            fig5 = px.bar(
                x=list(risk_counts.keys()), y=list(risk_counts.values()),
                color=list(risk_counts.keys()),
                color_discrete_map={"Critical":"#ff2d55","High":"#ff9500","Medium":"#ffcc00","Low":"#34c759"},
                title="Risk Level Distribution",
            )
            fig5.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT_PRI,
                margin=dict(t=40, b=20, l=20, r=20),
                height=230,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})

        # Top trigger words
        all_words = {}
        for r in results:
            for w, s in r["top_words"]:
                all_words[w] = all_words.get(w, 0) + s
        if all_words:
            top_n = sorted(all_words.items(), key=lambda x: x[1], reverse=True)[:10]
            fig6 = px.bar(
                x=[w for w,_ in top_n], y=[s for _,s in top_n],
                title="Top Trigger Words (TF-IDF Score Sum)",
                color=[s for _,s in top_n],
                color_continuous_scale=["#00e5ff","#7c4dff","#ff2d55"],
            )
            fig6.update_layout(
                showlegend=False, coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT_PRI,
                margin=dict(t=40, b=20, l=20, r=20),
                height=230,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig6, use_container_width=True, config={"displayModeBar": False})

        # Session history table
        st.markdown(f'<p class="section-header" style="margin-top:1rem;">Full Session History</p>', unsafe_allow_html=True)
        hist_rows = []
        for i, h in enumerate(history, 1):
            r = h["result"]
            hist_rows.append({
                "#": i,
                "Message": h["message"][:50] + ("…" if len(h["message"]) > 50 else ""),
                "Verdict": "🚫 Spam" if r["prediction"] == 1 else "✅ Safe",
                "Spam %": f"{r['spam_prob']}%",
                "Risk": r["risk_level"],
                "Type": r["spam_type"],
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)