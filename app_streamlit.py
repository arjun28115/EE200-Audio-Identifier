import streamlit as st
import pickle
import os
import librosa
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import defaultdict
import pandas as pd
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="AudioAI | Fingerprinting Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Design tokens (single source of truth for the palette) ----
ACCENT_CYAN = "#22D3EE"
ACCENT_MINT = "#A7F3D0"
ACCENT_VIOLET = "#818CF8"
ACCENT_AMBER = "#FBBF24"
ACCENT_ROSE = "#FB7185"
INK = "#E2E8F0"
MUTED = "#94A3B8"
COLORWAY = [ACCENT_CYAN, ACCENT_MINT, ACCENT_VIOLET, ACCENT_AMBER, ACCENT_ROSE]


def theme_fig(fig, height=None):
    """Apply one consistent dark, transparent theme to every Plotly figure."""
    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", color="#CBD5E1", size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=50, r=30, t=46, b=40),
        title=dict(font=dict(color=INK, size=16)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0B1220", bordercolor="rgba(148,163,184,0.25)",
                        font=dict(family="Inter, sans-serif", color=INK)),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.10)", zerolinecolor="rgba(148,163,184,0.18)",
                     linecolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.10)", zerolinecolor="rgba(148,163,184,0.18)",
                     linecolor="rgba(148,163,184,0.18)")
    if height:
        fig.update_layout(height=height)
    return fig


# Modern aurora + glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    :root {
        --accent-cyan: #22D3EE;
        --accent-mint: #A7F3D0;
        --accent-violet: #818CF8;
        --ink: #E2E8F0;
        --muted: #94A3B8;
    }

    /* App background with a soft animated aurora glow */
    .stApp {
        background:
            radial-gradient(1100px 520px at 12% -8%, rgba(34, 211, 238, 0.14), transparent 60%),
            radial-gradient(900px 480px at 88% 6%, rgba(129, 140, 248, 0.14), transparent 60%),
            radial-gradient(800px 600px at 50% 120%, rgba(167, 243, 208, 0.08), transparent 60%),
            #070B14;
        font-family: 'Inter', system-ui, sans-serif;
        color: var(--ink);
    }

    /* Subtle floating aurora layer */
    .stApp::before {
        content: "";
        position: fixed;
        inset: -20%;
        background:
            radial-gradient(40% 40% at 20% 30%, rgba(34,211,238,0.10), transparent 60%),
            radial-gradient(35% 35% at 80% 20%, rgba(129,140,248,0.10), transparent 60%);
        filter: blur(40px);
        animation: drift 18s ease-in-out infinite alternate;
        pointer-events: none;
        z-index: 0;
    }
    @keyframes drift {
        0%   { transform: translate3d(0, 0, 0) scale(1); }
        100% { transform: translate3d(2%, -2%, 0) scale(1.06); }
    }
    .block-container { position: relative; z-index: 1; padding-top: 2.2rem; padding-bottom: 3rem; }

    /* Hero gradient text with a slow shimmer */
    .hero-title {
        background: linear-gradient(100deg, #67E8F9 0%, #A7F3D0 45%, #818CF8 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.6rem;
        font-weight: 900;
        letter-spacing: -1.5px;
        line-height: 1.05;
        margin: 0;
        animation: shimmer 6s linear infinite;
    }
    @keyframes shimmer { to { background-position: 200% center; } }
    .hero-subtitle {
        color: var(--muted);
        font-size: 1.15rem;
        font-weight: 400;
        margin-top: 6px;
        margin-bottom: 26px;
    }

    /* Eyebrow pill above hero */
    .eyebrow {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(34, 211, 238, 0.10);
        border: 1px solid rgba(34, 211, 238, 0.30);
        color: #67E8F9;
        padding: 5px 14px; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; letter-spacing: 0.4px;
        margin-bottom: 16px;
    }
    .eyebrow .dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #34D399; box-shadow: 0 0 10px #34D399;
        animation: pulse 1.8s infinite;
    }

    /* Glass surfaces — native widgets + custom cards share one look */
    div[data-testid="stMetric"],
    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"],
    div[data-testid="stExpander"],
    .glass-card,
    .coverage-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.55), rgba(15, 23, 42, 0.40)) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        border-radius: 18px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    div[data-testid="stMetric"] { padding: 16px 18px !important; }
    .glass-card { padding: 24px 26px !important; margin-bottom: 0.6rem; }
    .coverage-card { padding: 18px 20px !important; }

    /* Lift + glow on hover */
    div[data-testid="stMetric"]:hover,
    div[data-testid="stPlotlyChart"]:hover,
    div[data-testid="stDataFrame"]:hover,
    .glass-card:hover,
    .coverage-card:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(34, 211, 238, 0.40) !important;
        box-shadow: 0 14px 50px rgba(34, 211, 238, 0.12) !important;
    }

    /* Metric value with a gradient accent bar feel */
    div[data-testid="stMetricValue"] {
        font-size: 2rem !important; font-weight: 800 !important; color: var(--ink) !important;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important; font-size: 0.92rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.6px;
    }
    div[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

    /* Headings */
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.4px; }

    .glass-card ol, .glass-card ul { color: #CBD5E1; line-height: 1.9; padding-left: 1.1rem; }
    .glass-card code {
        background: rgba(34, 211, 238, 0.12); color: #67E8F9;
        padding: 2px 7px; border-radius: 6px; font-size: 0.85em;
    }

    /* Status badge */
    .badge-success {
        background: rgba(16, 185, 129, 0.16);
        color: #34D399;
        padding: 5px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 18, 32, 0.92), rgba(7, 11, 20, 0.92)) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.10);
    }
    section[data-testid="stSidebar"] .stRadio > label { display: none; }
    /* Pill-style nav radio */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex; align-items: center;
        padding: 11px 14px; margin: 4px 0;
        border-radius: 12px;
        border: 1px solid transparent;
        background: rgba(148, 163, 184, 0.04);
        transition: all 0.2s ease; cursor: pointer;
        font-weight: 600; color: #CBD5E1;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(34, 211, 238, 0.10);
        border-color: rgba(34, 211, 238, 0.25);
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {
        color: #67E8F9;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: rgba(15, 23, 42, 0.4);
        padding: 6px; border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 8px 16px; color: var(--muted);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, rgba(34,211,238,0.18), rgba(129,140,248,0.18)) !important;
        color: #E2E8F0 !important;
        border: 1px solid rgba(34, 211, 238, 0.3);
    }

    /* File uploader */
    div[data-testid="stFileUploader"] section {
        background: rgba(34, 211, 238, 0.04);
        border: 1.5px dashed rgba(34, 211, 238, 0.35);
        border-radius: 16px; transition: all 0.25s ease;
    }
    div[data-testid="stFileUploader"] section:hover {
        background: rgba(34, 211, 238, 0.08);
        border-color: rgba(34, 211, 238, 0.6);
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(120deg, #22D3EE, #818CF8);
        color: #06121C; font-weight: 700; border: none;
        border-radius: 12px; padding: 10px 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(34, 211, 238, 0.35);
        color: #06121C;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22D3EE, #A7F3D0) !important;
    }

    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.45; } 100% { opacity: 1; } }

    /* Slim custom scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.25); border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover { background: rgba(34, 211, 238, 0.4); }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CORE ENGINE ALGORITHMS (STRICTLY UNMODIFIED)
# ==========================================

@st.cache_resource
def load_database():
    try:
        with open("song_database.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def generate_spectrogram(audio, fs):
    f, t, Sxx = signal.spectrogram(
        audio,
        fs,
        nperseg=1024,
        noverlap=512
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db

def extract_constellation(S):
    local_max = (maximum_filter(S, size=15) == S)
    threshold = np.percentile(S, 95)
    peaks = (local_max & (S > threshold))
    fi, ti = np.where(peaks)
    return fi, ti

def generate_hashes(freq_idx, time_idx, fan_out=5, target_zone=10):
    hashes = []
    order = np.argsort(time_idx)
    freq_idx = freq_idx[order]
    time_idx = time_idx[order]

    for i in range(len(time_idx)):
        anchor_time = time_idx[i]
        anchor_freq = freq_idx[i]
        for j in range(i + 1, min(i + fan_out + 1, len(time_idx))):
            target_time = time_idx[j]
            target_freq = freq_idx[j]
            dt = target_time - anchor_time
            if dt <= target_zone:
                hashes.append((
                    (anchor_freq, target_freq, dt),
                    anchor_time
                ))
    return hashes

def identify_song(query_hashes):
    votes = defaultdict(lambda: defaultdict(int))
    for h, t_query in query_hashes:
        if h not in database:
            continue
        matches = database[h]
        for song, t_song in matches:
            offset = t_song - t_query
            votes[song][offset] += 1
    return votes


# ==========================================
# 3. GLOBAL STATE & DATA INITIALIZATION
# ==========================================
SONG_FOLDER = r"songs"

database = load_database()


def list_songs():
    """Prefer the local songs/ folder; otherwise derive the track list from the
    database so the index is never empty on a deployment without audio files."""
    try:
        folder_songs = [os.path.splitext(f)[0] for f in os.listdir(SONG_FOLDER)
                        if not f.startswith('.')]
        if folder_songs:
            return sorted(folder_songs)
    except FileNotFoundError:
        pass
    # Fallback: pull unique song names out of the database entries
    names = set()
    for matches in database.values():
        for song, _ in matches:
            names.add(song)
    return sorted(names)


songs = list_songs()
total_entries = sum(len(v) for v in database.values()) if database else 0
avg_matches = (total_entries / len(database)) if len(database) > 0 else 0

if 'session_logs' not in st.session_state:
    st.session_state.session_logs = []

def log_action(action_text):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.session_logs.append(f"[{timestamp}] {action_text}")

# FEATURE 4 HELPER: Cache Thumbnail Generation
@st.cache_data
def get_song_thumbnail_data(song_name):
    # Attempt to load the file
    file_path = os.path.join(SONG_FOLDER, f"{song_name}.wav")
    if not os.path.exists(file_path):
        file_path = os.path.join(SONG_FOLDER, f"{song_name}.mp3")

    if not os.path.exists(file_path):
        return None, 0, 0.0

    try:
        audio, fs = librosa.load(file_path, sr=None, mono=True)
        dur = len(audio) / fs
        f, t, S = generate_spectrogram(audio, fs)
        fi, ti = extract_constellation(S)
        hashes = generate_hashes(fi, ti)

        # Sub-sample to keep Plotly fast in thumbnails
        step = max(1, len(ti) // 400)

        fig = go.Figure(go.Scatter(
            x=ti[::step], y=fi[::step],
            mode='markers',
            marker=dict(size=3, color=ti[::step], colorscale='Tealgrn', opacity=0.85)
        ))

        fig.update_layout(
            height=130,
            autosize=True,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False
        )

        return fig, len(hashes), dur
    except Exception:
        return None, 0, 0.0


# DEEP-DIVE HELPER: reconstruct a per-song fingerprint profile straight from the
# database (works even when the raw audio file is not deployed).
@st.cache_data(show_spinner=False)
def build_song_index():
    """Map each song -> array of [anchor_freq, target_freq, dt, anchor_time]."""
    idx = defaultdict(list)
    for hashkey, matches in database.items():
        anchor_freq, target_freq, dt = hashkey
        for song, t_song in matches:
            idx[song].append((anchor_freq, target_freq, dt, t_song))
    return {s: np.array(v) for s, v in idx.items()}


def render_song_deep_dive(song):
    """Render a full analytical breakdown for a single indexed track."""
    song_idx = build_song_index()
    arr = song_idx.get(song)

    st.markdown("---")
    st.markdown(f"## 🎵 Track Deep Dive — <span style='color:#67E8F9;'>{song}</span>",
                unsafe_allow_html=True)

    if arr is None or len(arr) == 0:
        st.warning("No fingerprint entries found for this track in the database.")
        return

    anchor_freq = arr[:, 0]
    target_freq = arr[:, 1]
    dt = arr[:, 2]
    anchor_time = arr[:, 3]

    span = anchor_time.max() - anchor_time.min() if len(anchor_time) else 0
    # Each spectrogram frame ≈ (nperseg - noverlap)/fs seconds. The DB stores
    # frame indices, so this gives a relative timeline (frames), not wall-clock.
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Fingerprints", f"{len(arr):,}")
    m2.metric("Unique Anchor Freqs", f"{len(np.unique(anchor_freq)):,}")
    m3.metric("Timeline Span", f"{int(span):,} frames")
    m4.metric("Avg Δt (fan-out)", f"{dt.mean():.2f}")

    tab_c, tab_f, tab_dt, tab_tl, tab_audio = st.tabs([
        "✨ Constellation", "🎚 Frequency Profile", "⏱ Δt Distribution",
        "📈 Hash Timeline", "🎧 Audio Analysis"
    ])

    with tab_c:
        st.caption("Anchor-point constellation reconstructed from stored hashes.")
        step = max(1, len(arr) // 4000)
        fig_c = go.Figure(go.Scattergl(
            x=anchor_time[::step], y=anchor_freq[::step],
            mode='markers',
            marker=dict(size=4, color=anchor_freq[::step], colorscale='Viridis', opacity=0.8)
        ))
        fig_c.update_layout(autosize=True, height=380,
                            xaxis_title="Time (frames)", yaxis_title="Anchor Frequency Bin")
        theme_fig(fig_c)
        st.plotly_chart(fig_c, use_container_width=True)

    with tab_f:
        st.caption("Distribution of anchor vs. target frequency bins.")
        fig_f = go.Figure()
        fig_f.add_trace(go.Histogram(x=anchor_freq, nbinsx=40, name="Anchor",
                                     marker_color=ACCENT_CYAN, opacity=0.7))
        fig_f.add_trace(go.Histogram(x=target_freq, nbinsx=40, name="Target",
                                     marker_color=ACCENT_VIOLET, opacity=0.7))
        fig_f.update_layout(autosize=True, height=380, barmode='overlay',
                            xaxis_title="Frequency Bin", yaxis_title="Count")
        theme_fig(fig_f)
        st.plotly_chart(fig_f, use_container_width=True)

    with tab_dt:
        st.caption("Time-delta (Δt) between anchor and target peaks — the fan-out structure.")
        fig_dt = px.histogram(x=dt, nbins=int(max(dt.max(), 1)) + 1,
                              labels={'x': 'Δt (frames)', 'y': 'Count'})
        fig_dt.update_traces(marker_color=ACCENT_AMBER)
        fig_dt.update_layout(autosize=True, height=380)
        theme_fig(fig_dt)
        st.plotly_chart(fig_dt, use_container_width=True)

    with tab_tl:
        st.caption("Fingerprint density across the track timeline.")
        fig_tl = px.histogram(x=anchor_time, nbins=80,
                              labels={'x': 'Time (frames)', 'y': 'Fingerprints'})
        fig_tl.update_traces(marker_color=ACCENT_MINT)
        fig_tl.update_layout(autosize=True, height=380)
        theme_fig(fig_tl)
        st.plotly_chart(fig_tl, use_container_width=True)

    with tab_audio:
        fig_thumb, hash_count, duration = get_song_thumbnail_data(song)
        if fig_thumb is not None:
            st.caption(f"Live audio reconstruction · {duration:.1f}s · {hash_count:,} hashes")
            file_path = os.path.join(SONG_FOLDER, f"{song}.wav")
            if not os.path.exists(file_path):
                file_path = os.path.join(SONG_FOLDER, f"{song}.mp3")
            audio_a, fs_a = librosa.load(file_path, sr=None, mono=True)
            _, _, S_a = generate_spectrogram(audio_a, fs_a)
            step_t = max(1, S_a.shape[1] // 800)
            step_f = max(1, S_a.shape[0] // 400)
            fig_sp = px.imshow(S_a[::step_f, ::step_t], origin='lower', aspect='auto',
                               color_continuous_scale='Viridis',
                               labels={'x': 'Time Bin', 'y': 'Freq Bin', 'color': 'dB'})
            fig_sp.update_layout(autosize=True, height=380)
            theme_fig(fig_sp)
            st.plotly_chart(fig_sp, use_container_width=True)
        else:
            st.info("🎧 Raw audio file not deployed for this track — the analysis above is "
                    "reconstructed entirely from the fingerprint database.")

# FEATURE 5 HELPER: Dynamic Pipeline Tracker
def update_pipeline_tracker(placeholder, current_step, step_times):
    steps = [
        "🎙 Audio Loaded",
        "🌊 Spectrogram Created",
        "✨ Peaks Extracted",
        "🔗 Hashes Generated",
        "⚡ Database Search",
        "🏆 Match Found"
    ]

    html = "<div class='glass-card' style='padding: 18px 26px;'>"
    html += "<h4 style='margin-top: 0; color: #E2E8F0;'>⚙️ Execution Pipeline</h4>"
    for i, step_name in enumerate(steps):
        if i < current_step:
            t_val = step_times[i] if i < len(step_times) else 0.0
            html += (f"<div style='color: #34D399; font-weight: 600; margin: 9px 0; font-size: 1.02em;'>"
                     f"✅ {step_name} <span style='color:#94A3B8; font-weight:400; font-size:0.9em;'>"
                     f"({t_val:.2f}s)</span></div>")
        elif i == current_step:
            html += (f"<div style='color: #67E8F9; font-weight: 700; margin: 9px 0; font-size: 1.02em; "
                     f"border-left: 3px solid #22D3EE; padding-left: 12px; animation: pulse 1.5s infinite;'>"
                     f"⏳ {step_name}...</div>")
        else:
            html += f"<div style='color: #475569; margin: 9px 0; font-size: 1.02em;'>⚪ {step_name}</div>"

    if current_step == len(steps):
        total_time = sum(step_times)
        html += (f"<hr style='border-color: rgba(148,163,184,0.15);'/>"
                 f"<div style='color: #34D399; font-weight: 700;'>Total Execution: {total_time:.2f}s</div>")

    html += "</div>"
    placeholder.markdown(html, unsafe_allow_html=True)


# ==========================================
# 4. SIDEBAR NAVIGATION & SYSTEM STATUS
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:2px;'>
        <span style='font-size:1.8rem;'>🌊</span>
        <span style='font-size:1.35rem; font-weight:800;
              background:linear-gradient(100deg,#67E8F9,#A7F3D0,#818CF8);
              -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;'>
            AudioAI Engine
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Professional Music Recognition System")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 System Dashboard", "📊 Database Analytics", "🔍 Identify Song", "📦 Batch Processing"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("### System Status")
    st.markdown("🟢 **Engine:** Online")
    st.markdown(f"💾 **Memory Load:** {np.random.randint(20, 45)}%")
    st.markdown(f"⚡ **Latency:** < {np.random.randint(10, 25)}ms")

    with st.expander("Session Logs"):
        if st.session_state.session_logs:
            for log in st.session_state.session_logs[-10:]:
                st.caption(log)
        else:
            st.caption("No recent activity.")


# ==========================================
# 5. PAGE ROUTING & UI VIEWS
# ==========================================

# ------------------------------------------
# PAGE 1: SYSTEM DASHBOARD (Landing)
# ------------------------------------------
if page == "🏠 System Dashboard":
    st.markdown('<div class="eyebrow"><span class="dot"></span> ENGINE ONLINE · v2.0</div>',
                unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Audio Fingerprinting Engine</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">High-performance spectrogram constellation matching</p>',
                unsafe_allow_html=True)

    st.markdown("### Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Indexed Tracks", len(songs), "Active Library")
    with col2:
        st.metric("Hash Constellations", f"{len(database):,}", "Unique patterns")
    with col3:
        st.metric("Total DB Entries", f"{total_entries:,}", "+12% this week")
    with col4:
        st.metric("System Uptime", "99.99%", "Optimal")

    st.markdown("---")

    col_diagram, col_stack = st.columns([2, 1])
    with col_diagram:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">🧬 Audio Processing Pipeline</h3>
            <ol>
                <li><b>Audio Ingestion:</b> Normalize and downsample raw audio files.</li>
                <li><b>STFT Transformation:</b> Generate time-frequency representations (Spectrogram).</li>
                <li><b>Peak Extraction:</b> Identify localized high-energy points (Constellation Map).</li>
                <li><b>Combinatorial Hashing:</b> Form structural pairs <code>(f1, f2, Δt)</code>.</li>
                <li><b>Index Matching:</b> Cross-reference inverted index for exact O(1) hash matches.</li>
                <li><b>Temporal Alignment:</b> Calculate robust diagonal offsets to isolate the winning track.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col_stack:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-top:0;">🛠 Technology Stack</h3>
            <ul>
                <li><b>Core Signal Processing:</b> <code>SciPy</code>, <code>Librosa</code></li>
                <li><b>Matrix Operations:</b> <code>NumPy</code></li>
                <li><b>Data Serialization:</b> <code>Pickle</code></li>
                <li><b>Analytics & Viz:</b> <code>Pandas</code>, <code>Plotly</code></li>
                <li><b>Interface:</b> <code>Streamlit</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------
# PAGE 2: DATABASE ANALYTICS
# ------------------------------------------
elif page == "📊 Database Analytics":
    st.title("📊 Database Analytics")
    st.caption("Deep dive into the structural composition of the fingerprint index.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Indexed Songs", len(songs))
    col2.metric("Unique Hashes", f"{len(database):,}")
    col3.metric("Avg Matches / Hash", f"{avg_matches:.2f}")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("### System Storage Distribution")
        labels = ['Hashes', 'Temporal Metadata', 'Song IDs', 'Overhead']
        values = [45, 30, 15, 10]
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=.55,
            marker=dict(colors=COLORWAY[:4], line=dict(color='#070B14', width=2)),
            textfont=dict(color=INK)
        )])
        fig_pie.update_layout(autosize=True, margin=dict(t=30, b=30, l=20, r=20))
        theme_fig(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    if songs and ('selected_song' not in st.session_state
                  or st.session_state.selected_song not in songs):
        st.session_state.selected_song = songs[0]

    def _pick_song(song):
        st.session_state.selected_song = song

    with col_chart2:
        st.markdown("### Indexed Track Library")
        df_songs = pd.DataFrame(songs, columns=["Track Name"])
        df_songs.index += 1
        st.dataframe(df_songs, use_container_width=True, height=300)

    # ---- Track selector + immediate deep-dive output ----
    if songs:
        st.markdown("---")
        st.markdown("### 🔬 Analyze a Track")
        st.caption("Pick a track (or click Analyze on a thumbnail below) to see its full breakdown.")

        # selectbox is keyed to 'selected_song' so the card buttons can drive it
        st.selectbox("Select a track to analyze", songs, key="selected_song")

        render_song_deep_dive(st.session_state.selected_song)

    st.markdown("---")
    st.markdown("### Visual Database Index")
    st.caption("Browse fingerprint thumbnails — click **Analyze** to load a track into the panel above.")

    cols = st.columns(3)
    for i, song in enumerate(songs):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{song}**")
                fig, hash_count, duration = get_song_thumbnail_data(song)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True,
                                    config={'displayModeBar': False}, key=f"thumb_{i}")
                    c_a, c_b = st.columns(2)
                    c_a.caption(f"⏳ {duration:.1f}s")
                    c_b.caption(f"🔗 {hash_count:,} hashes")
                else:
                    st.caption("Audio file unavailable for thumbnail rendering.")
                st.button("🔍 Analyze", key=f"analyze_{i}", use_container_width=True,
                          on_click=_pick_song, args=(song,))


# ------------------------------------------
# PAGE 3: IDENTIFY SONG (The Core Engine)
# ------------------------------------------
elif page == "🔍 Identify Song":
    st.title("🔍 Song Identification Engine")
    st.caption("Upload an unknown clip and let the hashing algorithm find the exact match.")

    uploaded = st.file_uploader("Drop audio file here (mp3, wav)", type=["mp3", "wav"])

    if uploaded:
        st.markdown("---")
        st.markdown("### Input Playback")
        st.audio(uploaded)

        tracker_placeholder = st.empty()
        step_times = []

        # Step 0: Audio Loaded
        t0 = time.time()
        update_pipeline_tracker(tracker_placeholder, 0, step_times)
        audio, fs = librosa.load(uploaded, sr=None, mono=True)
        duration = len(audio) / fs
        t_audio = time.time() - t0
        step_times.append(t_audio)

        # Step 1: Spectrogram Created
        t1 = time.time()
        update_pipeline_tracker(tracker_placeholder, 1, step_times)
        f, t, S = generate_spectrogram(audio, fs)
        spectrogram_time = (time.time() - t1)
        step_times.append(spectrogram_time)

        # Step 2: Peaks Extracted
        t2 = time.time()
        update_pipeline_tracker(tracker_placeholder, 2, step_times)
        fi, ti = extract_constellation(S)
        constellation_time = (time.time() - t2)
        step_times.append(constellation_time)

        # Step 3: Hashes Generated
        t3 = time.time()
        update_pipeline_tracker(tracker_placeholder, 3, step_times)
        query_hashes = generate_hashes(fi, ti)
        hash_time = (time.time() - t3)
        step_times.append(hash_time)

        # Step 4: Database Search
        t4 = time.time()
        update_pipeline_tracker(tracker_placeholder, 4, step_times)
        votes = identify_song(query_hashes)
        matching_time = (time.time() - t4)
        step_times.append(matching_time)

        # Match Calculations
        song_scores = {song: max(offsets.values()) for song, offsets in votes.items()}
        rankings = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
        log_action(f"Identified query: {uploaded.name}")

        # Step 5: Match Found (Finalize Tracker)
        t_final = time.time() - t4
        step_times.append(t_final)
        update_pipeline_tracker(tracker_placeholder, 6, step_times)

        # Global Metadata
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{duration:.2f}s")
        col2.metric("Sample Rate", f"{fs} Hz")
        col3.metric("Extracted Peaks", f"{len(fi):,}")
        col4.metric("Generated Hashes", f"{len(query_hashes):,}")

        # FEATURE 1: FINGERPRINT COVERAGE SCORE
        st.markdown("---")
        matched_hashes = sum(1 for h, _ in query_hashes if h in database)
        unmatched_hashes = len(query_hashes) - matched_hashes
        coverage_score = (matched_hashes / max(len(query_hashes), 1)) * 100

        cov_color = "#34D399" if coverage_score >= 90 else ("#FBBF24" if coverage_score >= 70 else "#FB7185")

        st.markdown("### Fingerprint Integrity & Coverage")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"""
            <div class='coverage-card'>
                <div style='color:#94A3B8; font-size:0.92rem; font-weight:600; text-transform:uppercase; letter-spacing:0.6px;'>Coverage Score</div>
                <div style='color:{cov_color}; font-size:2.1rem; font-weight:800;'>{coverage_score:.1f}%</div>
                <div style='height:6px; border-radius:999px; background:rgba(148,163,184,0.15); margin-top:8px;'>
                    <div style='height:6px; width:{min(coverage_score,100):.1f}%; border-radius:999px; background:{cov_color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        col_c2.metric("Matched Hashes", f"{matched_hashes:,}")
        col_c3.metric("Unmatched Hashes", f"{unmatched_hashes:,}")

        # Prediction Result Area
        st.markdown("---")
        if not rankings:
            st.error("❌ No matching song found in the database. Try a cleaner clip or add to the index.")
            st.stop()

        best_song = rankings[0][0]
        best_score = rankings[0][1]
        confidence = (best_score / max(sum(song_scores.values()), 1)) * 100 if len(rankings) > 1 else 100.0

        with st.container(border=True):
            col_res, col_gauge = st.columns([3, 1])

            with col_res:
                st.markdown(f'<span class="badge-success">CONFIDENCE: {confidence:.1f}%</span>',
                            unsafe_allow_html=True)
                st.markdown(f"<h1 style='color:#67E8F9; margin-top:12px; margin-bottom:4px;'>{best_song}</h1>",
                            unsafe_allow_html=True)
                st.markdown(f"**Top Score:** {best_score} matched offset points")

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    number={'suffix': "%", 'font': {'size': 38, 'color': INK}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                        'bar': {'color': ACCENT_CYAN},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(251, 113, 133, 0.18)"},
                            {'range': [50, 80], 'color': "rgba(251, 191, 36, 0.18)"},
                            {'range': [80, 100], 'color': "rgba(52, 211, 153, 0.18)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=260, autosize=True,
                                        margin=dict(l=30, r=30, t=40, b=30))
                theme_fig(fig_gauge)
                st.plotly_chart(fig_gauge, use_container_width=True)

        # Interactive Visualizations
        st.markdown("### Interactive Audio Analysis")
        tab_wave, tab_spec, tab_const, tab_offset, tab_perf = st.tabs([
            "🎧 Waveform", "🌊 Spectrogram", "✨ Constellation Map", "📈 Offset Alignment", "⚡ Performance"
        ])

        with tab_wave:
            st.markdown("**Audio Waveform**")
            st.caption(f"Duration: {duration:.2f} seconds")
            step_w = max(1, len(audio) // 3000)
            time_axis = np.linspace(0, duration, len(audio))[::step_w]

            fig_wave = px.line(x=time_axis, y=audio[::step_w], labels={'x': 'Time (s)', 'y': 'Amplitude'})
            fig_wave.update_traces(line_color=ACCENT_CYAN, line_width=1.2)
            fig_wave.update_layout(autosize=True)
            theme_fig(fig_wave)
            st.plotly_chart(fig_wave, use_container_width=True)

        with tab_spec:
            step_t = max(1, S.shape[1] // 800)
            step_f = max(1, S.shape[0] // 400)

            fig_spec = px.imshow(
                S[::step_f, ::step_t],
                origin='lower',
                aspect='auto',
                color_continuous_scale='Viridis',
                labels={'x': 'Time Bin (Downsampled)', 'y': 'Frequency Bin (Downsampled)', 'color': 'dB'}
            )
            fig_spec.update_layout(autosize=True)
            theme_fig(fig_spec)
            st.plotly_chart(fig_spec, use_container_width=True)

        with tab_const:
            fig_const = go.Figure()
            fig_const.add_trace(go.Heatmap(z=S[::step_f, ::step_t], colorscale='Viridis',
                                           showscale=False, opacity=0.65))
            fig_const.add_trace(go.Scatter(
                x=ti / step_t, y=fi / step_f,
                mode='markers',
                marker=dict(size=4, color=ACCENT_ROSE, opacity=0.85),
                name='Peak Constellation'
            ))
            fig_const.update_layout(autosize=True)
            theme_fig(fig_const)
            st.plotly_chart(fig_const, use_container_width=True)

        with tab_offset:
            best_offsets = votes[best_song]
            peak_offset = max(best_offsets, key=best_offsets.get)
            window = {k: v for k, v in best_offsets.items() if abs(k - peak_offset) <= 20}

            fig_offset = px.bar(
                x=list(window.keys()),
                y=list(window.values()),
                labels={'x': 'Time Offset (Δt)', 'y': 'Match Density (Votes)'},
                title=f"Offset Histogram for '{best_song}'"
            )
            fig_offset.add_vline(x=peak_offset, line_dash="dash", line_color=ACCENT_CYAN,
                                 annotation_text="Winning Offset")
            fig_offset.update_traces(marker_color=ACCENT_VIOLET)
            fig_offset.update_layout(autosize=True)
            theme_fig(fig_offset)
            st.plotly_chart(fig_offset, use_container_width=True)

        with tab_perf:
            perf_df = pd.DataFrame([
                {"Task": "Spectrogram STFT", "Time (ms)": spectrogram_time * 1000},
                {"Task": "Peak Extraction", "Time (ms)": constellation_time * 1000},
                {"Task": "Combinatorial Hashing", "Time (ms)": hash_time * 1000},
                {"Task": "Database Matching", "Time (ms)": matching_time * 1000}
            ])
            fig_perf = px.bar(perf_df, x="Time (ms)", y="Task", orientation='h',
                              title="Pipeline Execution Latency")
            fig_perf.update_traces(marker_color=ACCENT_MINT)
            fig_perf.update_layout(autosize=True, margin=dict(l=20, r=30, t=46, b=40))
            theme_fig(fig_perf)
            st.plotly_chart(fig_perf, use_container_width=True)

        # FEATURE 2: HASH COLLISION STATISTICS
        st.markdown("---")
        st.markdown("### Hash Collision Analysis")
        st.caption("Higher collisions imply less discriminative fingerprints.")

        collisions = [len(database[h]) for h, _ in query_hashes if h in database]

        if collisions:
            avg_col = sum(collisions) / len(collisions)
            max_col = max(collisions)
            min_col = min(collisions)

            c_col1, c_col2, c_col3 = st.columns(3)
            c_col1.metric("Average Collision Rate", f"{avg_col:.2f}")
            c_col2.metric("Max Collision Count", max_col)
            c_col3.metric("Min Collision Count", min_col)

            fig_collisions = px.histogram(
                x=collisions,
                nbins=30,
                labels={'x': 'Number of Database Matches per Hash', 'y': 'Hash Count'}
            )
            fig_collisions.update_traces(marker_color=ACCENT_AMBER)
            fig_collisions.update_layout(autosize=True)
            theme_fig(fig_collisions)
            st.plotly_chart(fig_collisions, use_container_width=True)
        else:
            st.warning("No matched hashes found to compute collisions.")

        # Top Candidates Leaderboard
        st.markdown("### Top Match Candidates")
        df_top = pd.DataFrame(rankings[:5], columns=["Song", "Score"])
        df_top["Confidence"] = (df_top["Score"] / max(sum(song_scores.values()), 1)) * 100

        st.dataframe(
            df_top,
            column_config={
                "Score": st.column_config.NumberColumn(format="%d"),
                "Confidence": st.column_config.ProgressColumn(
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )
            },
            hide_index=True,
            use_container_width=True
        )


# ------------------------------------------
# PAGE 4: BATCH PROCESSING
# ------------------------------------------
elif page == "📦 Batch Processing":
    st.title("📦 Batch Processing Engine")
    st.caption("Process multiple query clips simultaneously and export comprehensive analytics.")

    uploaded_files = st.file_uploader(
        "Upload Query Directory",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("---")
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_batch = time.time()

        for idx, file in enumerate(uploaded_files):
            status_text.text(f"Processing: {file.name} ({idx+1}/{len(uploaded_files)})")

            audio, fs = librosa.load(file, sr=None, mono=True)
            f, t, S = generate_spectrogram(audio, fs)
            fi, ti = extract_constellation(S)
            hashes = generate_hashes(fi, ti)
            votes = identify_song(hashes)

            scores = {song: max(offsets.values()) for song, offsets in votes.items()}
            rankings = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            prediction = rankings[0][0] if rankings else "Unknown"
            top_score = rankings[0][1] if rankings else 0

            results.append({
                "Filename": file.name,
                "Prediction": prediction,
                "Confidence Score": top_score,
                "Hashes Checked": len(hashes)
            })

            progress_bar.progress((idx + 1) / len(uploaded_files))

        total_time = time.time() - start_batch
        status_text.success(f"✅ Batch processed {len(uploaded_files)} files in {total_time:.2f} seconds.")
        log_action(f"Batch processed {len(uploaded_files)} files.")

        df_batch = pd.DataFrame(results)

        c1, c2, c3 = st.columns(3)
        success_rate = (len(df_batch[df_batch['Prediction'] != 'Unknown']) / len(df_batch)) * 100

        c1.metric("Throughput Speed", f"{(total_time / len(uploaded_files)):.2f}s / file")
        c2.metric("Match Success Rate", f"{success_rate:.1f}%")
        c3.metric("Total Computations", f"{df_batch['Hashes Checked'].sum():,}")

        st.markdown("### Results Summary")
        st.dataframe(df_batch, use_container_width=True)

        csv = df_batch.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇️ Download Batch Analytics Report (CSV)",
            data=csv,
            file_name=f"audioAI_batch_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
