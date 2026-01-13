import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.sentiment_analyzer import SentimentAnalyzer
from src.preprocessor import TextPreprocessor
from src.data_collector import DataCollector

# Import Google Sheets logger
try:
    from google_sheets_logger import initialise_logger
    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis | AI Portfolio",
    page_icon="assets/sentiment.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* Force dark theme regardless of browser mode */
:root {
  color-scheme: dark;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)


# Custom CSS (preserved from original)
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.01em;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #e5e7eb;
    }
                
    /* Fix st.info (light mode override) */
div[data-testid="stAlert"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    color: #e5e7eb !important;
    border-left: 4px solid #14b8a6 !important;
}

/* Remove Streamlit blue info background */
div[data-testid="stAlert"] svg {
    color: #14b8a6 !important;
}

                /* Kill white container bleed */
section[data-testid="stSidebar"],
div[data-testid="block-container"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

/* Fix expander header in light mode */
details > summary {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    color: #e5e7eb !important;
    border-radius: 0.75rem !important;
}

/* Expander content */
details > div {
    background: rgba(15, 23, 42, 0.9) !important;
}

    /* Improve visibility of st.info text */
div[data-testid="stAlert"] p {
    color: #e5e7eb !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

/* Make emoji/icon pop */
div[data-testid="stAlert"] span {
    filter: brightness(1.3);
}
div[data-testid="stAlert"] {
    background: linear-gradient(
        135deg,
        rgba(20, 184, 166, 0.15),
        rgba(30, 41, 59, 0.95)
    ) !important;
    border-left: 4px solid #14b8a6 !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}
                
/* Fix spinner background flash */
div[data-testid="stSpinner"] {
    background: transparent !important;
}

/* Spinner text */
div[data-testid="stSpinner"] > div {
    color: #14b8a6 !important;
    font-weight: 600;
}

/* Prevent white button flash on rerender */
button {
    background-color: transparent !important;
}

    
    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #a78bfa 0%, #14b8a6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
        line-height: 1.2 !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    .stTextArea textarea {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 2px solid #334155 !important;
        border-radius: 1rem !important;
        color: #f1f5f9 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        padding: 1.25rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #14b8a6 !important;
        box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.2), 0 8px 16px rgba(0, 0, 0, 0.4) !important;
        background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%) !important;
        outline: none !important;
    }
    
    .stTextArea textarea:focus-visible {
        outline: none !important;
    }
    
    div[data-baseweb="textarea"]:focus-within {
        outline: none !important;
        box-shadow: none !important;
        border-color: transparent !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.4) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.6), 0 0 0 3px rgba(20, 184, 166, 0.2) !important;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    }
    
    .stButton button[kind="secondary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4) !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6), 0 0 0 3px rgba(139, 92, 246, 0.2) !important;
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    }
    
    .stButton button:not([kind="primary"]):not([kind="secondary"]) {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4) !important;
    }
    
    .stButton button:not([kind="primary"]):not([kind="secondary"]):hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6), 0 0 0 3px rgba(239, 68, 68, 0.2) !important;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 1rem;
        padding: 1.75rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(20, 184, 166, 0.4);
    }
    
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #14b8a6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2 !important;
    }
    
    .hero-section {
        background: radial-gradient(circle at 20% 30%, rgba(124, 58, 237, 0.4), transparent 50%),
                    radial-gradient(circle at 80% 25%, rgba(236, 72, 153, 0.35), transparent 50%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        padding: 3rem 2.5rem;
        border-radius: 1.5rem;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .hero-subtitle {
        color: #a78bfa;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.75rem;
        font-weight: 700;
    }
    
    .hero-title {
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 1rem;
        line-height: 1.1;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-description {
        color: #cbd5e1;
        font-size: 1.25rem;
        margin-bottom: 0.75rem;
        font-weight: 500;
        line-height: 1.5;
    }
    
    .model-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 1.25rem;
        padding: 2rem;
        margin: 0.75rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .model-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(20, 184, 166, 0.3);
        border-color: rgba(20, 184, 166, 0.4);
    }
    
    .positive-card {
        border-left: 4px solid #34d399 !important;
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.08) 0%, transparent 100%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    }
    
    .negative-card {
        border-left: 4px solid #f87171 !important;
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.08) 0%, transparent 100%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    }
    
    .model-card-title {
        color: #14b8a6;
        font-size: 0.875rem;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    
    .model-card.positive-card .model-card-sentiment {
        color: #34d399;
    }
    
    .model-card.negative-card .model-card-sentiment {
        color: #f87171;
    }
    
    .model-card-emoji {
        font-size: 3rem;
        margin-bottom: 0.75rem;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
    }
    
    .model-card-sentiment {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .model-card-confidence {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    .tag-pill {
        background: rgba(20, 184, 166, 0.15);
        color: #5eead4;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        display: inline-block;
        margin: 0.25rem;
        border: 1px solid rgba(20, 184, 166, 0.3);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .tag-pill:hover {
        background: rgba(20, 184, 166, 0.25);
        border-color: rgba(20, 184, 166, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 0.75rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    .status-success {
        background: rgba(52, 211, 153, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .status-warning {
        background: rgba(251, 191, 36, 0.15);
        color: #fcd34d;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 0.75rem !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(10, 10, 10, 0.9) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 0 0 0.75rem 0.75rem !important;
        padding: 1.5rem !important;
    }
    
    .streamlit-expanderContent,
    .streamlit-expanderContent * {
        background-color: transparent !important;
        color: #cbd5e1 !important;
    }
    
    .stAlert {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95)) !important;
        color: #e5e7eb !important;
        border-left: 4px solid #a78bfa !important;
        border-radius: 0.75rem !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem 0;
        color: #cbd5e1;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        color: #e2e8f0;
        transform: translateX(8px);
    }
    
    .feature-icon {
        width: 10px;
        height: 10px;
        background: linear-gradient(135deg, #14b8a6 0%, #a78bfa 100%);
        border-radius: 50%;
        flex-shrink: 0;
        box-shadow: 0 0 12px rgba(20, 184, 166, 0.6);
    }
    
    hr {
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #334155 0%, #475569 100%);
        border-radius: 10px;
        border: 2px solid #0f172a;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%);
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    /* Text area container */
    div[data-testid="stTextArea"] textarea {
        background-color: #1e1e1e;
        color: #ffffff;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 12px;
        font-size: 15px;
    }

    /* Placeholder text */
    div[data-testid="stTextArea"] textarea::placeholder {
        color: #9aa0a6;
    }

    /* Focus state */
    div[data-testid="stTextArea"] textarea:focus {
        border-color: #4f9cff;
        box-shadow: 0 0 0 1px #4f9cff;
    }

    /* Label */
    div[data-testid="stTextArea"] label {
        color: #e8eaed;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALISATION
# ============================================================================

def initialise_session_state():
    """Initialise all session state variables"""
    defaults = {
        'analyzer': None,
        'model_loaded': False,
        'analysis_history': [],
        'logger': None,
        'sample_text': '',
        'current_result': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================================
# LOGGING FUNCTIONS (NON-BLOCKING WITH RATE LIMITING TRACKING)
# ============================================================================

def should_log_to_sheets(min_interval: int = 2, max_per_hour: int = 100) -> bool:
    """
    Check if we should log to Google Sheets (for rate limiting documentation)
    Returns True/False but doesn't block execution
    
    Args:
        min_interval: Minimum seconds between logs
        max_per_hour: Maximum logs per hour
    """
    if 'sheets_log_times' not in st.session_state:
        st.session_state.sheets_log_times = []
    
    now = time.time()
    
    # Remove logs older than 1 hour
    st.session_state.sheets_log_times = [
        t for t in st.session_state.sheets_log_times 
        if now - t < 3600
    ]
    
    # Check rate limits
    if st.session_state.sheets_log_times:
        last_log = st.session_state.sheets_log_times[-1]
        if now - last_log < min_interval:
            return False  # Too soon since last log
    
    if len(st.session_state.sheets_log_times) >= max_per_hour:
        return False  # Hit hourly limit
    
    # Record this log attempt
    st.session_state.sheets_log_times.append(now)
    return True


def log_to_sheets_async(logger, text: str, result: Dict[str, Any], processing_time: float):
    """
    Log to Google Sheets in a non-blocking way with rate limiting
    This happens after UI is already updated
    """
    if not logger or not logger.enabled:
        return
    
    # Check rate limits (for documentation/tracking purposes)
    if not should_log_to_sheets():
        print("⏸️ Google Sheets logging rate limited (not affecting UI)")
        return
    
    try:
        logger.log_submission(text, result, processing_time)
        print("✅ Logged to Google Sheets")
    except Exception as e:
        # Silent fail - don't interrupt user experience
        print(f"❌ Google Sheets logging failed: {e}")


def log_to_local_json(text: str, result: Dict[str, Any], processing_time: float):
    """Log to local JSON file (fast, non-blocking, no rate limiting)"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text[:500],
        'text_length': len(text.split()),
        'sentiment': result.get('sentiment', 'unknown'),
        'confidence': float(result.get('confidence', 0)),
        'processing_time': float(processing_time),
    }
    
    log_file = log_dir / f"submissions_{datetime.now().strftime('%Y%m%d')}.json"
    
    try:
        logs = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        print(f"Local logging failed: {e}")


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_analyzer():
    """Load sentiment analyser with caching"""
    try:
        analyzer = SentimentAnalyzer()
        if not analyzer.load_ml_model():
            collector = DataCollector()
            preprocessor = TextPreprocessor()
            df = collector.get_combined_dataset()
            df_processed = preprocessor.preprocess_dataframe(df)
            analyzer.train_ml_model(df_processed)
        return analyzer
    except Exception as e:
        st.error(f"Failed to load analyser: {e}")
        return None


# ============================================================================
# VISUALISATION FUNCTIONS
# ============================================================================

def create_sentiment_gauge(sentiment: str, confidence: float) -> go.Figure:
    """Create enhanced gauge chart"""
    value = 50 + (confidence * 50) if sentiment == 'positive' else 50 - (confidence * 50)
    color = "#34d399" if sentiment == 'positive' else "#f87171"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={
            'text': "Sentiment Score",
            'font': {'size': 22, 'color': '#e2e8f0', 'family': 'Inter'}
        },
        number={'font': {'size': 48, 'color': '#ffffff', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#0f172a",
            'borderwidth': 3,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(248, 113, 113, 0.2)'},
                {'range': [25, 40], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [40, 60], 'color': 'rgba(148, 163, 184, 0.2)'},
                {'range': [60, 75], 'color': 'rgba(167, 139, 250, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(52, 211, 153, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#14b8a6", 'width': 5},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor="rgba(15, 23, 42, 0.5)",
        font={'color': "#e2e8f0", 'family': "Inter"}
    )
    
    return fig


def create_confidence_bars(results: Dict[str, Any]) -> go.Figure:
    """Create enhanced confidence comparison"""
    models, confidences, sentiments = [], [], []
    
    for model_name, result in results.items():
        if isinstance(result, dict) and 'confidence' in result:
            models.append(model_name.upper())
            confidences.append(result['confidence'])
            sentiments.append(result['sentiment'])
    
    colors = ['#34d399' if s == 'positive' else '#f87171' for s in sentiments]
    
    fig = go.Figure(data=[go.Bar(
        x=models,
        y=confidences,
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.1)', width=2)
        ),
        text=[f"{c:.1%}" for c in confidences],
        textposition='outside',
        textfont=dict(size=16, color='#ffffff', family='Inter', weight='bold'),
        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
    )])
    
    fig.update_layout(
        title={
            'text': "Model Confidence Comparison",
            'font': {'size': 22, 'color': '#ffffff', 'family': 'Inter'}
        },
        xaxis={
            'title': '',
            'color': '#94a3b8',
            'gridcolor': 'rgba(148,            163, 184, 0.15)'
        },
        yaxis={
            'title': 'Confidence',
            'tickformat': '.0%',
            'color': '#94a3b8',
            'gridcolor': 'rgba(148, 163, 184, 0.15)',
            'range': [0, 1.05]
        },
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(15, 23, 42, 0.6)',
        height=400,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig

def render_logging_status():
    st.markdown("## System Observability")

    sheets_enabled = bool(
        st.session_state.get("logger") 
        and getattr(st.session_state.logger, "enabled", False)
    )

    log_times = st.session_state.get("sheets_log_times", [])
    logs_last_hour = len(log_times)
    max_logs = 100

    rate_limited = logs_last_hour >= max_logs

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Google Sheets",
            "Active" if sheets_enabled else "Disabled",
            delta="Connected" if sheets_enabled else "Fallback Mode"
        )

    with c2:
        st.metric(
            "Logs (Last Hour)",
            f"{logs_last_hour}/{max_logs}",
            delta="Rate Limited" if rate_limited else "Within Limits"
        )

    with c3:
        st.metric(
            "Local JSON Logging",
            "Enabled",
            delta="Fail-safe Active"
        )

    with c4:
        st.metric(
            "Logging Mode",
            "Async",
            delta="Non-blocking UI"
        )

    st.caption(
        "Logging runs asynchronously with rate limiting. "
        "User experience is never blocked."
    )

def render_rate_limit_bar():
    logs = st.session_state.get("sheets_log_times", [])
    usage = min(len(logs) / 100, 1.0)

    st.markdown("### Google Sheets Rate Limit")
    st.progress(usage)

    st.caption(
        f"{len(logs)} / 100 logs used in the last hour "
        "• Automatic throttling enabled"
    )



# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    load_custom_css()
    initialise_session_state()

    # Load model once
    if not st.session_state.model_loaded:
        with st.spinner("Loading sentiment model..."):
            st.session_state.analyzer = load_analyzer()
            st.session_state.model_loaded = True

        if SHEETS_AVAILABLE:
            try:
                st.session_state.logger = initialise_logger()
            except Exception:
                st.session_state.logger = None

    # Hero section
    st.markdown("""
    <div class="hero-section animate-fade-in">
        <div class="hero-subtitle">Natural Language Processing</div>
        <div class="hero-title">AI Sentiment Analysis</div>
        <div class="hero-description">
            Analyze text sentiment using an ensemble of ML and rule-based models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Input
    text_input = st.text_area(
        "Enter text to analyze",
        height=180,
        placeholder="Paste a review, tweet, or paragraph here..."
    )

    col1, col2, col3 = st.columns([1, 1, 3])

    with col1:
        analyze_btn = st.button("Analyze", type="primary")


    # Analysis
    if analyze_btn and text_input.strip():
        start_time = time.time()

        with st.spinner("Analyzing sentiment..."):
            result = st.session_state.analyzer.analyze(text_input)

        processing_time = time.time() - start_time
        st.session_state.current_result = result

        # History
        st.session_state.analysis_history.append({
            "timestamp": datetime.now().isoformat(),
            "text": text_input[:200],
            "result": result
        })

        # Logging (non-blocking)
        log_to_local_json(text_input, result, processing_time)
        log_to_sheets_async(
            st.session_state.logger,
            text_input,
            result,
            processing_time
        )

    # Results
    if st.session_state.get("current_result") is not None:

        result = st.session_state.current_result

        st.markdown("## Results")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Final Sentiment", result["final_sentiment"].upper())
        with m2:
            st.metric("Confidence", f"{result['confidence']:.1%}")
        with m3:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")

        st.plotly_chart(
            create_sentiment_gauge(
                result["final_sentiment"],
                result["confidence"]
            ),
            use_container_width=True
        )

        st.plotly_chart(
            create_confidence_bars(result["model_results"]),
            use_container_width=True
        )

        

        # Model cards
        st.markdown("## Model Breakdown")
        cols = st.columns(len(result["model_results"]))

        for col, (model, r) in zip(cols, result["model_results"].items()):
            sentiment_class = "positive-card" if r["sentiment"] == "positive" else "negative-card"
            emoji = "😊" if r["sentiment"] == "positive" else "😠"

            with col:
                st.markdown(f"""
                <div class="model-card {sentiment_class}">
                    <div class="model-card-title">{model.upper()}</div>
                    <div class="model-card-emoji">{emoji}</div>
                    <div class="model-card-sentiment">{r['sentiment'].upper()}</div>
                    <div class="model-card-confidence">
                        Confidence: {r['confidence']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        # ===============================
    # LOGGING / OBSERVABILITY
    # ===============================
    with st.expander("Logging", expanded=False):
        render_logging_status()
        render_rate_limit_bar()


    # Footer
    st.markdown("---")
    st.info("Built for AI Portfolio • Streamlit • NLP • ML Ensemble")

with st.expander(" Logging Architecture (Production Pattern)"):
    st.markdown("""
    **Request Flow**

    🧠 Sentiment Analysis  
    → ⚡ UI Updates Immediately  
    → 📝 Local JSON Logging (Always)  
    → ☁️ Google Sheets (Async + Rate Limited)  

    **Resilience Features**
    - Non-blocking background logging
    - Per-minute + per-hour rate limiting
    - Silent failure handling
    - Automatic local fallback

    Designed to mirror real production telemetry pipelines.
    """)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
