"""
Streamlit Sentiment Analysis App - Polished UI
Save as: app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import json
from datetime import datetime

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
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ========== GLOBAL STYLES ========== */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.01em;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #e5e7eb;
    }
    
    /* ========== SIDEBAR STYLES ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);

        /* ADD THESE LINES ↓↓↓ */
        width: 400px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 400px !important; /* Ensures inner container matches */
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 0.5rem 0;
    }

    [data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }

    
    /* ========== HEADER STYLES ========== */
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
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* ========== TEXT AREA STYLES ========== */
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
    }
    
    .stTextArea textarea::placeholder {
        color: #64748b !important;
        font-style: italic;
    }
    
    /* ========== BUTTON STYLES ========== */
    /* Primary Button - Teal Gradient */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.875rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.4), 
                    0 0 0 0 rgba(20, 184, 166, 0) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.6),
                    0 0 0 3px rgba(20, 184, 166, 0.2) !important;
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    }
    
    .stButton button[kind="primary"]:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary Button - Purple Gradient */
    .stButton button[kind="secondary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4) !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6),
                    0 0 0 3px rgba(139, 92, 246, 0.2) !important;
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    }
    
    /* Tertiary Button - Red/Clear */
    .stButton button:not([kind="primary"]):not([kind="secondary"]) {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4) !important;
    }
    
    .stButton button:not([kind="primary"]):not([kind="secondary"]):hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6),
                    0 0 0 3px rgba(239, 68, 68, 0.2) !important;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    }
    
    /* ========== METRIC CARDS ========== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 1rem;
        padding: 1.75rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(20, 184, 166, 0.4);
    }
    
    [data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
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
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    /* ========== HERO SECTION ========== */
    .hero-section {
        background: radial-gradient(circle at 20% 30%, rgba(124, 58, 237, 0.4), transparent 50%),
                    radial-gradient(circle at 80% 25%, rgba(236, 72, 153, 0.35), transparent 50%),
                    radial-gradient(circle at 70% 75%, rgba(244, 63, 94, 0.35), transparent 50%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        padding: 3rem 2.5rem;
        border-radius: 1.5rem;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 0%, rgba(20, 184, 166, 0.1), transparent 50%);
        pointer-events: none;
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
    
    .hero-text {
        color: #94a3b8;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* ========== MODEL RESULT CARDS ========== */
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
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, currentColor, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6),
                    0 0 0 1px rgba(20, 184, 166, 0.3);
        border-color: rgba(20, 184, 166, 0.4);
    }
    
    .model-card:hover::before {
        opacity: 1;
    }
    
    .positive-card {
        border-left: 4px solid #34d399 !important;
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.08) 0%, transparent 100%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    }
    
    .positive-card::before {
        color: #34d399;
    }
    
    .negative-card {
        border-left: 4px solid #f87171 !important;
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.08) 0%, transparent 100%),
                    linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
    }
    
    .negative-card::before {
        color: #f87171;
    }
    
    .model-card-title {
        color: #14b8a6;
        font-size: 0.875rem;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.1em;
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
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .model-card-confidence {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    /* ========== TAG PILLS ========== */
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
    
    /* ========== STATUS BADGES ========== */
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
    
    /* ========== EXPANDER STYLES ========== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 0.75rem !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        padding: 1rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: rgba(20, 184, 166, 0.4) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(10, 10, 10, 0.9) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 0.75rem 0.75rem !important;
        padding: 1.5rem !important;
    }
    
    /* ========== INFO/SUCCESS/WARNING BOXES ========== */
    .stAlert {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
        border-left: 4px solid #14b8a6 !important;
        border-radius: 0.75rem !important;
        color: #e2e8f0 !important;
        padding: 1.25rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px);
    }
    
    /* ========== DATAFRAME STYLES ========== */
    [data-testid="stDataFrame"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 1rem !important;
        overflow: hidden !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* ========== FEATURE ITEMS ========== */
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
    
    /* ========== DIVIDER ========== */
    hr {
        margin: 2.5rem 0;
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
    }
    
    /* ========== SPACING UTILITIES ========== */
    .spacing-sm { margin-top: 1rem; }
    .spacing-md { margin-top: 2rem; }
    .spacing-lg { margin-top: 3rem; }
    
    /* ========== HIDE STREAMLIT BRANDING ========== */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* ========== SCROLLBAR ========== */
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
    
    /* ========== ANIMATIONS ========== */
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
    </style>
    """, unsafe_allow_html=True)

# Initialise session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
    st.session_state.model_loaded = False
    st.session_state.analysis_history = []
    st.session_state.logger = None

# Logging function
def log_user_submission(text, result, processing_time):
    """Log to Google Sheets and local JSON"""
    if st.session_state.logger and st.session_state.logger.enabled:
        try:
            st.session_state.logger.log_submission(text, result, processing_time)
        except Exception as e:
            print(f"Google Sheets logging failed: {e}")
    
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

@st.cache_resource
def load_analyzer():
    """Load sentiment analyzer"""
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
        st.error(f"Failed to load analyzer: {e}")
        return None

def create_sentiment_gauge(sentiment, confidence):
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

def create_confidence_bars(results):
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
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'tickfont': {'size': 14, 'family': 'Inter'}
        },
        yaxis={
            'title': 'Confidence',
            'color': '#94a3b8',
            'range': [0, 1.15],
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'tickfont': {'size': 14, 'family': 'Inter'}
        },
        height=400,
        showlegend=False,
        paper_bgcolor="rgba(15, 23, 42, 0.5)",
        plot_bgcolor="rgba(30, 41, 59, 0.5)",
        font={'family': 'Inter'},
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return fig

def main():
    load_custom_css()
    
    # Initialise logger
    if SHEETS_AVAILABLE and st.session_state.logger is None:
        st.session_state.logger = initialise_logger()
    
    # Hero Header
    st.markdown("""
        <div class="hero-section animate-fade-in">
            <p class="hero-subtitle">NLP & Machine Learning</p>
            <h1 class="hero-title">Advanced Sentiment Analysis</h1>
            <p class="hero-description">Multi-model NLP pipeline for real-time sentiment classification</p>
            <p class="hero-text">
                Analyse text sentiment using VADER, TextBlob, traditional ML models, and transformer-based 
                deep learning for accurate classification with confidence scoring.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("")  # Spacing
        
        if st.button("🚀 Load Models", type="primary", use_container_width=True):
            with st.spinner("Loading models..."):
                st.session_state.analyzer = load_analyzer()
                if st.session_state.analyzer:
                    st.session_state.model_loaded = True
                    st.success("✅ Models loaded successfully!")
        
        st.markdown("")  # Spacing
        
        # Logging status
        if SHEETS_AVAILABLE and st.session_state.logger:
            if st.session_state.logger.enabled:
                st.markdown('<div class="status-badge status-success">✓ Google Sheets Active</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge status-warning">⚠ Local Logging Only</div>', 
                           unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 🤖 Available Models")
        st.markdown("")
        st.markdown("""
        <div style="line-height: 2.2;">
            <span class="tag-pill">VADER</span><br>
            <span class="tag-pill">TextBlob</span><br>
            <span class="tag-pill">ML Model</span><br>
            <span class="tag-pill">Transformer</span><br>
            <span class="tag-pill">Ensemble</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 System Metrics")
        st.markdown("")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "94.2%")
        with col2:
            st.metric("Speed", "<1s")
        
        if st.session_state.analysis_history:
            st.markdown("---")
            st.markdown("### 📜 History")
            st.markdown("")
            st.write(f"**Total Analyses:** {len(st.session_state.analysis_history)}")
            st.markdown("")
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.analysis_history = []
                st.rerun()
    
    # Main content
    if not st.session_state.model_loaded:
        st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
        st.info("👈 **Click 'Load Models' in the sidebar to get started**")
        
        st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ **About This Application**", expanded=True):
            st.markdown("""
            This sentiment analysis platform uses multiple machine learning models to determine whether 
            text expresses positive or negative sentiment with high accuracy and confidence scoring.
            
            **Key Features:**
            """)
            
            st.markdown("""
            <div style="margin-top: 1rem;">
                <div class="feature-item">
                    <div class="feature-icon"></div>
                    <span>Real-time sentiment prediction with ensemble modeling</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon"></div>
                    <span>Multiple model comparison and confidence scoring</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon"></div>
                    <span>Interactive visualizations and detailed analytics</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon"></div>
                    <span>Automated logging to Google Sheets for analysis</span>
                </div>
                <div class="feature-item">
                    <div class="feature-icon"></div>
                    <span>Processing speed under 1 second per analysis</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            
            **How to Use:**
            1. Click **'Load Models'** in the sidebar to initialise the system
            2. Enter your text in the input area or use sample texts
            3. Click **'Analyse Sentiment'** to get predictions
            4. View detailed results, confidence scores, and model comparisons
            """)
        
        return
    
    # Text input section
    st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
    st.markdown("## 📝 Enter Text to Analyse")
    st.markdown("")
    
    # Sample text buttons
    samples = {
        "Positive": "This product is absolutely amazing! The quality exceeded my expectations and the customer service was outstanding. Highly recommended!",
        "Negative": "Terrible experience. The product broke after one day and customer service was unhelpful. Complete waste of money.",
        "Mixed": "The product is okay. Some features work well but others are disappointing. Average quality for the price.",
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✨ Positive Sample", type="secondary", use_container_width=True):
            st.session_state.sample_text = samples["Positive"]
            st.rerun()
    with col2:
        if st.button("⚠️ Negative Sample", type="secondary", use_container_width=True):
            st.session_state.sample_text = samples["Negative"]
            st.rerun()
    with col3:
        if st.button("🔀 Mixed Sample", type="secondary", use_container_width=True):
            st.session_state.sample_text = samples["Mixed"]
            st.rerun()
    
    st.markdown("")
    
    # Text input area
    text_input = st.text_area(
        "Your text:",
        value=st.session_state.get('sample_text', ''),
        height=180,
        placeholder="Type or paste your review, comment, feedback, or any text you'd like to analyse for sentiment...",
        label_visibility="collapsed"
    )
    
    st.markdown("")
    
    # Action buttons
    col1, col2, col3 = st.columns([2.5, 1.5, 4])
    with col1:
        analyse = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.sample_text = ''
            st.rerun()
    
    # Analysis execution
    if analyse and text_input.strip():
        with st.spinner("🤖 Analysing sentiment..."):
            start = time.time()
            
            try:
                result = st.session_state.analyzer.get_ensemble_prediction(text_input)
                proc_time = time.time() - start
                
                if 'error' not in result:
                    # Log submission
                    log_user_submission(text_input, result, proc_time)
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'time': time.strftime('%H:%M:%S')
                    })
                    
                    st.markdown("---")
                    st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
                    st.markdown("## 📊 Analysis Results")
                    st.markdown("")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    with col1:
                        emoji = "😊" if sentiment == 'positive' else "😞"
                        st.metric("Overall Sentiment", f"{emoji} {sentiment.upper()}")
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.1%}")
                    with col3:
                        st.metric("Processing Time", f"{proc_time:.3f}s")
                    with col4:
                        st.metric("Text Length", f"{len(text_input.split())} words")
                    
                    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
                    
                    # Visualisations
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_sentiment_gauge(sentiment, confidence), 
                                      use_container_width=True, key="gauge")
                    with col2:
                        individual = {k: v for k, v in result.get('individual_results', {}).items() 
                                    if k != 'text'}
                        if individual:
                            st.plotly_chart(create_confidence_bars(individual), 
                                          use_container_width=True, key="bars")
                    
                    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
                    
                    # Individual model results
                    st.markdown("## 🔬 Individual Model Results")
                    st.markdown("")
                    
                    model_results = {k: v for k, v in result.get('individual_results', {}).items() 
                                   if k != 'text'}
                    
                    if model_results:
                        cols = st.columns(len(model_results))
                        for idx, (name, res) in enumerate(model_results.items()):
                            if isinstance(res, dict) and 'sentiment' in res:
                                with cols[idx]:
                                    sent = res['sentiment']
                                    conf = res.get('confidence', 0)
                                    emoji = "😊" if sent == 'positive' else "😞"
                                    card_class = "positive-card" if sent == 'positive' else "negative-card"
                                    
                                    st.markdown(f"""
                                    <div class="model-card {card_class} animate-fade-in">
                                        <div class="model-card-title">{name.upper()}</div>
                                        <div style="text-align: center;">
                                            <div class="model-card-emoji">{emoji}</div>
                                            <div class="model-card-sentiment">{sent.upper()}</div>
                                            <div class="model-card-confidence">Confidence: {conf:.1%}</div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    st.markdown('<div class="spacing-md"></div>', unsafe_allow_html=True)
                    
                    # Text preview
                    with st.expander("📄 View Analysed Text"):
                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.5); padding: 1.5rem; border-radius: 0.75rem; 
                                    line-height: 1.8; color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.1);">
                            {text_input}
                        </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
    
    elif analyse:
        st.warning("⚠️ Please enter some text to analyse")
    
    # Recent analyses history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
        st.markdown("## 📈 Recent Analyses")
        st.markdown("")
        
        history_df = pd.DataFrame(st.session_state.analysis_history[-5:])
        history_df['sentiment'] = history_df['sentiment'].apply(
            lambda x: f"{'😊' if x == 'positive' else '😞'} {x.upper()}"
        )
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            history_df[['time', 'text', 'sentiment', 'confidence']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "time": "Time",
                "text": st.column_config.TextColumn("Text", width="large"),
                "sentiment": "Sentiment",
                "confidence": "Confidence"
            }
        )
    
    # Footer
    st.markdown('<div class="spacing-lg"></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.875rem; padding: 2rem 0 1rem 0;">
            <p style="font-weight: 600;">Built with Python, Scikit-learn, Transformers & Streamlit</p>
            <p style="margin-top: 0.75rem; opacity: 0.7;">© 2026 Data Science Portfolio • All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()