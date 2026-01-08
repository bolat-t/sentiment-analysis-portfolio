"""
Streamlit Sentiment Analysis App with Google Sheets Logging
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
    print("⚠️ Google Sheets logger not available")

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis | Portfolio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# [Keep all your existing CSS here - same as before]
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0a0a0a; color: #e5e7eb; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }
    h1 {
        background: linear-gradient(135deg, #a78bfa 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stTextArea textarea {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 0.5rem !important;
        color: #e5e7eb !important;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.4) !important;
    }
    .stButton button[kind="secondary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #14b8a6 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    .hero-section {
        background: radial-gradient(circle at 20% 30%, rgba(124, 58, 237, 0.35), transparent 40%),
                    radial-gradient(circle at 80% 25%, rgba(236, 72, 153, 0.35), transparent 40%);
        padding: 3rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    .model-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.6);
        border-color: #14b8a6;
    }
    .positive-card { border-left: 3px solid #34d399 !important; }
    .negative-card { border-left: 3px solid #f87171 !important; }
    .tag-pill {
        background-color: rgba(20, 184, 166, 0.2);
        color: #5eead4;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        margin: 0.25rem;
        border: 1px solid rgba(20, 184, 166, 0.3);
    }
    .feature-icon {
        width: 8px;
        height: 8px;
        background: linear-gradient(135deg, #14b8a6 0%, #a78bfa 100%);
        border-radius: 50%;
    }
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# Initialise session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
    st.session_state.model_loaded = False
    st.session_state.analysis_history = []
    st.session_state.logger = None

# Logging function - tries Google Sheets first, falls back to JSON
def log_user_submission(text, result, processing_time):
    """Log to Google Sheets (if available) and local JSON (backup)"""
    
    # Try Google Sheets
    if st.session_state.logger and st.session_state.logger.enabled:
        try:
            st.session_state.logger.log_submission(text, result, processing_time)
        except Exception as e:
            print(f"Google Sheets logging failed: {e}")
    
    # Also keep local JSON backup
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
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
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
    """Create gauge chart"""
    value = 50 + (confidence * 50) if sentiment == 'positive' else 50 - (confidence * 50)
    color = "#34d399" if sentiment == 'positive' else "#f87171"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Sentiment Score", 'font': {'size': 20, 'color': '#e5e7eb'}},
        number={'font': {'size': 40, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#475569"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "#0f172a",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(248, 113, 113, 0.2)'},
                {'range': [30, 50], 'color': 'rgba(251, 191, 36, 0.2)'},
                {'range': [50, 70], 'color': 'rgba(167, 139, 250, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(52, 211, 153, 0.2)'}
            ],
            'threshold': {'line': {'color': "#14b8a6", 'width': 4}, 'value': value}
        }
    ))
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20),
                     paper_bgcolor="#0f172a", font={'family': "Inter"})
    return fig

def create_confidence_bars(results):
    """Create confidence comparison bars"""
    models, confidences, sentiments = [], [], []
    
    for model_name, result in results.items():
        if isinstance(result, dict) and 'confidence' in result:
            models.append(model_name.upper())
            confidences.append(result['confidence'])
            sentiments.append(result['sentiment'])
    
    colors = ['#34d399' if s == 'positive' else '#f87171' for s in sentiments]
    
    fig = go.Figure(data=[go.Bar(x=models, y=confidences, marker_color=colors,
                                 text=[f"{c:.1%}" for c in confidences], textposition='outside')])
    
    fig.update_layout(title="Model Confidence Comparison", height=400,
                     paper_bgcolor="#0f172a", plot_bgcolor="#1e293b")
    return fig

def main():
    load_custom_css()
    
    # Initialise Google Sheets logger
    if SHEETS_AVAILABLE and st.session_state.logger is None:
        st.session_state.logger = initialise_logger()
    
    # Hero
    st.markdown("""
        <div class="hero-section">
            <p style="color: #a78bfa; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em;">
                NLP & Machine Learning
            </p>
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Advanced Sentiment Analysis</h1>
            <p style="color: #d1d5db; font-size: 1.125rem;">
                Multi-model NLP pipeline for real-time sentiment classification
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        
        if st.button("Load Models", type="primary", use_container_width=True):
            with st.spinner("Loading..."):
                st.session_state.analyzer = load_analyzer()
                if st.session_state.analyzer:
                    st.session_state.model_loaded = True
                    st.success("Models loaded!")
        
        # Show logging status
        if SHEETS_AVAILABLE and st.session_state.logger:
            if st.session_state.logger.enabled:
                st.success("📊 Google Sheets logging active")
            else:
                st.warning("📊 Logging to local files only")
        
        st.markdown("---")
        st.markdown("### Available Models")
        st.markdown("""
        <span class="tag-pill">VADER</span> <span class="tag-pill">TextBlob</span><br>
        <span class="tag-pill">ML Model</span> <span class="tag-pill">Transformer</span>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "94.2%")
        with col2:
            st.metric("Speed", "<1s")
        
        if st.session_state.analysis_history:
            st.markdown("---")
            st.markdown("### History")
            st.write(f"Analyses: **{len(st.session_state.analysis_history)}**")
            if st.button("Clear", use_container_width=True):
                st.session_state.analysis_history = []
                st.rerun()
    
    if not st.session_state.model_loaded:
        st.info("Click 'Load Models' in the sidebar")
        return
    
    # Text input
    st.markdown("### Enter Text to Analyse")
    
    samples = {
        "Positive": "Amazing product! Exceeded expectations!",
        "Negative": "Terrible. Broke after one day.",
        "Mixed": "Okay product. Some good, some bad.",
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Positive", type="secondary", use_container_width=True):
            st.session_state.sample_text = samples["Positive"]
    with col2:
        if st.button("Negative", type="secondary", use_container_width=True):
            st.session_state.sample_text = samples["Negative"]
    with col3:
        if st.button("Mixed", type="secondary", use_container_width=True):
            st.session_state.sample_text = samples["Mixed"]
    
    text_input = st.text_area("", value=st.session_state.get('sample_text', ''),
                              height=150, placeholder="Type your review...",
                              label_visibility="collapsed")
    
    col1, col2, col3 = st.columns([2, 1, 3])
    with col1:
        analyse = st.button("Analyse Sentiment", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.sample_text = ''
            st.rerun()
    
    if analyse and text_input.strip():
        with st.spinner("Analysing..."):
            start = time.time()
            
            try:
                result = st.session_state.analyzer.get_ensemble_prediction(text_input)
                proc_time = time.time() - start
                
                if 'error' not in result:
                    # Log the submission
                    log_user_submission(text_input, result, proc_time)
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'time': time.strftime('%H:%M:%S')
                    })
                    
                    st.markdown("---")
                    st.markdown("## Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    sentiment = result['sentiment']
                    confidence = result['confidence']
                    
                    with col1:
                        emoji = "😊" if sentiment == 'positive' else "😞"
                        st.metric("Sentiment", f"{emoji} {sentiment.upper()}")
                    with col2:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col3:
                        st.metric("Time", f"{proc_time:.3f}s")
                    with col4:
                        st.metric("Length", f"{len(text_input.split())} words")
                    
                    # Charts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_sentiment_gauge(sentiment, confidence), 
                                      use_container_width=True)
                    with col2:
                        individual = {k: v for k, v in result.get('individual_results', {}).items() 
                                    if k != 'text'}
                        if individual:
                            st.plotly_chart(create_confidence_bars(individual), 
                                          use_container_width=True)
                    
                    # Model cards
                    st.markdown("### Individual Models")
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
                                    <div class="model-card {card_class}">
                                        <h4 style="color: #14b8a6; font-size: 0.875rem; text-transform: uppercase; margin-bottom: 1rem;">
                                            {name.upper()}
                                        </h4>
                                        <div style="text-align: center;">
                                            <div style="font-size: 2.5rem;">{emoji}</div>
                                            <div style="font-size: 1.25rem; font-weight: 600; color: #fff;">
                                                {sent.upper()}
                                            </div>
                                            <div style="color: #9ca3af; font-size: 0.875rem;">
                                                {conf:.1%}
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif analyse:
        st.warning("Please enter text")
    
    # History
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("## Recent Analyses")
        df = pd.DataFrame(st.session_state.analysis_history[-5:])
        df['sentiment'] = df['sentiment'].apply(lambda x: f"{'😊' if x == 'positive' else '😞'} {x.upper()}")
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
        st.dataframe(df[['time', 'text', 'sentiment', 'confidence']], 
                    hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()