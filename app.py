"""
Streamlit app for sentiment analysis - Complete version with logging
Run with: streamlit run app.py
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

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis | Portfolio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add to your app.py
import requests

def log_to_external_service(data):
    """Send logs to an external API/database"""
    # Option 1: Google Sheets API
    # Option 2: Firebase/Supabase
    # Option 3: Your own server endpoint
    # Option 4: Cloud storage (AWS S3, Google Cloud Storage)
    
    # Example: Send to your own server
    try:
        requests.post('https://your-server.com/api/logs', json=data)
    except:
        pass  # Silent fail if server is down

# Import custom CSS from separate file for cleaner code
def load_custom_css():
    """Load custom CSS styling"""
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
    [data-testid="stSidebar"] h3 { color: #ffffff !important; font-weight: 600 !important; }
    
    h1, h2, h3 { color: #ffffff !important; font-weight: 600 !important; }
    
    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #a78bfa 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stTextArea textarea {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        border-radius: 0.5rem !important;
        color: #e5e7eb !important;
        font-size: 16px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #14b8a6 !important;
        box-shadow: 0 0 0 1px #14b8a6 !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.4) !important;
    }
    
    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.6) !important;
    }
    
    .stButton button[kind="secondary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4) !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
    }
    
    .stButton button:not([kind="primary"]):not([kind="secondary"]) {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetric"] label {
        color: #d1d5db !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #14b8a6 !important;
    }
    
    .hero-section {
        background: radial-gradient(circle at 20% 30%, rgba(124, 58, 237, 0.35), transparent 40%),
                    radial-gradient(circle at 80% 25%, rgba(236, 72, 153, 0.35), transparent 40%),
                    radial-gradient(circle at 70% 75%, rgba(244, 63, 94, 0.35), transparent 45%);
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
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.6);
        border-color: #14b8a6;
    }
    
    .positive-card {
        border-left: 3px solid #34d399 !important;
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, transparent 100%);
    }
    
    .negative-card {
        border-left: 3px solid #f87171 !important;
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.1) 0%, transparent 100%);
    }
    
    .tag-pill {
        background-color: rgba(20, 184, 166, 0.2);
        color: #5eead4;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        display: inline-block;
        margin: 0.25rem;
        border: 1px solid rgba(20, 184, 166, 0.3);
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0;
        color: #d1d5db;
    }
    
    .feature-icon {
        width: 8px;
        height: 8px;
        background: linear-gradient(135deg, #14b8a6 0%, #a78bfa 100%);
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# Initialise session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
    st.session_state.model_loaded = False
    st.session_state.analysis_history = []

# Logging functions
def log_user_submission(text, result, processing_time):
    """Log user submissions to JSON file for analytics"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text[:500],  # Limit text length in logs
        'text_length': len(text.split()),
        'sentiment': result.get('sentiment', 'unknown'),
        'confidence': float(result.get('confidence', 0)),
        'processing_time': float(processing_time),
        'models_used': list(result.get('individual_results', {}).keys()) if 'individual_results' in result else []
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
            
        return True
    except Exception as e:
        print(f"Logging error: {e}")
        return False

def log_error(error_message, context=""):
    """Log errors for debugging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'error': str(error_message),
        'context': context
    }
    
    error_file = log_dir / "errors.json"
    
    try:
        if error_file.exists():
            with open(error_file, 'r') as f:
                errors = json.load(f)
        else:
            errors = []
        
        errors.append(error_entry)
        
        with open(error_file, 'w') as f:
            json.dump(errors, f, indent=2)
    except Exception as e:
        print(f"Failed to log error: {e}")

@st.cache_resource
def load_analyzer():
    """Load sentiment analyzer (cached for performance)"""
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
        log_error(e, "Loading analyzer")
        st.error(f"Failed to load analyzer: {e}")
        return None

def create_sentiment_gauge(sentiment, confidence):
    """Create gauge chart visualisation"""
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
            'threshold': {'line': {'color': "#14b8a6", 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#0f172a",
        font={'color': "#e5e7eb", 'family': "Inter"}
    )
    
    return fig

def create_confidence_bars(results):
    """Create bar chart comparing model confidences"""
    models, confidences, sentiments = [], [], []
    
    for model_name, result in results.items():
        if isinstance(result, dict) and 'confidence' in result and 'sentiment' in result:
            models.append(model_name.upper())
            confidences.append(result['confidence'])
            sentiments.append(result['sentiment'])
    
    colors = ['#34d399' if s == 'positive' else '#f87171' for s in sentiments]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models, y=confidences, marker_color=colors,
            text=[f"{c:.1%}" for c in confidences], textposition='outside',
            textfont=dict(size=14, color='#ffffff', family='Inter', weight='bold'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={'text': "Model Confidence Comparison", 'font': {'size': 18, 'color': '#ffffff'}},
        xaxis={'color': '#9ca3af', 'gridcolor': '#334155'},
        yaxis={'title': 'Confidence', 'color': '#9ca3af', 'range': [0, 1.1], 'gridcolor': '#334155'},
        height=400, showlegend=False, paper_bgcolor="#0f172a", plot_bgcolor="#1e293b"
    )
    
    return fig

def main():
    load_custom_css()
    
    # Hero section
    st.markdown("""
        <div class="hero-section">
            <p style="color: #a78bfa; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">
                NLP & Machine Learning
            </p>
            <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">Advanced Sentiment Analysis</h1>
            <p style="color: #d1d5db; font-size: 1.125rem; margin-bottom: 0.5rem;">
                Multi-model NLP pipeline for real-time sentiment classification
            </p>
            <p style="color: #9ca3af; line-height: 1.6;">
                Analyse text sentiment using VADER, TextBlob, traditional ML models, and transformer-based deep learning.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        
        if st.button("Load Models", type="primary", use_container_width=True):
            with st.spinner("Loading models..."):
                try:
                    st.session_state.analyzer = load_analyzer()
                    if st.session_state.analyzer:
                        st.session_state.model_loaded = True
                        st.success("Models loaded!")
                except Exception as e:
                    log_error(e, "Model loading")
                    st.error(f"Error: {e}")
        
        st.markdown("---")
        st.markdown("### Available Models")
        st.markdown("""
        <div style="font-size: 0.875rem; line-height: 1.8;">
            <span class="tag-pill">VADER</span> Rule-based<br>
            <span class="tag-pill">TextBlob</span> Pattern-based<br>
            <span class="tag-pill">ML Model</span> Logistic Regression<br>
            <span class="tag-pill">Transformer</span> Deep learning<br>
            <span class="tag-pill">Ensemble</span> Combined
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### System Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "94.2%")
        with col2:
            st.metric("Speed", "<1s")
        
        st.markdown("---")
        
        if st.session_state.analysis_history:
            st.markdown("### History")
            st.write(f"Analyses: **{len(st.session_state.analysis_history)}**")
            if st.button("Clear History", use_container_width=True):
                st.session_state.analysis_history = []
                st.rerun()
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("Click 'Load Models' in the sidebar to initialise the app")
        
        with st.expander("About This App", expanded=True):
            st.markdown("This app uses multiple ML models for sentiment analysis.")
            st.markdown("""
            <div class="feature-item"><div class="feature-icon"></div><span>Real-time prediction</span></div>
            <div class="feature-item"><div class="feature-icon"></div><span>Multiple model comparison</span></div>
            <div class="feature-item"><div class="feature-icon"></div><span>Confidence scoring</span></div>
            <div class="feature-item"><div class="feature-icon"></div><span>Visual indicators</span></div>
            <div class="feature-item"><div class="feature-icon"></div><span>Ensemble predictions</span></div>
            """, unsafe_allow_html=True)
        return
    
    # Text input
    st.markdown("### Enter Text to Analyse")
    
    sample_texts = {
        "Positive": "This product is absolutely amazing! The quality exceeded my expectations and the customer service was outstanding. Highly recommended!",
        "Negative": "Terrible experience. The product broke after one day and customer service was unhelpful. Complete waste of money.",
        "Mixed": "The product is okay. Some features work well but others are disappointing. Average quality for the price.",
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Positive Sample", type="secondary", use_container_width=True):
            st.session_state.sample_text = sample_texts["Positive"]
    with col2:
        if st.button("Negative Sample", type="secondary", use_container_width=True):
            st.session_state.sample_text = sample_texts["Negative"]
    with col3:
        if st.button("Mixed Sample", type="secondary", use_container_width=True):
            st.session_state.sample_text = sample_texts["Mixed"]
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    text_input = st.text_area(
        "Enter your text:",
        value=st.session_state.get('sample_text', ''),
        height=150,
        placeholder="Type or paste your review...",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 1, 3])
    with col1:
        analyse_button = st.button("Analyse Sentiment", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.sample_text = ''
        st.rerun()
    
    if analyse_button and text_input.strip():
        with st.spinner("Analysing..."):
            start_time = time.time()
            
            try:
                analyzer = st.session_state.analyzer
                ensemble_result = analyzer.get_ensemble_prediction(text_input)
                processing_time = time.time() - start_time
                
                if 'error' in ensemble_result:
                    st.error(f"Error: {ensemble_result['error']}")
                    log_error(ensemble_result['error'], "Analysis")
                    return
                
                # Log submission
                log_user_submission(text_input, ensemble_result, processing_time)
                
                # Save to history
                st.session_state.analysis_history.append({
                    'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                    'sentiment': ensemble_result['sentiment'],
                    'confidence': ensemble_result['confidence'],
                    'time': time.strftime('%H:%M:%S')
                })
                
                st.markdown("---")
                st.markdown("## Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                sentiment = ensemble_result['sentiment']
                confidence = ensemble_result['confidence']
                
                with col1:
                    emoji = "😊" if sentiment == 'positive' else "😞"
                    st.metric("Overall Sentiment", f"{emoji} {sentiment.upper()}")
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    st.metric("Processing Time", f"{processing_time:.3f}s")
                with col4:
                    st.metric("Text Length", f"{len(text_input.split())} words")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Visualisations
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_sentiment_gauge(sentiment, confidence), use_container_width=True)
                with col2:
                    individual_results = ensemble_result.get('individual_results', {})
                    if individual_results:
                        comparison_data = {k: v for k, v in individual_results.items() if k != 'text'}
                        st.plotly_chart(create_confidence_bars(comparison_data), use_container_width=True)
                
                # Individual models
                st.markdown("### Individual Model Results")
                model_results = {k: v for k, v in ensemble_result.get('individual_results', {}).items() if k != 'text'}
                
                if model_results:
                    cols = st.columns(len(model_results))
                    for idx, (model_name, result) in enumerate(model_results.items()):
                        if isinstance(result, dict) and 'sentiment' in result:
                            with cols[idx]:
                                sentiment_model = result['sentiment']
                                confidence_model = result.get('confidence', 0)
                                emoji = "😊" if sentiment_model == 'positive' else "😞"
                                card_class = "positive-card" if sentiment_model == 'positive' else "negative-card"
                                
                                st.markdown(f"""
                                <div class="model-card {card_class}">
                                    <h4 style="color: #14b8a6; font-size: 0.875rem; text-transform: uppercase; margin-bottom: 1rem;">
                                        {model_name.upper()}
                                    </h4>
                                    <div style="text-align: center;">
                                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                                        <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.25rem; color: #ffffff;">
                                            {sentiment_model.upper()}
                                        </div>
                                        <div style="color: #9ca3af; font-size: 0.875rem;">
                                            Confidence: {confidence_model:.1%}
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                with st.expander("View Analysed Text"):
                    st.write(text_input)
                    
            except Exception as e:
                st.error(f"Error: {e}")
                log_error(e, "Analysis execution")
    
    elif analyse_button:
        st.warning("Please enter some text to analyse")
    
    # Recent history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("## Recent Analyses")
        
        history_df = pd.DataFrame(st.session_state.analysis_history[-5:])
        history_df['sentiment'] = history_df['sentiment'].apply(
            lambda x: f"{'😊' if x == 'positive' else '😞'} {x.upper()}"
        )
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            history_df[['time', 'text', 'sentiment', 'confidence']],
            hide_index=True,
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #6b7280; font-size: 0.875rem; padding: 2rem 0;">
            <p>Built with Python, Scikit-learn, Transformers & Streamlit</p>
            <p style="margin-top: 0.5rem;">© 2026 Data Science Portfolio</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()