"""
Streamlit app for sentiment analysis.
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.sentiment_analyzer import SentimentAnalyzer
from src.preprocessor import TextPreprocessor
from src.data_collector import DataCollector

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
    st.session_state.model_loaded = False
    st.session_state.analysis_history = []

@st.cache_resource
def load_analyzer():
    """Load the sentiment analyzer (cached)."""
    analyzer = SentimentAnalyzer()
    
    # Try to load pre-trained model
    if not analyzer.load_ml_model():
        # Train a new model if none exists
        try:
            collector = DataCollector()
            preprocessor = TextPreprocessor()
            df = collector.get_combined_dataset()
            df_processed = preprocessor.preprocess_dataframe(df)
            analyzer.train_ml_model(df_processed)
        except Exception as e:
            st.warning(f"Could not train ML model: {e}")
    
    return analyzer

def create_sentiment_gauge(sentiment, confidence):
    """Create a gauge chart for sentiment visualization."""
    
    # Map sentiment to gauge value
    if sentiment == 'positive':
        value = 50 + (confidence * 50)  # 50-100
        color = "#4ecdc4"
    else:
        value = 50 - (confidence * 50)  # 0-50
        color = "#ff6b6b"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score", 'font': {'size': 20}},
        number={'suffix': "", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 50], 'color': '#ffe6cc'},
                {'range': [50, 70], 'color': '#e6f7ff'},
                {'range': [70, 100], 'color': '#ccf5e6'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        font={'color': "darkgray", 'family': "Arial"}
    )
    
    return fig

def create_confidence_bars(results):
    """Create bar chart comparing model confidences."""
    
    models = []
    confidences = []
    sentiments = []
    
    for model_name, result in results.items():
        if isinstance(result, dict) and 'confidence' in result and 'sentiment' in result:
            models.append(model_name.upper())
            confidences.append(result['confidence'])
            sentiments.append(result['sentiment'])
    
    colors = ['#4ecdc4' if s == 'positive' else '#ff6b6b' for s in sentiments]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.2%}" for c in confidences],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Model Confidence Comparison",
        xaxis_title="Model",
        yaxis_title="Confidence",
        yaxis_range=[0, 1.1],
        height=400,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    return fig

def main():
    # Header
    st.title("🎭 Sentiment Analysis App")
    st.markdown("### Analyse the sentiment of any text using multiple AI models")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model loading
        if st.button("🔄 Load/Reload Models", type="primary"):
            with st.spinner("Loading models..."):
                st.session_state.analyzer = load_analyzer()
                st.session_state.model_loaded = True
                st.success("✅ Models loaded successfully!")
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Available Models")
        st.markdown("""
        - **VADER**: Rule-based sentiment analysis
        - **TextBlob**: Pattern-based analysis
        - **ML Model**: Trained Logistic Regression
        - **Transformer**: Deep learning model (if available)
        - **Ensemble**: Weighted combination of all models
        """)
        
        st.markdown("---")
        
        # Analysis history
        if st.session_state.analysis_history:
            st.subheader("📜 Analysis History")
            st.write(f"Total analyses: {len(st.session_state.analysis_history)}")
            
            if st.button("Clear History"):
                st.session_state.analysis_history = []
                st.rerun()
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("👈 Click 'Load/Reload Models' in the sidebar to initialise the app")
        
        # Show example
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            This sentiment analysis app uses multiple machine learning models to determine whether 
            text expresses positive or negative sentiment.
            
            **Features:**
            - Real-time sentiment prediction
            - Multiple model comparison
            - Confidence scores
            - Visual sentiment indicators
            
            **How to use:**
            1. Click 'Load/Reload Models' in the sidebar
            2. Enter your text in the text area
            3. Click 'Analyse Sentiment'
            4. View the results and model comparisons
            """)
        
        return
    
    # Text input
    st.subheader("📝 Enter Text to Analyse")
    
    # Sample texts
    sample_texts = {
        "Positive Review": "This product is absolutely amazing! The quality exceeded my expectations and the customer service was outstanding. Highly recommended!",
        "Negative Review": "Terrible experience. The product broke after one day and customer service was unhelpful. Complete waste of money.",
        "Mixed Review": "The product is okay. Some features work well but others are disappointing. Average quality for the price.",
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 Sample: Positive"):
            st.session_state.sample_text = sample_texts["Positive Review"]
    with col2:
        if st.button("📌 Sample: Negative"):
            st.session_state.sample_text = sample_texts["Negative Review"]
    with col3:
        if st.button("📌 Sample: Mixed"):
            st.session_state.sample_text = sample_texts["Mixed Review"]
    
    # Text area
    text_input = st.text_area(
        "Enter your text here:",
        value=st.session_state.get('sample_text', ''),
        height=150,
        placeholder="Type or paste your review, comment, or any text you'd like to analyse..."
    )
    
    # Analyse button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        analyse_button = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.sample_text = ''
        st.rerun()
    
    # Perform analysis
    if analyse_button and text_input.strip():
        with st.spinner("Analysing sentiment..."):
            start_time = time.time()
            
            # Get predictions
            analyzer = st.session_state.analyzer
            ensemble_result = analyzer.get_ensemble_prediction(text_input)
            
            processing_time = time.time() - start_time
            
            if 'error' in ensemble_result:
                st.error(f"❌ Error: {ensemble_result['error']}")
                return
            
            # Save to history
            st.session_state.analysis_history.append({
                'text': text_input[:100] + '...' if len(text_input) > 100 else text_input,
                'sentiment': ensemble_result['sentiment'],
                'confidence': ensemble_result['confidence'],
                'time': time.strftime('%H:%M:%S')
            })
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            
            sentiment = ensemble_result['sentiment']
            confidence = ensemble_result['confidence']
            
            with col1:
                emoji = "😊" if sentiment == 'positive' else "😞"
                st.metric(
                    label="Overall Sentiment",
                    value=f"{emoji} {sentiment.upper()}"
                )
            
            with col2:
                st.metric(
                    label="Confidence Score",
                    value=f"{confidence:.2%}"
                )
            
            with col3:
                st.metric(
                    label="Processing Time",
                    value=f"{processing_time:.3f}s"
                )
            
            with col4:
                st.metric(
                    label="Text Length",
                    value=f"{len(text_input.split())} words"
                )
            
            # Visualisations
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment gauge
                gauge_fig = create_sentiment_gauge(sentiment, confidence)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Model comparison
                individual_results = ensemble_result.get('individual_results', {})
                if individual_results:
                    # Remove the 'text' key for cleaner display
                    comparison_data = {k: v for k, v in individual_results.items() if k != 'text'}
                    bars_fig = create_confidence_bars(comparison_data)
                    st.plotly_chart(bars_fig, use_container_width=True)
            
            # Detailed model results
            st.subheader("🔬 Individual Model Results")
            
            individual_results = ensemble_result.get('individual_results', {})
            
            model_cols = st.columns(len([k for k in individual_results.keys() if k != 'text']))
            
            col_idx = 0
            for model_name, result in individual_results.items():
                if model_name == 'text':
                    continue
                
                if isinstance(result, dict) and 'sentiment' in result:
                    with model_cols[col_idx]:
                        sentiment_model = result['sentiment']
                        confidence_model = result.get('confidence', 0)
                        
                        emoji = "😊" if sentiment_model == 'positive' else "😞"
                        color = "#d4edda" if sentiment_model == 'positive' else "#f8d7da"
                        
                        st.markdown(f"""
                        <div style="background-color: {color}; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                            <h4>{model_name.upper()}</h4>
                            <h2>{emoji}</h2>
                            <p style="font-size: 1.2em; margin: 0;"><strong>{sentiment_model.upper()}</strong></p>
                            <p style="margin: 0;">Confidence: {confidence_model:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_idx += 1
            
            # Text preview
            with st.expander("📄 View Analysed Text"):
                st.write(text_input)
    
    elif analyse_button:
        st.warning("⚠️ Please enter some text to analyse")
    
    # Recent history
    if st.session_state.analysis_history:
        st.markdown("---")
        st.subheader("📈 Recent Analyses")
        
        # Create DataFrame from history
        history_df = pd.DataFrame(st.session_state.analysis_history[-5:])  # Last 5
        
        # Format for display
        history_df['sentiment'] = history_df['sentiment'].apply(lambda x: f"{'😊' if x == 'positive' else '😞'} {x.upper()}")
        history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(
            history_df[['time', 'text', 'sentiment', 'confidence']],
            hide_index=True,
            use_container_width=True
        )

if __name__ == "__main__":
    main()