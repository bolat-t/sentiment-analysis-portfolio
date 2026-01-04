"""
Visualization module for sentiment analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SentimentVisualizer:
    """Create visualizations for sentiment analysis results."""
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'viridis'):
        # Set style
        sns.set_style(style)
        plt.style.use('default')
        self.palette = palette
        
        # Create output directory
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, 
                                  title: str = "Sentiment Distribution",
                                  save_path: Optional[str] = None) -> None:
        """Plot distribution of sentiments."""
        plt.figure(figsize=(10, 6))
        
        # Count plot
        sentiment_counts = df['label'].value_counts()
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=['#ff6b6b', '#4ecdc4'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
               autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'])
        plt.title('Sentiment Proportion')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sentiment distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, results: List[Dict], 
                             save_path: Optional[str] = None) -> None:
        """Compare performance of different models."""
        
        # Extract model accuracies (you'll need to implement model evaluation)
        models = ['VADER', 'TextBlob', 'ML Model', 'Transformer']
        accuracies = [0.851, 0.823, 0.887, 0.942]  # Example values
        
        plt.figure(figsize=(12, 8))
        
        # Bar plot
        plt.subplot(2, 2, 1)
        bars = plt.bar(models, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0.8, 1.0)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Example confidence distributions
        plt.subplot(2, 2, 2)
        confidence_data = [
            np.random.beta(2, 2, 1000) for _ in models
        ]
        
        for i, (model, data) in enumerate(zip(models, confidence_data)):
            plt.hist(data, alpha=0.7, label=model, bins=30)
        
        plt.title('Confidence Score Distributions')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Processing time comparison
        plt.subplot(2, 2, 3)
        processing_times = [0.001, 0.002, 0.050, 0.300]  # Example times in seconds
        plt.bar(models, processing_times, color='coral')
        plt.title('Average Processing Time')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        
                # Model agreement matrix
        plt.subplot(2, 2, 4)
        # Example agreement matrix between models
        agreement_matrix = np.array([
            [1.00, 0.78, 0.85, 0.82],
            [0.78, 1.00, 0.73, 0.76],
            [0.85, 0.73, 1.00, 0.89],
            [0.82, 0.76, 0.89, 1.00]
        ])
        
        im = plt.imshow(agreement_matrix, cmap='Blues', vmin=0.7, vmax=1.0)
        plt.title('Model Agreement Matrix')
        plt.xticks(range(len(models)), models, rotation=45)
        plt.yticks(range(len(models)), models)
        
        # Add correlation values
        for i in range(len(models)):
            for j in range(len(models)):
                plt.text(j, i, f'{agreement_matrix[i, j]:.2f}', 
                        ha='center', va='center')
        
        plt.colorbar(im, shrink=0.8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def create_word_clouds(self, df: pd.DataFrame, 
                          text_column: str = 'text_cleaned',
                          save_path: Optional[str] = None) -> None:
        """Create word clouds for positive and negative sentiments."""
        
        plt.figure(figsize=(15, 8))
        
        # Positive sentiment word cloud
        positive_texts = ' '.join(df[df['label'] == 'positive'][text_column].astype(str))
        
        plt.subplot(1, 2, 1)
        if positive_texts.strip():
            wordcloud_pos = WordCloud(
                width=600, height=400, 
                background_color='white',
                colormap='Greens',
                max_words=100
            ).generate(positive_texts)
            
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.title('Positive Sentiment Word Cloud', fontsize=16, pad=20)
        else:
            plt.text(0.5, 0.5, 'No positive text data', 
                    transform=plt.gca().transAxes, ha='center', va='center')
        plt.axis('off')
        
        # Negative sentiment word cloud
        negative_texts = ' '.join(df[df['label'] == 'negative'][text_column].astype(str))
        
        plt.subplot(1, 2, 2)
        if negative_texts.strip():
            wordcloud_neg = WordCloud(
                width=600, height=400,
                background_color='white',
                colormap='Reds',
                max_words=100
            ).generate(negative_texts)
            
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.title('Negative Sentiment Word Cloud', fontsize=16, pad=20)
        else:
            plt.text(0.5, 0.5, 'No negative text data', 
                    transform=plt.gca().transAxes, ha='center', va='center')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Word clouds saved to {save_path}")
        
        plt.show()
    
    def plot_confidence_analysis(self, results: List[Dict],
                               save_path: Optional[str] = None) -> None:
        """Plot confidence analysis across different models."""
        
        plt.figure(figsize=(15, 10))
        
        # Extract confidence scores for each model
        vader_conf = [r['vader']['confidence'] for r in results if 'vader' in r]
        textblob_conf = [r['textblob']['confidence'] for r in results if 'textblob' in r]
        
        # Confidence distribution
        plt.subplot(2, 3, 1)
        plt.hist(vader_conf, alpha=0.7, label='VADER', bins=20, color='blue')
        plt.hist(textblob_conf, alpha=0.7, label='TextBlob', bins=20, color='red')
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Box plot for confidence comparison
        plt.subplot(2, 3, 2)
        conf_data = [vader_conf, textblob_conf]
        plt.boxplot(conf_data, labels=['VADER', 'TextBlob'])
        plt.title('Confidence Score Box Plot')
        plt.ylabel('Confidence')
        
        # Confidence vs Text Length
        plt.subplot(2, 3, 3)
        text_lengths = [len(r['text'].split()) for r in results]
        plt.scatter(text_lengths, vader_conf, alpha=0.6, label='VADER', color='blue')
        plt.scatter(text_lengths, textblob_conf, alpha=0.6, label='TextBlob', color='red')
        plt.xlabel('Text Length (words)')
        plt.ylabel('Confidence')
        plt.title('Confidence vs Text Length')
        plt.legend()
        
        # Model agreement visualization
        plt.subplot(2, 3, 4)
        vader_sentiments = [1 if r['vader']['sentiment'] == 'positive' else 0 for r in results if 'vader' in r]
        textblob_sentiments = [1 if r['textblob']['sentiment'] == 'positive' else 0 for r in results if 'textblob' in r]
        
        agreement = [1 if v == t else 0 for v, t in zip(vader_sentiments, textblob_sentiments)]
        agreement_rate = np.mean(agreement)
        
        plt.bar(['Agreement', 'Disagreement'], 
               [agreement_rate, 1-agreement_rate], 
               color=['green', 'red'])
        plt.title(f'VADER vs TextBlob Agreement: {agreement_rate:.2%}')
        plt.ylabel('Proportion')
        
        # High vs Low confidence comparison
        plt.subplot(2, 3, 5)
        high_conf_threshold = 0.7
        vader_high_conf = [conf for conf in vader_conf if conf > high_conf_threshold]
        vader_low_conf = [conf for conf in vader_conf if conf <= high_conf_threshold]
        
        plt.hist([vader_high_conf, vader_low_conf], 
                label=[f'High Confidence (>{high_conf_threshold})', f'Low Confidence (<={high_conf_threshold})'],
                alpha=0.7, bins=15)
        plt.title('VADER High vs Low Confidence')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Average confidence by sentiment
        plt.subplot(2, 3, 6)
        pos_conf = [r['vader']['confidence'] for r in results if r['vader']['sentiment'] == 'positive']
        neg_conf = [r['vader']['confidence'] for r in results if r['vader']['sentiment'] == 'negative']
        
        plt.bar(['Positive', 'Negative'], 
               [np.mean(pos_conf) if pos_conf else 0, np.mean(neg_conf) if neg_conf else 0],
               color=['lightgreen', 'lightcoral'])
        plt.title('Average Confidence by Sentiment')
        plt.ylabel('Average Confidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence analysis plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, results: List[Dict]) -> go.Figure:
        """Create an interactive Plotly dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Model Confidence', 
                          'Text Length vs Confidence', 'Model Agreement'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Extract data
        sentiments = [r['vader']['sentiment'] for r in results if 'vader' in r]
        confidences = [r['vader']['confidence'] for r in results if 'vader' in r]
        text_lengths = [len(r['text'].split()) for r in results]
        
        # Sentiment distribution pie chart
        sentiment_counts = pd.Series(sentiments).value_counts()
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                  name="Sentiment Distribution"),
            row=1, col=1
        )
        
        # Confidence histogram
        fig.add_trace(
            go.Histogram(x=confidences, name="Confidence Distribution", nbinsx=20),
            row=1, col=2
        )
        
        # Text length vs confidence scatter
        fig.add_trace(
            go.Scatter(x=text_lengths, y=confidences, mode='markers',
                      name="Length vs Confidence", opacity=0.6),
            row=2, col=1
        )
        
        # Model agreement
        if len(results) > 0 and 'textblob' in results[0]:
            vader_sent = [1 if r['vader']['sentiment'] == 'positive' else 0 for r in results if 'vader' in r]
            textblob_sent = [1 if r['textblob']['sentiment'] == 'positive' else 0 for r in results if 'textblob' in r]
            
            agreement = np.mean([1 if v == t else 0 for v, t in zip(vader_sent, textblob_sent)])
            
            fig.add_trace(
                go.Bar(x=['Agreement', 'Disagreement'], 
                      y=[agreement, 1-agreement],
                      name="Model Agreement"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Sentiment Analysis Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_all_plots(self, df: pd.DataFrame, results: List[Dict]) -> None:
        """Save all visualization plots."""
        logger.info("Generating and saving all plots...")
        
        # Create output directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate all plots
        self.plot_sentiment_distribution(df, save_path=plots_dir / "sentiment_distribution.png")
        self.plot_model_comparison(results, save_path=plots_dir / "model_comparison.png")
        self.create_word_clouds(df, save_path=plots_dir / "word_clouds.png")
        self.plot_confidence_analysis(results, save_path=plots_dir / "confidence_analysis.png")
        
        # Save interactive dashboard
        if results:
            dashboard = self.create_interactive_dashboard(results)
            dashboard.write_html(plots_dir / "interactive_dashboard.html")
            logger.info("Interactive dashboard saved as HTML")
        
        logger.info("All plots saved successfully!")

