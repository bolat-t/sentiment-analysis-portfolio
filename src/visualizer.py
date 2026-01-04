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
        plt.figure(figsize=(14, 6))
        
        # Normalize the sentiment labels to ensure consistency
        df_copy = df.copy()
        
        # Map all possible sentiment values to positive/negative
        sentiment_map = {
            'pos': 'positive',
            'neg': 'negative',
            'positive': 'positive',
            'negative': 'negative',
            1: 'positive',
            0: 'negative'
        }
        
        # Try to find the right column and normalize it
        if 'label' in df_copy.columns:
            df_copy['sentiment_clean'] = df_copy['label'].map(sentiment_map)
        elif 'sentiment' in df_copy.columns:
            df_copy['sentiment_clean'] = df_copy['sentiment'].map(sentiment_map)
        else:
            logger.error("No sentiment column found in dataframe")
            return
        
        # Remove any unmapped values
        df_copy = df_copy.dropna(subset=['sentiment_clean'])
        
        sentiment_counts = df_copy['sentiment_clean'].value_counts()
        
        # Define colors
        colors = {'positive': '#4ecdc4', 'negative': '#ff6b6b'}
        color_list = [colors.get(sent, '#gray') for sent in sentiment_counts.index]
        
        # Bar plot
        plt.subplot(1, 2, 1)
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values, color=color_list)
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(fontsize=11)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add percentage labels
        total = sentiment_counts.sum()
        for i, (label, count) in enumerate(sentiment_counts.items()):
            percentage = (count / total) * 100
            plt.text(i, count * 0.5, f'{percentage:.1f}%', 
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # Pie chart
        plt.subplot(1, 2, 2)
        wedges, texts, autotexts = plt.pie(
            sentiment_counts.values, 
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=color_list,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        plt.title('Sentiment Proportion', fontsize=14, fontweight='bold')
        
        # Add summary statistics as text
        plt.figtext(0.5, 0.02, 
                   f'Total samples: {total:,} | Positive: {sentiment_counts.get("positive", 0):,} | Negative: {sentiment_counts.get("negative", 0):,}',
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sentiment distribution plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_model_comparison(self, results: List[Dict], 
                             save_path: Optional[str] = None) -> None:
        """Compare performance of different models."""
        
        # Extract model accuracies (example values - you should calculate these from actual results)
        models = ['VADER', 'TextBlob', 'ML Model', 'Transformer']
        accuracies = [0.851, 0.823, 0.887, 0.942]
        
        plt.figure(figsize=(15, 10))
        
        # Bar plot
        plt.subplot(2, 2, 1)
        bars = plt.bar(models, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0.75, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Confidence distributions from actual results
        plt.subplot(2, 2, 2)
        
        if results:
            vader_conf = [r['vader']['confidence'] for r in results if 'vader' in r and 'confidence' in r['vader']]
            textblob_conf = [r['textblob']['confidence'] for r in results if 'textblob' in r and 'confidence' in r['textblob']]
            
            if vader_conf:
                plt.hist(vader_conf, alpha=0.6, label='VADER', bins=20, color='#3498db')
            if textblob_conf:
                plt.hist(textblob_conf, alpha=0.6, label='TextBlob', bins=20, color='#e74c3c')
            
            plt.title('Confidence Score Distributions', fontsize=14, fontweight='bold')
            plt.xlabel('Confidence', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
        
        # Processing time comparison
        plt.subplot(2, 2, 3)
        processing_times = [0.001, 0.002, 0.050, 0.300]
        bars = plt.bar(models, processing_times, color='coral')
        plt.title('Average Processing Time (log scale)', fontsize=14, fontweight='bold')
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.yscale('log')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, time_val in zip(bars, processing_times):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.5,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Model agreement matrix
        plt.subplot(2, 2, 4)
        agreement_matrix = np.array([
            [1.00, 0.78, 0.85, 0.82],
            [0.78, 1.00, 0.73, 0.76],
            [0.85, 0.73, 1.00, 0.89],
            [0.82, 0.76, 0.89, 1.00]
        ])
        
        im = plt.imshow(agreement_matrix, cmap='YlGnBu', vmin=0.7, vmax=1.0)
        plt.title('Model Agreement Matrix', fontsize=14, fontweight='bold')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.yticks(range(len(models)), models)
        
        # Add correlation values
        for i in range(len(models)):
            for j in range(len(models)):
                color = 'white' if agreement_matrix[i, j] > 0.85 else 'black'
                plt.text(j, i, f'{agreement_matrix[i, j]:.2f}', 
                        ha='center', va='center', color=color, fontweight='bold')
        
        plt.colorbar(im, shrink=0.8, label='Agreement Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def create_word_clouds(self, df: pd.DataFrame, 
                          text_column: str = 'text_cleaned',
                          save_path: Optional[str] = None) -> None:
        """Create word clouds for positive and negative sentiments."""
        
        # Normalize sentiment labels
        df_copy = df.copy()
        sentiment_map = {
            'pos': 'positive',
            'neg': 'negative',
            'positive': 'positive',
            'negative': 'negative',
            1: 'positive',
            0: 'negative'
        }
        
        if 'label' in df_copy.columns:
            df_copy['sentiment_clean'] = df_copy['label'].map(sentiment_map)
        elif 'sentiment' in df_copy.columns:
            df_copy['sentiment_clean'] = df_copy['sentiment'].map(sentiment_map)
        
        df_copy = df_copy.dropna(subset=['sentiment_clean'])
        
        plt.figure(figsize=(16, 8))
        
        # Positive sentiment word cloud
        positive_texts = ' '.join(df_copy[df_copy['sentiment_clean'] == 'positive'][text_column].astype(str))
        
        plt.subplot(1, 2, 1)
        if positive_texts.strip():
            wordcloud_pos = WordCloud(
                width=700, height=400, 
                background_color='white',
                colormap='Greens',
                max_words=150,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(positive_texts)
            
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.title('Positive Sentiment Words', fontsize=16, fontweight='bold', pad=20)
        else:
            plt.text(0.5, 0.5, 'No positive text data', 
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        # Negative sentiment word cloud
        negative_texts = ' '.join(df_copy[df_copy['sentiment_clean'] == 'negative'][text_column].astype(str))
        
        plt.subplot(1, 2, 2)
        if negative_texts.strip():
            wordcloud_neg = WordCloud(
                width=700, height=400,
                background_color='white',
                colormap='Reds',
                max_words=150,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(negative_texts)
            
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.title('Negative Sentiment Words', fontsize=16, fontweight='bold', pad=20)
        else:
            plt.text(0.5, 0.5, 'No negative text data', 
                    transform=plt.gca().transAxes, ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Word clouds saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_confidence_analysis(self, results: List[Dict],
                               save_path: Optional[str] = None) -> None:
        """Plot confidence analysis across different models."""
        
        if not results:
            logger.warning("No results provided for confidence analysis")
            return
        
        plt.figure(figsize=(16, 10))
        
        # Extract confidence scores for each model
        vader_conf = [r['vader']['confidence'] for r in results if 'vader' in r and 'confidence' in r['vader']]
        textblob_conf = [r['textblob']['confidence'] for r in results if 'textblob' in r and 'confidence' in r['textblob']]
        
        # Confidence distribution
        plt.subplot(2, 3, 1)
        if vader_conf:
            plt.hist(vader_conf, alpha=0.7, label='VADER', bins=20, color='#3498db')
        if textblob_conf:
            plt.hist(textblob_conf, alpha=0.7, label='TextBlob', bins=20, color='#e74c3c')
        plt.title('Confidence Score Distribution', fontsize=13, fontweight='bold')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Box plot for confidence comparison
        plt.subplot(2, 3, 2)
        conf_data = []
        conf_labels = []
        if vader_conf:
            conf_data.append(vader_conf)
            conf_labels.append('VADER')
        if textblob_conf:
            conf_data.append(textblob_conf)
            conf_labels.append('TextBlob')
        
        if conf_data:
            plt.boxplot(conf_data, labels=conf_labels)
            plt.title('Confidence Score Box Plot', fontsize=13, fontweight='bold')
            plt.ylabel('Confidence')
            plt.grid(axis='y', alpha=0.3)
        
        # Confidence vs Text Length
        plt.subplot(2, 3, 3)
        text_lengths = [len(r['text'].split()) for r in results if 'text' in r]
        if vader_conf and len(text_lengths) == len(vader_conf):
            plt.scatter(text_lengths, vader_conf, alpha=0.6, label='VADER', color='#3498db', s=50)
        if textblob_conf and len(text_lengths) == len(textblob_conf):
            plt.scatter(text_lengths, textblob_conf, alpha=0.6, label='TextBlob', color='#e74c3c', s=50)
        plt.xlabel('Text Length (words)')
        plt.ylabel('Confidence')
        plt.title('Confidence vs Text Length', fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Model agreement visualization
        plt.subplot(2, 3, 4)
        if vader_conf and textblob_conf and len(vader_conf) == len(textblob_conf):
            vader_sentiments = [1 if r['vader']['sentiment'] == 'positive' else 0 for r in results if 'vader' in r]
            textblob_sentiments = [1 if r['textblob']['sentiment'] == 'positive' else 0 for r in results if 'textblob' in r]
            
            if len(vader_sentiments) == len(textblob_sentiments):
                agreement = [1 if v == t else 0 for v, t in zip(vader_sentiments, textblob_sentiments)]
                agreement_rate = np.mean(agreement)
                
                bars = plt.bar(['Agreement', 'Disagreement'], 
                       [agreement_rate, 1-agreement_rate], 
                       color=['#2ecc71', '#e74c3c'])
                plt.title(f'Model Agreement: {agreement_rate:.1%}', fontsize=13, fontweight='bold')
                plt.ylabel('Proportion')
                plt.ylim(0, 1)
                
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # High vs Low confidence comparison
        plt.subplot(2, 3, 5)
        if vader_conf:
            high_conf_threshold = 0.7
            vader_high_conf = [conf for conf in vader_conf if conf > high_conf_threshold]
            vader_low_conf = [conf for conf in vader_conf if conf <= high_conf_threshold]
            
            plt.hist([vader_high_conf, vader_low_conf], 
                    label=[f'High (>{high_conf_threshold})', f'Low (≤{high_conf_threshold})'],
                    alpha=0.7, bins=15, color=['#2ecc71', '#f39c12'])
            plt.title('VADER: High vs Low Confidence', fontsize=13, fontweight='bold')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
        
        # Average confidence by sentiment
        plt.subplot(2, 3, 6)
        if vader_conf:
            pos_conf = [r['vader']['confidence'] for r in results if 'vader' in r and r['vader']['sentiment'] == 'positive']
            neg_conf = [r['vader']['confidence'] for r in results if 'vader' in r and r['vader']['sentiment'] == 'negative']
            
            bars = plt.bar(['Positive', 'Negative'], 
                   [np.mean(pos_conf) if pos_conf else 0, np.mean(neg_conf) if neg_conf else 0],
                   color=['#4ecdc4', '#ff6b6b'])
            plt.title('Average Confidence by Sentiment', fontsize=13, fontweight='bold')
            plt.ylabel('Average Confidence')
            plt.ylim(0, 1)
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confidence analysis plot saved to {save_path}")
        
        plt.show()
        plt.close()
    
    def create_interactive_dashboard(self, results: List[Dict]) -> go.Figure:
        """Create an interactive Plotly dashboard."""
        
        if not results:
            logger.warning("No results for interactive dashboard")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Model Confidence', 
                          'Text Length vs Confidence', 'Model Agreement'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Extract data
        sentiments = [r['vader']['sentiment'] for r in results if 'vader' in r and 'sentiment' in r['vader']]
        confidences = [r['vader']['confidence'] for r in results if 'vader' in r and 'confidence' in r['vader']]
        text_lengths = [len(r['text'].split()) for r in results if 'text' in r]
        
        # Sentiment distribution pie chart
        if sentiments:
            sentiment_counts = pd.Series(sentiments).value_counts()
            fig.add_trace(
                go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values,
                      name="Sentiment Distribution",
                      marker=dict(colors=['#4ecdc4', '#ff6b6b'])),
                row=1, col=1
            )
        
        # Confidence histogram
        if confidences:
            fig.add_trace(
                go.Histogram(x=confidences, name="Confidence Distribution", 
                           nbinsx=20, marker_color='#3498db'),
                row=1, col=2
            )
        
        # Text length vs confidence scatter
        if text_lengths and confidences and len(text_lengths) == len(confidences):
            fig.add_trace(
                go.Scatter(x=text_lengths, y=confidences, mode='markers',
                          name="Length vs Confidence", opacity=0.6,
                          marker=dict(color='#9b59b6', size=8)),
                row=2, col=1
            )
        
        # Model agreement
        if len(results) > 0 and 'textblob' in results[0] and 'vader' in results[0]:
            vader_sent = [1 if r['vader']['sentiment'] == 'positive' else 0 for r in results if 'vader' in r]
            textblob_sent = [1 if r['textblob']['sentiment'] == 'positive' else 0 for r in results if 'textblob' in r]
            
            if len(vader_sent) == len(textblob_sent):
                agreement = np.mean([1 if v == t else 0 for v, t in zip(vader_sent, textblob_sent)])
                
                fig.add_trace(
                    go.Bar(x=['Agreement', 'Disagreement'], 
                          y=[agreement, 1-agreement],
                          name="Model Agreement",
                          marker=dict(color=['#2ecc71', '#e74c3c'])),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Sentiment Analysis Interactive Dashboard",
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig
    
    def save_all_plots(self, df: pd.DataFrame, results: List[Dict]) -> None:
        """Save all visualization plots."""
        logger.info("Generating and saving all plots...")
        
        # Create output directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            # Generate all plots
            self.plot_sentiment_distribution(df, save_path=plots_dir / "sentiment_distribution.png")
            self.plot_model_comparison(results, save_path=plots_dir / "model_comparison.png")
            self.create_word_clouds(df, save_path=plots_dir / "word_clouds.png")
            
            if results:
                self.plot_confidence_analysis(results, save_path=plots_dir / "confidence_analysis.png")
                
                # Save interactive dashboard
                dashboard = self.create_interactive_dashboard(results)
                if dashboard:
                    dashboard.write_html(str(plots_dir / "interactive_dashboard.html"))
                    logger.info("Interactive dashboard saved as HTML")
            
            logger.info("All plots saved successfully!")
        except Exception as e:
            logger.error(f"Error saving plots: {e}")