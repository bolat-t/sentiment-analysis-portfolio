"""
Main sentiment analysis module implementing multiple approaches.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pickle
import logging
from typing import Dict, List, Tuple, Union
from pathlib import Path

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Multi-model sentiment analysis system."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None
        self.ml_model = None
        self.transformer_model = None
        
        # Load pre-trained transformer if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("Transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
    
    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Convert compound score to binary sentiment
        sentiment = "positive" if scores['compound'] >= 0 else "negative"
        confidence = abs(scores['compound'])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'scores': scores
        }
    
    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        sentiment = "positive" if polarity >= 0 else "negative"
        confidence = abs(polarity)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'polarity': polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def transformer_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment using transformer model."""
        if not self.transformer_model:
            return {"error": "Transformer model not available"}
        
        try:
            results = self.transformer_model(text)[0]
            
            # Find the highest scoring sentiment
            best_result = max(results, key=lambda x: x['score'])
            
            # Map labels to positive/negative
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive',
                'NEGATIVE': 'negative',
                'POSITIVE': 'positive'
            }
            
            sentiment = label_mapping.get(best_result['label'], best_result['label'].lower())
            
            # Filter out neutral for binary classification
            if sentiment == 'neutral':
                sentiment = 'positive' if best_result['score'] > 0.5 else 'negative'
            
            return {
                'sentiment': sentiment,
                'confidence': best_result['score'],
                'all_scores': results
            }
            
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            return {"error": str(e)}
    
    def train_ml_model(self, df: pd.DataFrame, text_column: str = 'text_cleaned') -> Dict[str, float]:
        """Train traditional ML model for sentiment analysis."""
        logger.info("Training ML model...")
        
        # Prepare data
        X = df[text_column].values
        y = df['sentiment'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        # Train model
        self.ml_model = LogisticRegression(random_state=42, max_iter=1000)
        self.ml_model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        self.save_ml_model()
        
        logger.info(f"ML Model trained. Accuracy: {accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def ml_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Predict sentiment using trained ML model."""
        if not self.ml_model or not self.tfidf_vectorizer:
            return {"error": "ML model not trained"}
        
        try:
            # Vectorize text
            text_tfidf = self.tfidf_vectorizer.transform([text])
            
            # Predict
            prediction = self.ml_model.predict(text_tfidf)[0]
            probability = self.ml_model.predict_proba(text_tfidf)[0]
            
            sentiment = "positive" if prediction == 1 else "negative"
            confidence = max(probability)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'probabilities': {
                    'negative': probability[0],
                    'positive': probability[1]
                }
            }
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return {"error": str(e)}
    
    def analyze_text(self, text: str) -> Dict[str, Dict]:
        """Analyze text using all available models."""
        results = {
            'text': text,
            'vader': self.vader_sentiment(text),
            'textblob': self.textblob_sentiment(text),
            'ml_model': self.ml_sentiment(text)
        }
        
        if TRANSFORMERS_AVAILABLE:
            results['transformer'] = self.transformer_sentiment(text)
        
        return results
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        logger.info(f"Analyzing {len(texts)} texts...")
        return [self.analyze_text(text) for text in texts]
    
    def get_ensemble_prediction(self, text: str) -> Dict[str, Union[str, float]]:
        """Get ensemble prediction from all models."""
        results = self.analyze_text(text)
        
        # Collect predictions
        predictions = []
        weights = []
        
        # VADER
        if 'vader' in results and 'sentiment' in results['vader']:
            predictions.append(1 if results['vader']['sentiment'] == 'positive' else 0)
            weights.append(results['vader']['confidence'])
        
        # TextBlob
        if 'textblob' in results and 'sentiment' in results['textblob']:
            predictions.append(1 if results['textblob']['sentiment'] == 'positive' else 0)
            weights.append(results['textblob']['confidence'])
        
        # ML Model
        if 'ml_model' in results and 'sentiment' in results['ml_model']:
            predictions.append(1 if results['ml_model']['sentiment'] == 'positive' else 0)
            weights.append(results['ml_model']['confidence'])
        
        # Transformer
        if 'transformer' in results and 'sentiment' in results['transformer']:
            predictions.append(1 if results['transformer']['sentiment'] == 'positive' else 0)
            weights.append(results['transformer']['confidence'])
        
        if not predictions:
            return {"error": "No valid predictions"}
        
        # Weighted average
        weighted_prediction = np.average(predictions, weights=weights)
        ensemble_sentiment = "positive" if weighted_prediction >= 0.5 else "negative"
        ensemble_confidence = abs(weighted_prediction - 0.5) * 2
        
        return {
            'sentiment': ensemble_sentiment,
            'confidence': ensemble_confidence,
            'individual_results': results
        }
    
    def save_ml_model(self):
        """Save trained ML model and vectorizer."""
        if self.ml_model and self.tfidf_vectorizer:
            model_path = self.models_dir / "ml_model.pkl"
            vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.ml_model, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            
            logger.info("ML model and vectorizer saved")
    
    def load_ml_model(self):
        """Load pre-trained ML model and vectorizer."""
        try:
            model_path = self.models_dir / "ml_model.pkl"
            vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
            
            if model_path.exists() and vectorizer_path.exists():
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                
                with open(vectorizer_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                
                logger.info("ML model and vectorizer loaded")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
        
        return False
