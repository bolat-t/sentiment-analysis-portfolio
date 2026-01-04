"""
Data collection module for sentiment analysis.
Downloads and prepares datasets automatically.
"""

import pandas as pd
import requests
import nltk
from pathlib import Path
import zipfile
import io
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection and preparation for sentiment analysis."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.download('movie_reviews', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"NLTK download failed: {e}")
    
    def get_movie_reviews_data(self) -> pd.DataFrame:
        """Download and prepare movie reviews dataset."""
        try:
            from nltk.corpus import movie_reviews
            import random
            
            # Prepare data
            documents = []
            for category in movie_reviews.categories():
                for fileid in movie_reviews.fileids(category):
                    text = movie_reviews.raw(fileid)
                    documents.append({
                        'text': text,
                        'sentiment': 1 if category == 'pos' else 0,
                        'label': category
                    })
            
            # Shuffle and convert to DataFrame
            random.shuffle(documents)
            df = pd.DataFrame(documents)
            
            logger.info(f"Loaded {len(df)} movie reviews")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load movie reviews: {e}")
            return self._create_sample_data()
    
    def get_twitter_sample_data(self) -> pd.DataFrame:
        """Create sample Twitter-like data for demonstration."""
        sample_tweets = [
            {"text": "I love this new product! Amazing quality!", "sentiment": 1, "label": "positive"},
            {"text": "Terrible service, very disappointed", "sentiment": 0, "label": "negative"},
            {"text": "Great customer support, highly recommend", "sentiment": 1, "label": "positive"},
            {"text": "Waste of money, poor quality", "sentiment": 0, "label": "negative"},
            {"text": "Outstanding performance, exceeded expectations", "sentiment": 1, "label": "positive"},
            {"text": "Broke after one week, avoid this", "sentiment": 0, "label": "negative"},
            {"text": "Perfect for my needs, love it!", "sentiment": 1, "label": "positive"},
            {"text": "Confusing interface, not user-friendly", "sentiment": 0, "label": "negative"},
            {"text": "Best purchase I've made this year", "sentiment": 1, "label": "positive"},
            {"text": "Overpriced and underdelivered", "sentiment": 0, "label": "negative"},
        ] * 100  # Repeat to create larger dataset
        
        return pd.DataFrame(sample_tweets)
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data as fallback."""
        logger.info("Creating sample data...")
        
        positive_texts = [
            "I absolutely love this movie! The acting was phenomenal.",
            "What an amazing experience! Highly recommend to everyone.",
            "Brilliant performance by the lead actor. Outstanding!",
            "This film exceeded all my expectations. Fantastic!",
            "Incredible storytelling and beautiful cinematography.",
        ] * 20
        
        negative_texts = [
            "Terrible movie, waste of time and money.",
            "Poor acting and weak storyline. Very disappointing.",
            "I couldn't even finish watching this boring film.",
            "Worst movie I've seen this year. Avoid at all costs.",
            "Completely overrated. Don't believe the hype.",
        ] * 20
        
        # Create DataFrame
        data = []
        for text in positive_texts:
            data.append({"text": text, "sentiment": 1, "label": "positive"})
        for text in negative_texts:
            data.append({"text": text, "sentiment": 0, "label": "negative"})
        
        return pd.DataFrame(data)
    
    def get_combined_dataset(self) -> pd.DataFrame:
        """Combine multiple datasets for training."""
        logger.info("Loading combined dataset...")
        
        # Try to get movie reviews
        movie_data = self.get_movie_reviews_data()
        
        # Get sample Twitter data
        twitter_data = self.get_twitter_sample_data()
        
        # Combine datasets
        combined_data = pd.concat([movie_data, twitter_data], ignore_index=True)
        
        # Clean and shuffle
        combined_data = combined_data.dropna().reset_index(drop=True)
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
        
        logger.info(f"Combined dataset size: {len(combined_data)}")
        return combined_data
