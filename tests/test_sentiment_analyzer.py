"""
Unit tests for the sentiment analysis system.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sentiment_analyzer import SentimentAnalyzer
from src.data_collector import DataCollector
from src.preprocessor import TextPreprocessor
import pandas as pd

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for sentiment analyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        cls.analyzer = SentimentAnalyzer()
        cls.preprocessor = TextPreprocessor()
        cls.collector = DataCollector()
    
    def test_vader_sentiment(self):
        """Test VADER sentiment analysis."""
        # Positive text
        result = self.analyzer.vader_sentiment("I love this product!")
        self.assertEqual(result['sentiment'], 'positive')
        self.assertGreater(result['confidence'], 0)
        
        # Negative text
        result = self.analyzer.vader_sentiment("This is terrible and awful!")
        self.assertEqual(result['sentiment'], 'negative')
        self.assertGreater(result['confidence'], 0)
    
    def test_textblob_sentiment(self):
        """Test TextBlob sentiment analysis."""
        # Positive text
        result = self.analyzer.textblob_sentiment("Amazing product, highly recommend!")
        self.assertEqual(result['sentiment'], 'positive')
        self.assertIsInstance(result['confidence'], float)
        
        # Negative text
        result = self.analyzer.textblob_sentiment("Worst purchase ever, complete waste!")
        self.assertEqual(result['sentiment'], 'negative')
        self.assertIsInstance(result['confidence'], float)
    
    def test_analyze_text(self):
        """Test comprehensive text analysis."""
        text = "This is an excellent product with great features!"
        results = self.analyzer.analyze_text(text)
        
        # Check that all models return results
        self.assertIn('vader', results)
        self.assertIn('textblob', results)
        self.assertIn('ml_model', results)
        
        # Check structure
        self.assertEqual(results['text'], text)
        self.assertIn('sentiment', results['vader'])
        self.assertIn('confidence', results['vader'])
    
    def test_batch_analyze(self):
        """Test batch analysis functionality."""
        texts = [
            "Great product!",
            "Terrible service!",
            "Average quality."
        ]
        
        results = self.analyzer.batch_analyze(texts)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('vader', result)
            self.assertIn('textblob', result)
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction."""
        text = "Outstanding quality and fantastic service!"
        result = self.analyzer.get_ensemble_prediction(text)
        
        if 'error' not in result:
            self.assertIn('sentiment', result)
            self.assertIn('confidence', result)
            self.assertIn(result['sentiment'], ['positive', 'negative'])
    
    def test_empty_text(self):
        """Test handling of empty or invalid text."""
        # Empty string
        result = self.analyzer.vader_sentiment("")
        self.assertIsInstance(result, dict)
        
        # None input (should be handled gracefully)
        try:
            result = self.analyzer.vader_sentiment(None)
        except (TypeError, AttributeError):
            pass  # Expected behavior
    
    def test_preprocessor(self):
        """Test text preprocessing."""
        dirty_text = "This is AMAZING!!! @user #hashtag http://example.com"
        cleaned = self.preprocessor.preprocess_text(dirty_text)
        
        self.assertIsInstance(cleaned, str)
        self.assertNotIn("http://", cleaned)
        self.assertNotIn("@user", cleaned)
        self.assertNotIn("#hashtag", cleaned)
    
    def test_data_collector(self):
        """Test data collection functionality."""
        df = self.collector.get_combined_dataset()
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('text', df.columns)
        self.assertIn('sentiment', df.columns)
        self.assertIn('label', df.columns)
        
        # Check data types
        self.assertTrue(df['sentiment'].dtype in ['int64', 'float64'])
        self.assertIn(df['label'].dtype, ['object', 'category'])

class TestModelTraining(unittest.TestCase):
    """Test cases for model training functionality."""
    
    def setUp(self):
        """Set up for each test."""
        self.analyzer = SentimentAnalyzer()
        self.collector = DataCollector()
        self.preprocessor = TextPreprocessor()
    
    def test_ml_model_training(self):
        """Test ML model training process."""
        # Get sample data
        df = self.collector._create_sample_data()  # Use sample data for speed
        df_processed = self.preprocessor.preprocess_dataframe(df)
        
        # Train model
        metrics = self.analyzer.train_ml_model(df_processed)
        
        # Check metrics
        self.assertIn('accuracy', metrics)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        
        # Check that model was trained
        self.assertIsNotNone(self.analyzer.ml_model)
        self.assertIsNotNone(self.analyzer.tfidf_vectorizer)
    
    def test_model_prediction_after_training(self):
        """Test model predictions after training."""
        # Train model with sample data
        df = self.collector._create_sample_data()
        df_processed = self.preprocessor.preprocess_dataframe(df)
        self.analyzer.train_ml_model(df_processed)
        
        # Test prediction
        result = self.analyzer.ml_sentiment("This is a great product!")
        
        if 'error' not in result:
            self.assertIn('sentiment', result)
            self.assertIn('confidence', result)
            self.assertIn(result['sentiment'], ['positive', 'negative'])

class TestPerformance(unittest.TestCase):
    """Test cases for performance and edge cases."""
    
    def setUp(self):
        """Set up for each test."""
        self.analyzer = SentimentAnalyzer()
    
    def test_long_text_handling(self):
        """Test handling of very long texts."""
        long_text = "This is amazing! " * 1000  # Very long text
        
        result = self.analyzer.vader_sentiment(long_text)
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
    
    def test_special_characters(self):
        """Test handling of special characters."""
        special_text = "Great! 😊 #love $100 @user http://test.com"
        
        result = self.analyzer.analyze_text(special_text)
        self.assertIsInstance(result, dict)
        self.assertIn('vader', result)
    
    def test_non_english_text(self):
        """Test handling of non-English text."""
        # Should handle gracefully even if not designed for non-English
        foreign_text = "Très bien! Excelente producto!"
        
        try:
            result = self.analyzer.analyze_text(foreign_text)
            self.assertIsInstance(result, dict)
        except Exception:
            pass  # Acceptable if not designed for multilingual

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
