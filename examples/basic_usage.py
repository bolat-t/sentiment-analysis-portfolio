"""
Sentiment Analysis Portfolio Demo Script
This script demonstrates all the capabilities of our sentiment analysis system.
"""

import sys
import os
sys.path.append('../')

from src.data_collector import DataCollector
from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer
from src.visualizer import SentimentVisualizer
from src.utils import setup_logging, PerformanceMonitor
import pandas as pd
import numpy as np
import time

def main_demo():
    """Run the complete sentiment analysis demo."""
    
    print("🎭 Starting Sentiment Analysis Portfolio Demo")
    print("="*50)
    
    # Setup logging
    setup_logging()
    
    # 1. Data Collection
    print("\n📊 Step 1: Data Collection")
    print("-" * 30)
    
    collector = DataCollector()
    df = collector.get_combined_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nSentiment distribution:")
    print(df['label'].value_counts())
    
    # 2. Data Preprocessing
    print("\n🧹 Step 2: Data Preprocessing")
    print("-" * 30)
    
    preprocessor = TextPreprocessor()
    df_processed = preprocessor.preprocess_dataframe(df)
    
    print("Before and After Preprocessing Examples:")
    for i in range(3):
        print(f"\nOriginal: {df_processed.iloc[i]['text'][:80]}...")
        print(f"Cleaned:  {df_processed.iloc[i]['text_cleaned'][:80]}...")
    
    # 3. Model Training
    print("\n🤖 Step 3: Training ML Model")
    print("-" * 30)
    
    analyzer = SentimentAnalyzer()
    training_metrics = analyzer.train_ml_model(df_processed)
    print(f"ML Model Accuracy: {training_metrics['accuracy']:.4f}")
    
    # 4. Model Testing
    print("\n🔍 Step 4: Testing Different Models")
    print("-" * 30)
    
    test_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "Terrible quality, waste of money. Very disappointed.",
        "It's okay, nothing special but does the job.",
        "Outstanding service and great value for money!",
        "Worst experience ever, would not recommend to anyone."
    ]
    
    results = []
    for text in test_texts:
        result = analyzer.analyze_text(text)
        results.append(result)
        
        print(f"\nText: {text[:50]}...")
        print(f"  VADER: {result['vader']['sentiment']} (conf: {result['vader']['confidence']:.3f})")
        print(f"  TextBlob: {result['textblob']['sentiment']} (conf: {result['textblob']['confidence']:.3f})")
        
        if 'error' not in result['ml_model']:
            print(f"  ML Model: {result['ml_model']['sentiment']} (conf: {result['ml_model']['confidence']:.3f})")
        
        # Ensemble prediction
        ensemble = analyzer.get_ensemble_prediction(text)
        if 'error' not in ensemble:
            print(f"  Ensemble: {ensemble['sentiment']} (conf: {ensemble['confidence']:.3f})")
    
    # 5. Performance Testing
    print("\n🚀 Step 5: Performance Analysis")
    print("-" * 30)
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    test_text = "This is a great product with excellent quality and fast delivery!"
    
    # Test VADER speed
    start_time = time.time()
    vader_result = analyzer.vader_sentiment(test_text)
    vader_time = time.time() - start_time
    monitor.log_prediction('VADER', vader_result['confidence'], vader_time)
    
    # Test TextBlob speed
    start_time = time.time()
    textblob_result = analyzer.textblob_sentiment(test_text)
    textblob_time = time.time() - start_time
    monitor.log_prediction('TextBlob', textblob_result['confidence'], textblob_time)
    
    # Test ML Model speed
    start_time = time.time()
    ml_result = analyzer.ml_sentiment(test_text)
    ml_time = time.time() - start_time
    if 'error' not in ml_result:
        monitor.log_prediction('ML_Model', ml_result['confidence'], ml_time)
    
    performance_summary = monitor.get_performance_summary()
    print(f"Average processing time: {performance_summary['avg_processing_time']:.4f} seconds")
    print(f"Predictions per second: {performance_summary['predictions_per_second']:.2f}")
    
    # 6. Visualizations
    print("\n📊 Step 6: Creating Visualizations")
    print("-" * 30)
    
    visualizer = SentimentVisualizer()
    
    print("Creating sentiment distribution plot...")
    visualizer.plot_sentiment_distribution(df_processed)
    
    print("Creating model comparison plot...")
    visualizer.plot_model_comparison(results)
    
    print("Creating word clouds...")
    visualizer.create_word_clouds(df_processed)
    
    print("Creating confidence analysis...")
    visualizer.plot_confidence_analysis(results)
    
    # Save all visualizations
    visualizer.save_all_plots(df_processed, results)
    
    # 7. Real-world Application Example
    print("\n🛍️ Step 7: Real-world Application - Product Review Analysis")
    print("-" * 30)
    
    product_reviews = [
        "This laptop is incredibly fast and the battery life is amazing!",
        "Poor build quality, screen stopped working after 2 months",
        "Great value for money, does everything I need",
        "Customer service was unhelpful, took forever to get a response",
        "Love the design and performance, highly recommend!"
    ]
    
    positive_count = 0
    negative_count = 0
    
    for i, review in enumerate(product_reviews, 1):
        ensemble_result = analyzer.get_ensemble_prediction(review)
        
        if 'error' not in ensemble_result:
            sentiment = ensemble_result['sentiment']
            confidence = ensemble_result['confidence']
            
            if sentiment == 'positive':
                positive_count += 1
                emoji = "😊"
            else:
                negative_count += 1
                emoji = "😞"
            
            print(f"Review {i}: {emoji} {sentiment.upper()} (confidence: {confidence:.2f})")
            print(f"  Text: {review}")
    
    print(f"\n📈 Summary: {positive_count} positive, {negative_count} negative reviews")
    overall_sentiment = 'POSITIVE' if positive_count > negative_count else 'NEGATIVE'
    print(f"📊 Overall product sentiment: {overall_sentiment}")
    
    print("\n✅ Demo completed successfully!")
    print("Check the 'visualizations/plots' directory for generated charts.")

if __name__ == "__main__":
    main_demo()
