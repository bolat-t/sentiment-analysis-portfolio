import argparse
import logging
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_collector import DataCollector
from preprocessor import TextPreprocessor
from sentiment_analyzer import SentimentAnalyzer
from visualizer import SentimentVisualizer
from utils import setup_logging, create_project_structure, PerformanceMonitor

def setup_project():
    """Initialize project structure and setup."""
    print("🔧 Setting up project structure...")
    create_project_structure()
    setup_logging("INFO", "logs/sentiment_analysis.log")
    print("✅ Project setup complete!")

def train_models():
    """Train all sentiment analysis models."""
    print("🤖 Training sentiment analysis models...")
    
    # Initialize components
    collector = DataCollector()
    preprocessor = TextPreprocessor()
    analyzer = SentimentAnalyzer()
    
    # Load and preprocess data
    print("📊 Loading data...")
    df = collector.get_combined_dataset()
    
    print("🧹 Preprocessing data...")
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Train ML model
    print("🎯 Training ML model...")
    metrics = analyzer.train_ml_model(df_processed)
    print(f"✅ ML model trained with accuracy: {metrics['accuracy']:.4f}")
    
    return df_processed, analyzer

def run_analysis(text: str = None):
    """Run sentiment analysis on provided text or demo texts."""
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Try to load pre-trained model
    if not analyzer.load_ml_model():
        print("⚠️ Pre-trained ML model not found. Training new model...")
        _, analyzer = train_models()
    
    if text:
        # Analyze single text
        print(f"\n🔍 Analyzing text: '{text}'")
        print("-" * 50)
        
        result = analyzer.get_ensemble_prediction(text)
        
        if 'error' not in result:
            print(f"🎭 Sentiment: {result['sentiment'].upper()}")
            print(f"📊 Confidence: {result['confidence']:.4f}")
            print(f"🔬 Individual model results:")
            
            individual = result['individual_results']
            for model_name, model_result in individual.items():
                if isinstance(model_result, dict) and 'sentiment' in model_result:
                    print(f"  - {model_name.upper()}: {model_result['sentiment']} "
                          f"(conf: {model_result['confidence']:.3f})")
        else:
            print(f"❌ Error: {result['error']}")
    
    else:
        # Run demo analysis
        print("🎯 Running demo analysis...")
        
        demo_texts = [
            "I absolutely love this new smartphone! The camera quality is outstanding.",
            "Terrible customer service, waited 2 hours and got no help.",
            "The product is okay, nothing special but does what it's supposed to do.",
            "Amazing experience! Fast delivery and great quality. Highly recommend!",
            "Complete waste of money. Poor quality and broke after one week."
        ]
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        for i, demo_text in enumerate(demo_texts, 1):
            print(f"\n📝 Demo {i}: {demo_text}")
            print("-" * 40)
            
            start_time = time.time()
            result = analyzer.get_ensemble_prediction(demo_text)
            processing_time = time.time() - start_time
            
            if 'error' not in result:
                sentiment = result['sentiment']
                confidence = result['confidence']
                emoji = "😊" if sentiment == 'positive' else "😞"
                
                print(f"{emoji} Sentiment: {sentiment.upper()} (confidence: {confidence:.4f})")
                
                # Log to performance monitor
                monitor.log_prediction('Ensemble', confidence, processing_time)
            else:
                print(f"❌ Error: {result['error']}")
        
        # Show performance summary
        try:
            perf_summary = monitor.get_performance_summary()
            if 'error' not in perf_summary:
                print(f"\n⚡ Performance Summary:")
                print(f"   Total predictions: {perf_summary['total_predictions']}")
                print(f"   Average processing time: {perf_summary['avg_processing_time']:.4f}s")
                print(f"   Predictions per second: {perf_summary['predictions_per_second']:.2f}")
        except Exception as e:
            print(f"\n⚠️ Performance summary unavailable: {e}")

def generate_visualizations():
    """Generate all visualization plots and reports."""
    print("📊 Generating visualizations...")
    
    # Check if we have processed data
    try:
        # Initialize components
        collector = DataCollector()
        analyzer = SentimentAnalyzer()
        visualizer = SentimentVisualizer()
        
        # Load data
        df = collector.get_combined_dataset()
        preprocessor = TextPreprocessor()
        df_processed = preprocessor.preprocess_dataframe(df)
        
        # Generate sample results for visualization
        sample_texts = df_processed['text'].head(50).tolist()
        results = analyzer.batch_analyze(sample_texts)
        
        # Create all visualizations
        visualizer.save_all_plots(df_processed, results)
        
        print("✅ All visualizations saved to 'visualizations/plots' directory!")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        logging.error(f"Visualization generation failed: {e}")

def run_interactive_mode():
    """Run interactive sentiment analysis mode."""
    print("🎮 Starting Interactive Sentiment Analysis Mode")
    print("Type 'quit' or 'exit' to stop, 'help' for commands\n")
    
    analyzer = SentimentAnalyzer()
    
    # Try to load pre-trained model
    if not analyzer.load_ml_model():
        print("⚠️ No pre-trained model found. Training...")
        _, analyzer = train_models()
    
    while True:
        try:
            text = input("📝 Enter text to analyze: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if text.lower() == 'help':
                print("\n📖 Available commands:")
                print("  - Type any text to analyze its sentiment")
                print("  - 'quit' or 'exit' to stop")
                print("  - 'help' to show this message\n")
                continue
            
            if not text:
                print("⚠️ Please enter some text to analyze.\n")
                continue
            
            # Analyze the text
            result = analyzer.get_ensemble_prediction(text)
            
            if 'error' not in result:
                sentiment = result['sentiment']
                confidence = result['confidence']
                emoji = "😊" if sentiment == 'positive' else "😞"
                
                print(f"\n{emoji} Result: {sentiment.upper()} (confidence: {confidence:.4f})")
                
                # Show individual model results
                individual = result['individual_results']
                print("🔬 Individual model results:")
                for model_name, model_result in individual.items():
                    if isinstance(model_result, dict) and 'sentiment' in model_result:
                        print(f"  - {model_name}: {model_result['sentiment']} "
                              f"({model_result['confidence']:.3f})")
                print()
            else:
                print(f"❌ Error: {result['error']}\n")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="🎭 Advanced Sentiment Analysis Portfolio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --setup                    # Initialize project
  python main.py --train                    # Train models
  python main.py --text "I love this!"      # Analyze specific text
  python main.py --demo                     # Run demo analysis
  python main.py --interactive              # Interactive mode
  python main.py --visualize                # Generate visualizations
        """
    )
    
    parser.add_argument('--setup', action='store_true',
                       help='Setup project structure')
    parser.add_argument('--train', action='store_true',
                       help='Train sentiment analysis models')
    parser.add_argument('--text', type=str,
                       help='Analyze sentiment of specific text')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo analysis with sample texts')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # Show welcome message
    print("🎭 Advanced Sentiment Analysis Portfolio")
    print("="*50)
    
    try:
        if args.setup:
            setup_project()
        
        elif args.train:
            train_models()
        
        elif args.text:
            run_analysis(args.text)
        
        elif args.demo:
            run_analysis()
        
        elif args.interactive:
            run_interactive_mode()
        
        elif args.visualize:
            generate_visualizations()
        
        else:
            # Default behavior - run demo
            print("No specific action specified. Running demo...")
            print("Use --help to see all available options.\n")
            run_analysis()
    
    except KeyboardInterrupt:
        print("\n👋 Process interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()