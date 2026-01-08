"""
Log Analysis Script - Analyse user submissions and retrain models
Run locally: python analyse_logs.py
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class LogAnalyser:
    """Analyse logs from the Streamlit app"""
    
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        
    def load_all_submissions(self):
        """Load all submission logs into a DataFrame"""
        all_data = []
        
        # Find all submission log files
        submission_files = list(self.logs_dir.glob("submissions_*.json"))
        
        if not submission_files:
            print("No submission logs found!")
            return pd.DataFrame()
        
        for log_file in submission_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        df = pd.DataFrame(all_data)
        print(f"\n✅ Loaded {len(df)} total submissions from {len(submission_files)} files")
        
        return df
    
    def get_statistics(self, df):
        """Get usage statistics"""
        if df.empty:
            return
        
        print("\n" + "="*50)
        print("📊 USAGE STATISTICS")
        print("="*50)
        
        # Basic stats
        print(f"\nTotal Analyses: {len(df)}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Sentiment distribution
        print("\n📈 Sentiment Distribution:")
        print(df['sentiment'].value_counts())
        print(f"\nPositive rate: {(df['sentiment'] == 'positive').sum() / len(df):.1%}")
        
        # Confidence stats
        print("\n💯 Confidence Scores:")
        print(f"Average: {df['confidence'].mean():.2%}")
        print(f"Median: {df['confidence'].median():.2%}")
        print(f"Min: {df['confidence'].min():.2%}")
        print(f"Max: {df['confidence'].max():.2%}")
        
        # Performance stats
        print("\n⚡ Performance:")
        print(f"Average processing time: {df['processing_time'].mean():.3f}s")
        print(f"Fastest: {df['processing_time'].min():.3f}s")
        print(f"Slowest: {df['processing_time'].max():.3f}s")
        
        # Text length stats
        print("\n📝 Text Length:")
        print(f"Average words: {df['text_length'].mean():.1f}")
        print(f"Shortest: {df['text_length'].min()} words")
        print(f"Longest: {df['text_length'].max()} words")
    
    def create_visualisations(self, df):
        """Create visualisation charts"""
        if df.empty:
            return
        
        print("\n📊 Creating visualisations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sentiment Analysis App - Usage Analytics', fontsize=16, fontweight='bold')
        
        # 1. Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, 
                       color=['#34d399', '#f87171'])
        axes[0, 0].set_title('Sentiment Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Confidence distribution
        axes[0, 1].hist(df['confidence'], bins=20, color='#14b8a6', alpha=0.7)
        axes[0, 1].set_title('Confidence Score Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Processing time over submissions
        axes[1, 0].plot(df['processing_time'], color='#a78bfa', linewidth=2)
        axes[1, 0].set_title('Processing Time Over Time')
        axes[1, 0].set_xlabel('Submission Number')
        axes[1, 0].set_ylabel('Time (seconds)')
        
        # 4. Text length distribution
        axes[1, 1].hist(df['text_length'], bins=20, color='#fbbf24', alpha=0.7)
        axes[1, 1].set_title('Text Length Distribution')
        axes[1, 1].set_xlabel('Number of Words')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.logs_dir / 'analytics_report.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualisations saved to: {output_path}")
        
        plt.show()
    
    def export_for_retraining(self, df, output_file="retraining_data.csv"):
        """Export data in format suitable for retraining models"""
        if df.empty:
            print("No data to export!")
            return
        
        # Prepare data for retraining
        retraining_df = df[['text', 'sentiment']].copy()
        
        # Map sentiment to labels (0 for negative, 1 for positive)
        retraining_df['label'] = retraining_df['sentiment'].map({
            'positive': 1,
            'negative': 0
        })
        
        # Remove any rows with missing values
        retraining_df = retraining_df.dropna()
        
        # Save to CSV
        output_path = self.logs_dir / output_file
        retraining_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Exported {len(retraining_df)} samples for retraining")
        print(f"📁 Saved to: {output_path}")
        print("\n💡 To retrain your models with this data:")
        print(f"   1. Load the CSV: df = pd.read_csv('{output_path}')")
        print("   2. Preprocess: df_processed = preprocessor.preprocess_dataframe(df)")
        print("   3. Retrain: analyzer.train_ml_model(df_processed)")
        
        return retraining_df
    
    def check_errors(self):
        """Check error logs"""
        error_file = self.logs_dir / "errors.json"
        
        if not error_file.exists():
            print("\n✅ No errors logged!")
            return
        
        try:
            with open(error_file, 'r') as f:
                errors = json.load(f)
            
            print("\n" + "="*50)
            print("⚠️  ERROR REPORT")
            print("="*50)
            print(f"\nTotal errors: {len(errors)}")
            
            # Group by error type
            error_df = pd.DataFrame(errors)
            print("\nError counts by context:")
            print(error_df['context'].value_counts())
            
            print("\n🔴 Recent errors:")
            for error in errors[-5:]:  # Last 5 errors
                print(f"\n  Time: {error['timestamp']}")
                print(f"  Context: {error['context']}")
                print(f"  Error: {error['error']}")
                
        except Exception as e:
            print(f"Error reading error logs: {e}")
    
    def generate_report(self):
        """Generate complete analytics report"""
        print("\n" + "="*60)
        print("🎭 SENTIMENT ANALYSIS APP - ANALYTICS REPORT")
        print("="*60)
        
        # Load data
        df = self.load_all_submissions()
        
        if df.empty:
            print("\n⚠️  No data to analyse. Start using the app to generate logs!")
            return
        
        # Get statistics
        self.get_statistics(df)
        
        # Check for errors
        self.check_errors()
        
        # Create visualisations
        self.create_visualisations(df)
        
        # Export for retraining
        self.export_for_retraining(df)
        
        print("\n" + "="*60)
        print("✅ REPORT COMPLETE")
        print("="*60)


def main():
    """Main function"""
    analyser = LogAnalyser()
    analyser.generate_report()


if __name__ == "__main__":
    main()