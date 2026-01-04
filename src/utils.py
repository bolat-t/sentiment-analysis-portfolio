"""
Utility functions for the sentiment analysis project.
"""

import json
import pickle
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Union
import pandas as pd
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Set up logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )

def time_it(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """Save data as JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Data saved to {filepath}")

def load_json(filepath: Union[str, Path]) -> Dict:
    """Load data from JSON file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from {filepath}")
    return data

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """Save object as pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Object saved to {filepath}")

def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.info(f"Object loaded from {filepath}")
    return obj

def calculate_metrics(y_true: List, y_pred: List) -> Dict[str, float]:
    """Calculate classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

def clean_text_simple(text: str) -> str:
    """Simple text cleaning utility."""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove very short texts
    if len(text.strip()) < 3:
        return ""
    
    return text.strip()

def split_data(df: pd.DataFrame, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split
    
    return train_test_split(
        df.drop('sentiment', axis=1), 
        df['sentiment'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['sentiment']
    )

def create_project_structure(base_path: str = ".") -> None:
    """Create the complete project directory structure."""
    base_path = Path(base_path)
    
    directories = [
        'data',
        'src',
        'models',
        'notebooks',
        'tests',
        'examples',
        'visualizations/plots',
        'logs'
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory in ['src', 'tests', 'data', 'models']:
            (base_path / directory / '__init__.py').touch()
    
    logger.info("Project structure created successfully")

class PerformanceMonitor:
    """Monitor model performance and system resources."""
    
    def __init__(self):
        self.metrics = []
        self.start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")
    
    def log_prediction(self, model_name: str, confidence: float, 
                      processing_time: float):
        """Log a prediction with performance metrics."""
        if self.start_time is None:
            self.start_monitoring()
        
        metric = {
            'timestamp': time.time(),
            'model': model_name,
            'confidence': confidence,
            'processing_time': processing_time,
            'session_time': time.time() - self.start_time
        }
        
        self.metrics.append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics:
            return {"error": "No metrics recorded"}
        
        df = pd.DataFrame(self.metrics)
        
        summary = {
            'total_predictions': len(df),
            'avg_processing_time': df['processing_time'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'total_session_time': time.time() - self.start_time if self.start_time else 0,
            'predictions_per_second': len(df) / (time.time() - self.start_time) if self.start_time else 0
        }
        
        # Per-model statistics
        if 'model' in df.columns:
            model_stats = df.groupby('model').agg({
                'confidence': ['mean', 'std'],
                'processing_time': ['mean', 'std']
            }).round(4)
            
            summary['model_statistics'] = model_stats.to_dict()
        
        return summary
    
    def save_metrics(self, filepath: Union[str, Path]):
        """Save collected metrics to file."""
        if self.metrics:
            save_json(self.metrics, filepath)
            logger.info(f"Performance metrics saved to {filepath}")
