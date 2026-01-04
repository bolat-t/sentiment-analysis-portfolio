"""
Text preprocessing module for sentiment analysis.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text preprocessing for sentiment analysis."""
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('punkt', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"NLTK setup warning: {e}")
            self.stop_words = set()
            self.lemmatizer = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (Twitter-style)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation while preserving sentence structure."""
        # Keep some punctuation that might be important for sentiment
        important_punct = '!?'
        
        # Create translation table
        translator = str.maketrans('', '', 
            ''.join([char for char in string.punctuation if char not in important_punct]))
        
        return text.translate(translator)
    
    def tokenize_and_filter(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text and optionally remove stopwords."""
        try:
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords if requested
            if remove_stopwords and self.stop_words:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Filter out very short tokens
            tokens = [token for token in tokens if len(token) > 1]
            
            return tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return text.split()
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base forms."""
        if not self.lemmatizer:
            return tokens
        
        try:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return tokens
    
    def preprocess_text(self, text: str, 
                       clean: bool = True,
                       remove_punct: bool = True,
                       remove_stopwords: bool = True,
                       lemmatize: bool = True) -> str:
        """Complete text preprocessing pipeline."""
        
        if clean:
            text = self.clean_text(text)
        
        if remove_punct:
            text = self.remove_punctuation(text)
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(text, remove_stopwords)
        
        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, 
                           text_column: str = 'text') -> pd.DataFrame:
        """Preprocess text data in a pandas DataFrame."""
        logger.info("Preprocessing text data...")
        
        df_copy = df.copy()
        
        # Apply preprocessing
        df_copy[f'{text_column}_cleaned'] = df_copy[text_column].apply(
            lambda x: self.preprocess_text(x)
        )
        
        # Remove empty texts after preprocessing
        df_copy = df_copy[df_copy[f'{text_column}_cleaned'].str.len() > 0]
        
        logger.info(f"Preprocessing complete. {len(df_copy)} samples remaining.")
        return df_copy.reset_index(drop=True)
