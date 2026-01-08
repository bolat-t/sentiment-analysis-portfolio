"""
Google Sheets logging integration for Streamlit app
Logs all user submissions to Google Sheets for easy access and analysis
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import streamlit as st
from pathlib import Path


class GoogleSheetsLogger:
    """Log sentiment analysis data to Google Sheets"""
    
    def __init__(self, credentials_dict=None):
        """
        Initialise Google Sheets logger
        
        Args:
            credentials_dict: Dictionary with Google service account credentials
        """
        self.client = None
        self.sheet = None
        self.enabled = False
        
        try:
            if credentials_dict:
                self._setup_google_sheets(credentials_dict)
        except Exception as e:
            print(f"Google Sheets logging disabled: {e}")
    
    def _setup_google_sheets(self, credentials_dict):
        """Setup Google Sheets connection"""
        # Define the scope
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Create credentials
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=scope
        )
        
        # Authorise the client
        self.client = gspread.authorize(credentials)
        self.enabled = True
    
    def get_or_create_sheet(self, sheet_name="Sentiment Analysis Logs"):
        """Get existing sheet or create new one"""
        if not self.enabled:
            return None
        
        try:
            # Try to open existing sheet
            self.sheet = self.client.open(sheet_name).sheet1
        except gspread.SpreadsheetNotFound:
            # Create new sheet
            spreadsheet = self.client.create(sheet_name)
            self.sheet = spreadsheet.sheet1
            
            # Share with your email (replace with your email)
            spreadsheet.share('sentiment-logger@sentiment-analysis-logger.iam.gserviceaccount.com', perm_type='user', role='writer')
            
            # Add headers
            headers = [
                'Timestamp',
                'Text',
                'Text Length',
                'Sentiment',
                'Confidence',
                'Processing Time',
                'VADER Sentiment',
                'VADER Confidence',
                'TextBlob Sentiment',
                'TextBlob Confidence',
                'ML Model Sentiment',
                'ML Model Confidence',
                'Transformer Sentiment',
                'Transformer Confidence'
            ]
            self.sheet.append_row(headers)
            
            print(f"✅ Created new Google Sheet: {sheet_name}")
        
        return self.sheet
    
    def log_submission(self, text, result, processing_time):
        """
        Log a user submission to Google Sheets
        
        Args:
            text: User input text
            result: Analysis result dictionary
            processing_time: Time taken for analysis
        """
        if not self.enabled or not self.sheet:
            return False
        
        try:
            # Extract individual model results
            individual = result.get('individual_results', {})
            
            vader_data = individual.get('vader', {})
            textblob_data = individual.get('textblob', {})
            ml_data = individual.get('ml_model', {})
            transformer_data = individual.get('transformer', {})
            
            # Prepare row data
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                text[:500],  # Truncate long text
                len(text.split()),
                result.get('sentiment', 'unknown'),
                float(result.get('confidence', 0)),
                float(processing_time),
                vader_data.get('sentiment', ''),
                float(vader_data.get('confidence', 0)),
                textblob_data.get('sentiment', ''),
                float(textblob_data.get('confidence', 0)),
                ml_data.get('sentiment', ''),
                float(ml_data.get('confidence', 0)),
                transformer_data.get('sentiment', ''),
                float(transformer_data.get('confidence', 0))
            ]
            
            # Append to sheet
            self.sheet.append_row(row)
            return True
            
        except Exception as e:
            print(f"Error logging to Google Sheets: {e}")
            return False
    
    def log_error(self, error_message, context=""):
        """Log errors to a separate sheet"""
        if not self.enabled:
            return False
        
        try:
            # Get or create error sheet
            try:
                error_sheet = self.client.open("Sentiment Analysis Logs").worksheet("Errors")
            except:
                error_sheet = self.client.open("Sentiment Analysis Logs").add_worksheet(
                    title="Errors", rows="1000", cols="3"
                )
                error_sheet.append_row(['Timestamp', 'Context', 'Error'])
            
            # Log error
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                context,
                str(error_message)
            ]
            error_sheet.append_row(row)
            return True
            
        except Exception as e:
            print(f"Error logging error to Google Sheets: {e}")
            return False



def load_google_credentials():
    """
    Load Google Sheets credentials from Streamlit secrets or local file
    """
    try:
        # First try Streamlit secrets
        if hasattr(st, 'secrets') and 'google_sheets' in st.secrets:
            return dict(st.secrets['google_sheets'])

        # Use path relative to this file
        creds_file = Path(__file__).parent / 'google_credentials.json'
        if creds_file.exists():
            with open(creds_file, 'r') as f:
                return json.load(f)

        return None
    except Exception as e:
        print(f"Could not load Google credentials: {e}")
        return None



# Example usage in your app
def initialise_logger():
    """Initialise Google Sheets logger for the app"""
    credentials = load_google_credentials()
    
    if credentials:
        logger = GoogleSheetsLogger(credentials)
        logger.get_or_create_sheet("Sentiment Analysis Logs")
        return logger
    else:
        print("⚠️ Google Sheets logging not configured")
        return GoogleSheetsLogger()  # Return disabled logger


# Test function
def test_logging():
    """Test Google Sheets logging"""
    logger = initialise_logger()
    
    if logger.enabled:
        # Test submission
        test_result = {
            'sentiment': 'positive',
            'confidence': 0.89,
            'individual_results': {
                'vader': {'sentiment': 'positive', 'confidence': 0.85},
                'textblob': {'sentiment': 'positive', 'confidence': 0.82},
                'ml_model': {'sentiment': 'positive', 'confidence': 0.91},
                'transformer': {'sentiment': 'positive', 'confidence': 0.94}
            }
        }
        
        success = logger.log_submission(
            "This is a test review for logging!",
            test_result,
            0.234
        )
        
        if success:
            print("✅ Test log successfully written to Google Sheets!")
        else:
            print("❌ Failed to write test log")
    else:
        print("❌ Logger not enabled. Check your credentials.")


if __name__ == "__main__":
    test_logging()