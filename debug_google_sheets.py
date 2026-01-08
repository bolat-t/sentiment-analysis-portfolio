"""
Improved Google Sheets logging with better error handling and debugging
"""

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json
import streamlit as st
from pathlib import Path

class GoogleSheetsLogger:
    """Enhanced Google Sheets logger with detailed error reporting"""
    
    def __init__(self, credentials_dict=None):
        self.client = None
        self.sheet = None
        self.spreadsheet = None
        self.enabled = False
        self.error_message = None
        
        if credentials_dict:
            self._setup_google_sheets(credentials_dict)
    
    def _setup_google_sheets(self, credentials_dict):
        """Setup Google Sheets connection with error handling"""
        try:
            # Define scope
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            
            # Create credentials
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=scope
            )
            
            # Authorise client
            self.client = gspread.authorize(credentials)
            self.enabled = True
            print("✅ Google Sheets client authorised successfully")
            
        except Exception as e:
            self.error_message = f"Failed to authorise: {str(e)}"
            print(f"❌ Google Sheets setup failed: {self.error_message}")
            self.enabled = False
    
    def get_or_create_sheet(self, sheet_name="Sentiment Analysis Logs"):
        """Get existing sheet or create new one with detailed feedback"""
        if not self.enabled:
            print(f"⚠️ Logger not enabled. Error: {self.error_message}")
            return None
        
        try:
            # Try to open existing sheet
            print(f"🔍 Looking for sheet: '{sheet_name}'...")
            self.spreadsheet = self.client.open(sheet_name)
            self.sheet = self.spreadsheet.sheet1
            print(f"✅ Found existing sheet!")
            print(f"📊 URL: {self.spreadsheet.url}")
            
        except gspread.SpreadsheetNotFound:
            print(f"📝 Sheet not found. Creating: '{sheet_name}'...")
            
            try:
                # Create new sheet
                self.spreadsheet = self.client.create(sheet_name)
                self.sheet = self.spreadsheet.sheet1
                
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
                
                print(f"✅ Created new sheet!")
                print(f"📊 URL: {self.spreadsheet.url}")
                print(f"\n⚠️ IMPORTANT: You must share this sheet with your service account!")
                print(f"   Go to: {self.spreadsheet.url}")
                print(f"   Click 'Share' and add: {self.client.auth.service_account_email}")
                print(f"   Give 'Editor' permission")
                
            except Exception as e:
                print(f"❌ Failed to create sheet: {str(e)}")
                print("\nCommon fixes:")
                print("1. Enable Google Sheets API in Google Cloud Console")
                print("2. Enable Google Drive API in Google Cloud Console")
                self.enabled = False
                return None
        
        except Exception as e:
            print(f"❌ Failed to access sheet: {str(e)}")
            self.enabled = False
            return None
        
        return self.sheet
    
    def log_submission(self, text, result, processing_time):
        """Log submission with detailed error handling"""
        if not self.enabled:
            print(f"⚠️ Logging disabled: {self.error_message}")
            return False
        
        if not self.sheet:
            print("⚠️ Sheet not initialised. Call get_or_create_sheet() first")
            return False
        
        try:
            # Extract individual model results
            individual = result.get('individual_results', {})
            
            vader_data = individual.get('vader', {})
            textblob_data = individual.get('textblob', {})
            ml_data = individual.get('ml_model', {})
            transformer_data = individual.get('transformer', {})
            
            # Prepare row
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                text[:500],  # Truncate long text
                len(text.split()),
                result.get('sentiment', 'unknown'),
                float(result.get('confidence', 0)),
                float(processing_time),
                vader_data.get('sentiment', ''),
                float(vader_data.get('confidence', 0)) if vader_data.get('confidence') else 0,
                textblob_data.get('sentiment', ''),
                float(textblob_data.get('confidence', 0)) if textblob_data.get('confidence') else 0,
                ml_data.get('sentiment', ''),
                float(ml_data.get('confidence', 0)) if ml_data.get('confidence') else 0,
                transformer_data.get('sentiment', ''),
                float(transformer_data.get('confidence', 0)) if transformer_data.get('confidence') else 0
            ]
            
            # Append to sheet
            self.sheet.append_row(row)
            print(f"✅ Logged to Google Sheets successfully")
            return True
            
        except gspread.exceptions.APIError as e:
            error_details = str(e)
            print(f"❌ Google Sheets API Error: {error_details}")
            
            if "PERMISSION_DENIED" in error_details:
                print("\n⚠️ PERMISSION ERROR!")
                print("Fix: Share the sheet with your service account:")
                print(f"   1. Open: {self.spreadsheet.url}")
                print(f"   2. Click 'Share'")
                print(f"   3. Add: {self.client.auth.service_account_email}")
                print(f"   4. Give 'Editor' access")
            
            return False
            
        except Exception as e:
            print(f"❌ Failed to log: {str(e)}")
            return False
    
    def test_connection(self):
        """Test the Google Sheets connection"""
        print("\n" + "="*50)
        print("🧪 TESTING GOOGLE SHEETS CONNECTION")
        print("="*50)
        
        if not self.enabled:
            print(f"❌ Logger not enabled: {self.error_message}")
            return False
        
        print("✅ Client authorised")
        
        if not self.sheet:
            print("⚠️ No sheet initialised")
            return False
        
        print(f"✅ Sheet initialised: {self.spreadsheet.title}")
        print(f"📊 URL: {self.spreadsheet.url}")
        
        # Try to write test data
        try:
            test_row = ['TEST', 'Test message', 5, 'positive', 0.95, 0.001, '', 0, '', 0, '', 0, '', 0]
            self.sheet.append_row(test_row)
            print("✅ Test write successful!")
            print("\n🎉 Google Sheets logging is working!")
            return True
        except Exception as e:
            print(f"❌ Test write failed: {str(e)}")
            return False


def load_google_credentials():
    """Load credentials with better error handling"""
    print("\n🔑 Loading Google credentials...")
    
    # Try Streamlit secrets first (for deployed app)
    try:
        if hasattr(st, 'secrets') and 'google_sheets' in st.secrets:
            print("✅ Found credentials in Streamlit secrets")
            return dict(st.secrets['google_sheets'])
    except Exception as e:
        print(f"⚠️ Could not load from Streamlit secrets: {e}")
    
    # Try local file (for development)
    creds_file = Path('google_credentials.json')
    if creds_file.exists():
        try:
            with open(creds_file, 'r') as f:
                creds = json.load(f)
            print(f"✅ Found credentials in: {creds_file}")
            return creds
        except Exception as e:
            print(f"❌ Failed to load {creds_file}: {e}")
            return None
    
    print("❌ No credentials found!")
    print("\nTo enable Google Sheets logging:")
    print("1. Download service account JSON from Google Cloud")
    print("2. Save as 'google_credentials.json' in project root")
    print("3. OR add to Streamlit secrets if deploying")
    
    return None


def initialise_logger():
    """Initialise logger with detailed feedback"""
    print("\n" + "="*60)
    print("🚀 INITIALISING GOOGLE SHEETS LOGGER")
    print("="*60)
    
    credentials = load_google_credentials()
    
    if not credentials:
        print("\n⚠️ Google Sheets logging will be DISABLED")
        return GoogleSheetsLogger()
    
    logger = GoogleSheetsLogger(credentials)
    
    if logger.enabled:
        sheet = logger.get_or_create_sheet("Sentiment Analysis Logs")
        
        if sheet:
            print("\n✅ Logger initialised successfully!")
            # Run connection test
            logger.test_connection()
        else:
            print("\n❌ Failed to initialise sheet")
    else:
        print("\n❌ Logger initialisation failed")
    
    print("="*60 + "\n")
    
    return logger


# Quick test function
def test_logging():
    """Quick test of Google Sheets logging"""
    logger = initialise_logger()
    
    if logger.enabled and logger.sheet:
        print("\n🧪 Running test log...")
        
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
            "This is a test review for Google Sheets logging!",
            test_result,
            0.234
        )
        
        if success:
            print("\n🎉 TEST SUCCESSFUL!")
            print(f"Check your sheet: {logger.spreadsheet.url}")
        else:
            print("\n❌ TEST FAILED")
    else:
        print("\n❌ Logger not ready for testing")


if __name__ == "__main__":
    test_logging()