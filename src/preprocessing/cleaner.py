import re

class TextCleaner:
    """Provides methods for cleaning raw Vietnamese text."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans text by removing extra whitespace, special characters, 
        and standardizing formatting.
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Standardize quotes
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    @staticmethod
    def normalize_vietnamese_diacritics(text: str) -> str:
        """
        Optional: Standardizes Vietnamese diacritic placement if needed.
        For now, this is a placeholder as most modern libraries handle this well.
        """
        return text
