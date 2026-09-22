"""
Configuration module for News Analyzer.
Loads environment variables and provides project-wide settings.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file from the project root
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def validate_config():
    """Check that all required NewsAPI configuration is present."""
    if not NEWS_API_KEY or NEWS_API_KEY == "your_api_key_here":
        print("ERROR: NewsAPI key is not configured.")
        print("Please edit the .env file and add your NEWS_API_KEY.")
        print("Get a free key at: https://newsapi.org/register")
        sys.exit(1)


def validate_gemini_config():
    """Check that AI API configuration is present."""
    if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
        print("ERROR: Neither OpenRouter nor Gemini API key is configured.")
        print("Please edit the .env file and add OPENROUTER_API_KEY or GEMINI_API_KEY.")
        sys.exit(1)


