import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Secret & JWT configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set in environment variables")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# Database configuration (using SQLite for simplicity)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database.db")

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
