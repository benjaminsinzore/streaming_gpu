
import os
import sys
import subprocess
import logging
import torch
import time
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all required Python packages are installed"""
    logger.info("Checking requirements...")
    
    requirements = [
        "torch", "torchaudio", "fastapi", "uvicorn", "websockets", "numpy",
        "sqlalchemy", "pydantic", "jinja2", "openai-whisper", "sounddevice",
        "sentence-transformers"
    ]
    
    missing = []
    for req in requirements:
        try:
            __import__(req.replace("-", "_"))  # Handle package name differences
        except ImportError:
            missing.append(req)
    
    if missing:
        logger.warning(f"Missing required packages: {', '.join(missing)}")
        logger.info("Installing missing requirements...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install"] + missing, check=True)
            logger.info("Requirements installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install requirements: {e}")
            sys.exit(1)
    else:
        logger.info("All requirements are satisfied")

def download_vad_model():
    """Download the Silero VAD model using PyTorch Hub"""
    model_path = "silero_vad.jit"
    
    if os.path.exists(model_path):
        logger.info(f"Silero VAD model already exists at {model_path}")
        return
    
    logger.info("Downloading Silero VAD model using PyTorch Hub...")
    try:
        torch.hub.set_dir("./models")
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                                      model="silero_vad",
                                      force_reload=True,
                                      onnx=False)
        torch.jit.save(model, model_path)
        logger.info(f"Model downloaded and saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to download Silero VAD model: {e}")
        logger.info("Falling back to energy-based VAD - the system will still work but with simpler voice detection")

def download_embedding_models():
    """Download the sentence transformer models for RAG"""
    logger.info("Setting up sentence transformer models...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("Downloading embedding models (this may take a few minutes)...")
        models = [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-mpnet-base-dot-v1"
        ]
        
        for model_name in models:
            logger.info(f"Setting up model: {model_name}")
            _ = SentenceTransformer(model_name)
            logger.info(f"Model {model_name} is ready")
    except Exception as e:
        logger.error(f"Failed to download embedding models: {e}")
        logger.error("Please try running the script again or download models manually")

def setup_directories():
    """Create necessary directories for the application"""
    directories = ["static", "responses", "embeddings_cache", "templates", "audio/user", "audio/ai"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory {directory} is ready")
    
    template_dir = Path("templates")
    index_html = template_dir / "index.html"
    
    with open(index_html, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url=/setup">
</head>
<body>
    <p>Redirecting to <a href="/setup">AI Companion Setup</a>...</p>
</body>
</html>
        """)
    logger.info("Created index template for redirection")

def setup_database():
    """Initialize the SQLite database"""
    logger.info("Setting up database...")
    
    try:
        from sqlalchemy import create_engine, Column, Integer, String, Text
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        
        Base = declarative_base()
        engine = create_engine("sqlite:///companion.db")
        
        class Conversation(Base):
            __tablename__ = "conversations"
            id = Column(Integer, primary_key=True, index=True)
            session_id = Column(String, index=True)
            timestamp = Column(String)
            user_message = Column(Text)
            ai_message = Column(Text)
            audio_path = Column(String)
        
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to set up database: {e}")

def check_cuda():
    """Check hardware compatibility for macOS"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA is available: {device_name}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        logger.info("Running on CPU (CUDA not available on macOS)")
        logger.info("Ensure sufficient RAM (16 GB or more recommended) for optimal performance")
        logger.info("Consider using a smaller Whisper model (e.g., openai/whisper-small) for faster processing")

def main():
    """Main setup function"""
    logger.info("Starting AI Companion setup...")
    
    check_cuda()
    
    # Uncomment to enable requirements check
    # check_requirements()
    
    setup_directories()
    
    setup_database()
    
    download_vad_model()
    download_embedding_models()
    
    logger.info("Setup completed successfully!")
    logger.info("You can now start the application with:")
    logger.info("   python companion.py")

if __name__ == "__main__":
    main()