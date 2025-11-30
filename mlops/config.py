# mlops/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class MLOpsConfig:
    """Configuration for MLOps pipeline"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    METRICS_DIR = PROJECT_ROOT / "metrics"
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_VIDEOS_DIR = PROJECT_ROOT / "processed_videos"
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = "face_recognition_attendance"
    MLFLOW_MODEL_NAME = "arcface_face_recognition"
    
    # Model Configuration
    MODEL_PATH = "arcface_v3_256D.h5"
    EMBEDDING_DIM = 256
    SIMILARITY_THRESHOLD = 1.04
    
    # Monitoring Thresholds
    FAR_THRESHOLD = float(os.getenv("FAR_THRESHOLD", "0.01"))
    FRR_THRESHOLD = float(os.getenv("FRR_THRESHOLD", "0.05"))
    ACCURACY_THRESHOLD = float(os.getenv("ACCURACY_THRESHOLD", "0.95"))
    
    # Retraining Configuration
    MIN_NEW_SAMPLES = int(os.getenv("MIN_NEW_SAMPLES", "50"))
    PERFORMANCE_CHECK_INTERVAL = 3600
    RETRAINING_SCHEDULE = "0 2 * * 0"
    
    # Alert Configuration
    ALERT_EMAIL = os.getenv("ALERT_EMAIL", "admin@example.com")
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL",'postgresql://neondb_owner:npg_AO7fphz9ieod@ep-lingering-glade-aghpvil3-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require')
    
    # Session Management
    SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "5"))
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for dir_path in [cls.MODELS_DIR, cls.LOGS_DIR, cls.METRICS_DIR, 
                        cls.DATA_DIR, cls.PROCESSED_VIDEOS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

config = MLOpsConfig()
config.setup_directories()