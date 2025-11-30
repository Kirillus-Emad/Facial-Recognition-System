# database_setup.py
"""
Database schema setup and verification for MLOps
"""
import psycopg2
import logging
from mlops.config import config

logger = logging.getLogger(__name__)

def setup_database_schema():
    """Ensure all required database tables exist for MLOps"""
    try:
        conn = psycopg2.connect(config.DATABASE_URL)
        cur = conn.cursor()
        
        logger.info("Setting up MLOps database schema...")
        
        # Create prediction_logs table for performance monitoring
        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255),
                frame_id INTEGER,
                predicted_person_id VARCHAR(50),
                actual_person_id VARCHAR(50),
                similarity_score FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_correct BOOLEAN
            )
        """)
        
        # Create embedding_metadata table for retraining tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                person_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_retrained TIMESTAMP,
                quality_score FLOAT
            )
        """)
        
        # Create indices for better performance
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp 
            ON prediction_logs(timestamp)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_logs_session 
            ON prediction_logs(session_id)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_metadata_created 
            ON embedding_metadata(created_at)
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("✅ MLOps database schema verified and updated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

if __name__ == "__main__":
    setup_database_schema()