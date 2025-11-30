#database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import psycopg2
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# تحميل الإعدادات من .env
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:5555@localhost:5432/test_emb")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", 256))

# Connection Pool Setup - الإعدادات المحسنة
engine = create_engine(
    DATABASE_URL,
    pool_size=10,           # Maximum connections
    max_overflow=20,        # Extra connections if pool is full
    pool_pre_ping=True,     # Verify connections before use
    pool_recycle=3600       # Recycle connections after 1 hour
)

# Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Dependency that provides database session
    Used in FastAPI route dependencies
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  # Always close connection

def get_vector_connection():
    """اتصال مباشر لعمليات pgvector"""
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def create_tables():
    """Creates all tables in the database"""
    Base.metadata.create_all(bind=engine)

def get_engine():
    """Provides direct engine access for complex operations"""
    return engine

print("Successfully true")


