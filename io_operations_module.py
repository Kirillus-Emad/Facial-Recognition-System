# io_operations_module.py - Asynchronous I/O Operations
"""
BACKGROUND PROCESS MODULE - Handles all I/O operations
- Database writes
- Logging
- File writing (processed videos)
- MLOps logging
Runs in separate process to avoid blocking main detection loop
"""
import multiprocessing as mp
from queue import Empty
import psycopg2
import cv2
import logging
from datetime import datetime, date
import json
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseHandler:
    """Handles all database operations in background"""
    
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.attendance_recorded = set()
        
        # NEW: Only connect if config provided
        if db_config:
            self._connect()
            self._ensure_schema()
            self._load_today_attendance()
        else:
            print("⚠️ Running without database")
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                dbname=self.db_config['dbname'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                sslmode=self.db_config.get('sslmode', 'require')
            )
            logger.info("✅ Database connected")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def _ensure_schema(self):
        """Ensure required tables exist"""
        try:
            cur = self.conn.cursor()
            
            # Attendance unique index
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_person_date
                ON attendance (person_id, attendance_date);
            """)
            
            self.conn.commit()
            cur.close()
            logger.info("✅ Database schema verified")
        except Exception as e:
            logger.error(f"❌ Schema setup failed: {e}")
    
    def _load_today_attendance(self):
        """Load today's attendance records"""
        try:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT person_id FROM attendance
                WHERE attendance_date = CURRENT_DATE
            """)
            rows = cur.fetchall()
            cur.close()
            self.attendance_recorded = set(int(r[0]) for r in rows)
            logger.info(f"Loaded {len(self.attendance_recorded)} attendance records for today")
        except Exception as e:
            logger.error(f"Error loading attendance: {e}")
            self.attendance_recorded = set()
            
            
    def log_prediction(self, session_id, frame_id, predicted_id, actual_id, score):
        """Log prediction for MLOps monitoring"""
        try:
            cur = self.conn.cursor()
            is_correct = (predicted_id == actual_id)
            
            cur.execute("""
                INSERT INTO prediction_logs 
                (session_id, frame_id, predicted_person_id, actual_person_id, 
                 similarity_score, is_correct)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (session_id, frame_id, predicted_id, actual_id, score, is_correct))
            
            self.conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"Prediction logging error: {e}")
            
            
    def mark_attendance(self, person_id):
        """Mark attendance (idempotent)"""
        try:
            if person_id in self.attendance_recorded:
                return False  # Already recorded
            
            cur = self.conn.cursor()
            cur.execute("""
                INSERT INTO attendance (person_id, attendance_date)
                VALUES (%s, CURRENT_DATE)
                ON CONFLICT DO NOTHING;
            """, (person_id,))
            self.conn.commit()
            cur.close()
            
            self.attendance_recorded.add(person_id)
            logger.info(f"✅ Attendance marked for person {person_id}")
            return True
        except Exception as e:
            logger.error(f"Attendance error for {person_id}: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


class VideoWriter:
    """Handles video writing in background"""
    
    def __init__(self, output_path, fps, frame_size):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        logger.info(f"Video writer initialized: {output_path}")
    
    def write_frame(self, frame):
        """Write frame to video file"""
        try:
            self.writer.write(frame)
        except Exception as e:
            logger.error(f"Frame write error: {e}")
    
    def release(self):
        """Release video writer"""
        if self.writer:
            self.writer.release()
            logger.info(f"Video writer released: {self.output_path}")


class IOProcessor:
    """Main I/O processor running in separate process"""
    
    def __init__(self, io_queue, db_config):
        self.io_queue = io_queue
        self.db_handler = DatabaseHandler(db_config)
        self.video_writers = {}  # session_id -> VideoWriter
        self.running = True
    
    def process_queue(self):
        """Main processing loop"""
        logger.info("🚀 IO Processor started")
        
        while self.running:
            try:
                # Non-blocking get with timeout
                task = self.io_queue.get(timeout=0.1)
                
                task_type = task.get('type')
                
                if task_type == 'attendance':
                    person_id = task['person_id']
                    self.db_handler.mark_attendance(person_id)
                

                
                elif task_type == 'init_video':
                    session_id = task['session_id']
                    self.video_writers[session_id] = VideoWriter(
                        task['output_path'],
                        task['fps'],
                        task['frame_size']
                    )
                
                elif task_type == 'write_frame':
                    session_id = task['session_id']
                    if session_id in self.video_writers:
                        self.video_writers[session_id].write_frame(task['frame'])
                
                elif task_type == 'close_video':
                    session_id = task['session_id']
                    if session_id in self.video_writers:
                        self.video_writers[session_id].release()
                        del self.video_writers[session_id]
                
                elif task_type == 'shutdown':
                    logger.info("Shutdown signal received")
                    self.running = False
                    break
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Task processing error: {e}")
        
        self._cleanup()
        logger.info("IO Processor stopped")
    
    def _cleanup(self):
        """Cleanup resources"""
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
        self.db_handler.close()


def start_io_process(io_queue, db_config):
    """Function to run in separate process"""
    processor = IOProcessor(io_queue, db_config)
    processor.process_queue()