# face_processor_optimized.py - Main Processing Coordinator
"""
OPTIMIZED VERSION - Separates computation and I/O
- Face detection/recognition in main process (high priority)
- Database/logging/file I/O in separate process (background)
- Multiprocessing queue for communication
"""
import cv2
import asyncio
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
from datetime import datetime
import uuid
import json

# Import our modules
from face_detection_module import FaceDetector, FrameAnnotator
from io_operations_module import start_io_process

# Database configuration
DB_CONFIG = {
    'host': "ep-lingering-glade-aghpvil3-pooler.c-2.eu-central-1.aws.neon.tech",
    'dbname': "neondb",
    'user': "neondb_owner",
    'password': "npg_AO7fphz9ieod",
    'sslmode': "require"
}

# Global I/O queue (will be initialized in main)
io_queue = None
io_process = None

# Dashboard counters
dashboard_stats = {
    'total_faces': 0,
    'known_faces': 0,
    'unknown_faces': 0,
    'known_faces_list': [],
    'unknown_faces_list': []
}


def initialize_io_process():
    """Initialize background I/O process"""
    global io_queue, io_process
    
    if io_queue is None:
        io_queue = Queue(maxsize=1000)  # Bounded queue
        io_process = Process(target=start_io_process, args=(io_queue, DB_CONFIG), daemon=True)
        io_process.start()
        print("✅ IO Process started")


def shutdown_io_process():
    """Shutdown background I/O process"""
    global io_queue, io_process
    
    if io_queue:
        io_queue.put({'type': 'shutdown'})
    
    if io_process:
        io_process.join(timeout=5)
        if io_process.is_alive():
            io_process.terminate()
        print("✅ IO Process stopped")


def load_gallery_embeddings():
    """Load gallery embeddings from database"""
    import psycopg2
    
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            sslmode=DB_CONFIG['sslmode']
        )
        cur = conn.cursor()
        cur.execute("SELECT person_id, embedding FROM person_embeddings ORDER BY person_id")
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []
    
    gallery = []
    for person_id, emb_vector in rows:
        try:
            if isinstance(emb_vector, list):
                embedding_np = np.array(emb_vector, dtype=float)
            elif isinstance(emb_vector, str):
                embedding_np = np.array(json.loads(emb_vector), dtype=float)
            else:
                embedding_np = np.array(emb_vector, dtype=float)
            
            norm = np.linalg.norm(embedding_np)
            if norm == 0:
                continue
            embedding_np = embedding_np / norm
            
            gallery.append({
                "subject": str(person_id),
                "embedding": embedding_np
            })
        except Exception as e:
            print(f"Error processing embedding for person {person_id}: {e}")
            continue
    
    print(f"✅ Loaded {len(gallery)} gallery embeddings")
    return gallery


def reset_dashboard():
    """Reset dashboard statistics"""
    global dashboard_stats,tracked_id
    tracked_id=set()
    dashboard_stats = {
        'total_faces': 0,
        'known_faces': 0,
        'unknown_faces': 0,
        'known_faces_list': [],
        'unknown_faces_list': []
    }


def update_dashboard(detections):
    """Update dashboard with new detections"""
    global dashboard_stats,tracked_id
    
    for det in detections:
        if det['status'] == 'recognized' and det['track_id'] not in tracked_id:
            tracked_id.add(det['track_id'])
            dashboard_stats['total_faces'] += 1
            dashboard_stats['known_faces'] += 1
            
            # Store face data (limit to 100 faces)
            if len(dashboard_stats['known_faces_list']) < 100:
                success, buffer = cv2.imencode('.jpg', det['face_crop'])
                if success:
                    import base64
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    dashboard_stats['known_faces_list'].append({
                        'subject': det['subject'],
                        'face_image': face_base64,
                        'track_id': det['track_id'],
                        'score': det['score']
                    })
        
        elif det['status'] == 'unknown' and det['track_id'] not in tracked_id:
            tracked_id.add(det['track_id'])
            dashboard_stats['total_faces'] += 1
            dashboard_stats['unknown_faces'] += 1
            
            if len(dashboard_stats['unknown_faces_list']) < 100:
                success, buffer = cv2.imencode('.jpg', det['face_crop'])
                if success:
                    import base64
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    dashboard_stats['unknown_faces_list'].append({
                        'face_image': face_base64,
                        'track_id': det['track_id'],
                        'score': det['score']
                    })


def get_dashboard_data():
    """Get current dashboard data"""
    return {
        'total_faces': dashboard_stats['total_faces'],
        'known_faces_num': dashboard_stats['known_faces'],
        'unknown_faces_num': dashboard_stats['unknown_faces'],
        'known_faces': dashboard_stats['known_faces_list'],
        'unknown_faces': dashboard_stats['unknown_faces_list']
    }


async def process_video_realtime_frames(input_path: str, filename: str, 
                                       active_streams: dict, camera_view: str = "Front"):
    """Process video frames with optimized pipeline"""
    global io_queue
    
    # Initialize I/O process if not done
    initialize_io_process()
    
    # Reset dashboard
    reset_dashboard()
    
    # Generate session ID
    session_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Load gallery
    gallery = load_gallery_embeddings()
    
    # Initialize detector
    detector = FaceDetector(gallery, camera_view=camera_view)
    
    cap = None
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize stream info
        stream_info = active_streams[filename]
        stream_info['total_frames'] = total_frames
        stream_info['processed_frames'] = 0
        stream_info['session_id'] = session_id
        
        # Initialize video writer in I/O process
        output_path = f"processed_videos/processed_{filename}"
        io_queue.put({
            'type': 'init_video',
            'session_id': session_id,
            'output_path': output_path,
            'fps': fps,
            'frame_size': (width, height)
        })
        
        frame_id = 0
        print(f"🚀 Starting video processing: {filename}")
        print(f"📹 Camera View: {camera_view}")
        print(f"⚡ Total frames: {total_frames}")
        
        while True:
            if stream_info.get("stop_requested", False):
                print(f"🛑 Stop requested for: {filename}")
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            stream_info['processed_frames'] = frame_id
            
            # CRITICAL PATH: Detection and recognition (main process)
            if detector.should_process_frame():
                detections = detector.detect_and_recognize(frame, frame_id)
                
                # Update dashboard
                update_dashboard(detections)
                
                # Send I/O tasks to background process (non-blocking)
                for det in detections:
                    if det['status'] == 'recognized' and det['subject']:
                        try:
                            person_id = int(det['subject'])
                            
                            # Queue attendance marking
                            io_queue.put({
                                'type': 'attendance',
                                'person_id': person_id
                            }, block=False)
                            
                            # Queue prediction logging
                            io_queue.put({
                                'type': 'prediction_log',
                                'session_id': session_id,
                                'frame_id': frame_id,
                                'predicted_id': det['subject'],
                                'actual_id': det['subject'],
                                'score': det['score']
                            }, block=False)
                        except:
                            pass
                
                # Annotate frame
                annotated = FrameAnnotator.annotate_frame(
                    frame, detections, frame_id, camera_view,
                    {
                        'total': dashboard_stats['total_faces'],
                        'known': dashboard_stats['known_faces'],
                        'unknown': dashboard_stats['unknown_faces']
                    }
                )
            else:
                # Fast path: just add frame counter
                annotated = frame.copy()
                cv2.putText(annotated, f"Frame: {frame_id} | View: {camera_view} | SKIPPED", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Queue frame for video writing (non-blocking)
            try:
                io_queue.put({
                    'type': 'write_frame',
                    'session_id': session_id,
                    'frame': annotated.copy()
                }, block=False)
            except:
                pass  # Skip if queue full
            
            # Update dashboard data
            stream_info['dashboard_data'] = get_dashboard_data()
            
            # Stream frame to client
            success, jpeg_frame = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                try:
                    await stream_info["frame_queue"].put(jpeg_frame.tobytes())
                except:
                    pass
            
            if stream_info.get("stop_requested", False):
                break
            
            await asyncio.sleep(0.03)  # ~30 FPS
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        if filename in active_streams:
            active_streams[filename]["status"] = "error"
            active_streams[filename]["error"] = str(e)
    
    finally:
        if cap:
            cap.release()
        
        # Close video writer in I/O process
        io_queue.put({
            'type': 'close_video',
            'session_id': session_id
        })
        
        # Send end signal
        if filename in active_streams and not active_streams[filename].get("stop_requested", False):
            try:
                await active_streams[filename]["frame_queue"].put("END")
            except:
                pass
        
        print(f"✅ Video processing completed: {filename}")


async def process_webcam_frames(session_id: str, webcam_streams: dict, camera_view: str = "Front"):
    """Process webcam frames with optimized pipeline"""
    global io_queue
    
    initialize_io_process()
    reset_dashboard()
    
    gallery = load_gallery_embeddings()
    detector = FaceDetector(gallery, camera_view=camera_view)
    
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Cannot open webcam")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        stream_info = webcam_streams[session_id]
        stream_info["processed_frames"] = 0
        stream_info["webcam_capture"] = cap
        
        frame_id = 0
        print(f"📷 Webcam started: {session_id}")
        print(f"📹 Camera View: {camera_view}")
        
        while True:
            if stream_info.get("stop_requested", False):
                print(f"🛑 Stop requested for webcam: {session_id}")
                break
            
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame from webcam")
                break
            
            frame_id += 1
            stream_info["processed_frames"] = frame_id
            
            # Detection and recognition
            if detector.should_process_frame():
                detections = detector.detect_and_recognize(frame, frame_id)
                update_dashboard(detections)
                
                # Queue I/O tasks
                for det in detections:
                    if det['status'] == 'recognized' and det['subject']:
                        try:
                            person_id = int(det['subject'])
                            io_queue.put({
                                'type': 'attendance',
                                'person_id': person_id
                            }, block=False)
                        except:
                            pass
                
                annotated = FrameAnnotator.annotate_frame(
                    frame, detections, frame_id, camera_view,
                    {
                        'total': dashboard_stats['total_faces'],
                        'known': dashboard_stats['known_faces'],
                        'unknown': dashboard_stats['unknown_faces']
                    }
                )
            else:
                annotated = frame.copy()
                cv2.putText(annotated, f"Frame: {frame_id} | View: {camera_view} | SKIPPED", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            stream_info['dashboard_data'] = get_dashboard_data()
            
            success, jpeg_frame = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                try:
                    stream_info["frame_queue"].put_nowait(jpeg_frame.tobytes())
                except:
                    pass
            
            if stream_info.get("stop_requested", False):
                break
            
            await asyncio.sleep(0.03)
    
    except Exception as e:
        print(f"❌ Webcam processing error: {e}")
        if session_id in webcam_streams:
            webcam_streams[session_id]["status"] = "error"
            webcam_streams[session_id]["error"] = str(e)
    
    finally:
        if cap:
            cap.release()
            print(f"📷 Webcam released: {session_id}")
        
        if session_id in webcam_streams:
            webcam_streams[session_id].pop("webcam_capture", None)
        
        print(f"✅ Webcam processing completed: {session_id}")


# Export functions for main.py
__all__ = [
    'process_video_realtime_frames',
    'process_webcam_frames',
    'get_dashboard_data',
    'initialize_io_process',
    'shutdown_io_process',
    'load_gallery_embeddings'
]