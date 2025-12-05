# face_processor_optimized.py - FIXED: 480x640 resolution for all outputs
"""
FIXED VERSION:
1. All frames resized to 480x640 for streaming and output video
2. Maintains YOLO detection at original resolution for accuracy
3. Resizes annotated frames before streaming/saving
"""
import cv2
import asyncio
import multiprocessing as mp
from multiprocessing import Process, Queue
import numpy as np
from datetime import datetime
import uuid
import json
import psycopg2

from face_detection_module import FaceDetector, FrameAnnotator
from io_operations_module import start_io_process

# FIXED: Target resolution for streaming and output
STREAM_WIDTH =  640
STREAM_HEIGHT =  640

GLOBAL_DB_CONFIG = None
current_gallery = []
gallery_loaded_for_config = None
in_memory_embeddings = {}

def set_database_config(db_config):
    global GLOBAL_DB_CONFIG, in_memory_embeddings
    GLOBAL_DB_CONFIG = db_config
    in_memory_embeddings = {}
    clear_gallery_embeddings()
    
    if db_config:
        print(f"✅ Database config set, gallery cleared for reload")
    else:
        print("⚠️ Database config cleared, running without database (in-memory mode)")
        print("⚠️ All in-memory embeddings cleared - starting fresh")

def get_database_config():
    return GLOBAL_DB_CONFIG

def clear_gallery_embeddings():
    global current_gallery, gallery_loaded_for_config
    current_gallery = []
    gallery_loaded_for_config = None
    print("✅ Gallery embeddings cleared")

def clear_all_temporary_data():
    global in_memory_embeddings, current_gallery, gallery_loaded_for_config
    in_memory_embeddings = {}
    current_gallery = []
    gallery_loaded_for_config = None
    print("✅ All temporary data cleared")

def add_in_memory_embedding(person_id, embedding):
    in_memory_embeddings[str(person_id)] = embedding
    print(f"✅ Added in-memory embedding for person {person_id}")

def remove_in_memory_embedding(person_id):
    person_id_str = str(person_id)
    if person_id_str in in_memory_embeddings:
        del in_memory_embeddings[person_id_str]
        print(f"✅ Removed in-memory embedding for person {person_id}")

def get_in_memory_embeddings():
    return in_memory_embeddings

def load_gallery_embeddings():
    global current_gallery, gallery_loaded_for_config
    
    db_config = get_database_config()
    
    if not db_config:
        gallery = []
        for person_id, embedding in in_memory_embeddings.items():
            try:
                norm = np.linalg.norm(embedding)
                if norm == 0:
                    continue
                normalized_embedding = embedding / norm
                
                gallery.append({
                    "subject": str(person_id),
                    "embedding": normalized_embedding
                })
            except Exception as e:
                print(f"Error processing in-memory embedding for person {person_id}: {e}")
                continue
        
        current_gallery = gallery
        gallery_loaded_for_config = "IN_MEMORY"
        print(f"✅ Loaded {len(gallery)} in-memory gallery embeddings")
        return current_gallery
    
    config_key = f"{db_config.get('host', '')}:{db_config.get('dbname', '')}"
    if gallery_loaded_for_config == config_key:
        return current_gallery
    
    try:
        conn = psycopg2.connect(**db_config)
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
    
    current_gallery = gallery
    gallery_loaded_for_config = config_key
    print(f"✅ Loaded {len(gallery)} gallery embeddings from {config_key}")
    return current_gallery


io_queue = None
io_process = None

dashboard_stats = {
    'total_faces': 0,
    'known_faces': 0,
    'unknown_faces': 0,
    'known_faces_list': [],
    'unknown_faces_list': []
}

tracked_id = set()


def initialize_io_process():
    global io_queue, io_process
    
    if io_queue is None:
        db_config = get_database_config()
        io_queue = Queue(maxsize=1000)
        io_process = Process(target=start_io_process, args=(io_queue, db_config), daemon=True)
        io_process.start()
        print("✅ IO Process started")


def shutdown_io_process():
    global io_queue, io_process
    
    if io_queue:
        try:
            io_queue.put({'type': 'shutdown'})
        except:
            pass
    
    if io_process:
        io_process.join(timeout=5)
        if io_process.is_alive():
            io_process.terminate()
        print("✅ IO Process stopped")


def reset_dashboard():
    global dashboard_stats, tracked_id
    tracked_id = set()
    dashboard_stats = {
        'total_faces': 0,
        'known_faces': 0,
        'unknown_faces': 0,
        'known_faces_list': [],
        'unknown_faces_list': []
    }


def update_dashboard(detections):
    global dashboard_stats, tracked_id
    
    for det in detections:
        if det['status'] == 'recognized' and det['track_id'] not in tracked_id:
            tracked_id.add(det['track_id'])
            dashboard_stats['total_faces'] += 1
            dashboard_stats['known_faces'] += 1
            
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
    return {
        'total_faces': dashboard_stats['total_faces'],
        'known_faces_num': dashboard_stats['known_faces'],
        'unknown_faces_num': dashboard_stats['unknown_faces'],
        'known_faces': dashboard_stats['known_faces_list'],
        'unknown_faces': dashboard_stats['unknown_faces_list']
    }


async def process_video_realtime_frames(input_path: str, filename: str, 
                                       active_streams: dict, camera_view: str = "Front"):
    """FIXED: Process video with 480x640 output resolution"""
    global io_queue
    
    initialize_io_process()
    reset_dashboard()
    
    session_id = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    gallery = load_gallery_embeddings()
    detector = FaceDetector(gallery, camera_view=camera_view)
    
    cap = None
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        stream_info = active_streams[filename]
        stream_info['total_frames'] = total_frames
        stream_info['processed_frames'] = 0
        stream_info['session_id'] = session_id
        
        output_path = f"/tmp/face_recognition/processed_videos/processed_{filename}"
        
        # FIXED: Initialize video writer with 480x640 resolution
        io_queue.put({
            'type': 'init_video',
            'session_id': session_id,
            'output_path': output_path,
            'fps': fps,
            'frame_size': (STREAM_WIDTH, STREAM_HEIGHT)  # FIXED: Use target resolution
        })
        
        frame_id = 0
        print(f"🚀 Starting video processing: {filename}")
        print(f"📹 Camera View: {camera_view}")
        print(f"📐 Output Resolution: {STREAM_WIDTH}x{STREAM_HEIGHT}")
        print(f"⚡ Total frames: {total_frames}")
        
        while True:
            if stream_info.get("stop_requested", False):
                print(f"🛑 Stop requested for: {filename}")
                stream_info["status"] = "stopped"
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            stream_info['processed_frames'] = frame_id
            
            if detector.should_process_frame():
                # Detect on full resolution frame for accuracy
                detections = detector.detect_and_recognize(frame, frame_id)
                update_dashboard(detections)
                
                # Mark attendance
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
                annotated = frame.copy()
                cv2.putText(annotated, f"Frame: {frame_id} | View: {camera_view} | SKIPPED", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # FIXED: Resize frame to 480x640 for streaming and output
            resized_frame = cv2.resize(annotated, (STREAM_WIDTH, STREAM_HEIGHT))
            
            # Write resized frame to output video
            try:
                io_queue.put({
                    'type': 'write_frame',
                    'session_id': session_id,
                    'frame': resized_frame.copy()  # FIXED: Write resized frame
                }, block=False)
            except:
                pass
            
            stream_info['dashboard_data'] = get_dashboard_data()
            
            # FIXED: Stream resized frame
            success, jpeg_frame = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                try:
                    await stream_info["frame_queue"].put(jpeg_frame.tobytes())
                except:
                    pass
            
            if stream_info.get("stop_requested", False):
                stream_info["status"] = "stopped"
                break
            
            await asyncio.sleep(0.03)
        
        if not stream_info.get("stop_requested", False):
            stream_info["status"] = "completed"
            print(f"✅ Video processing completed: {filename}")
        else:
            print(f"🛑 Video processing stopped by user: {filename}")
        
    except Exception as e:
        print(f"❌ Video processing error: {e}")
        if filename in active_streams:
            active_streams[filename]["status"] = "error"
            active_streams[filename]["error"] = str(e)
    
    finally:
        if cap:
            cap.release()
        
        try:
            io_queue.put({
                'type': 'close_video',
                'session_id': session_id
            })
        except:
            pass
        
        if filename in active_streams:
            try:
                await active_streams[filename]["frame_queue"].put("END")
            except:
                pass


async def process_webcam_frames(session_id: str, webcam_streams: dict, camera_view: str = "Front"):
    """FIXED: Process webcam with 480x640 output resolution"""
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
        
        # FIXED: Set webcam to capture at 640x480
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        stream_info = webcam_streams[session_id]
        stream_info["processed_frames"] = 0
        stream_info["webcam_capture"] = cap
        
        frame_id = 0
        print(f"📷 Webcam started: {session_id}")
        print(f"📹 Camera View: {camera_view}")
        print(f"📐 Resolution: {STREAM_WIDTH}x{STREAM_HEIGHT}")
        
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
            
            if detector.should_process_frame():
                detections = detector.detect_and_recognize(frame, frame_id)
                update_dashboard(detections)
                
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
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            stream_info['dashboard_data'] = get_dashboard_data()
            
            # Frame is already at target resolution from webcam
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


__all__ = [
    'process_video_realtime_frames',
    'process_webcam_frames',
    'get_dashboard_data',
    'initialize_io_process',
    'shutdown_io_process',
    'load_gallery_embeddings',
    'set_database_config',
    'get_database_config',
    'add_in_memory_embedding',
    'remove_in_memory_embedding',
    'get_in_memory_embeddings',
    'clear_gallery_embeddings',
    'clear_all_temporary_data'
]