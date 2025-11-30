# face_detection_module.py - Core Face Detection (High Priority)
"""
HIGH PRIORITY MODULE - Handles only face detection and recognition
Runs in main process with maximum CPU/GPU resources
NO database operations, NO file I/O, NO logging here
"""
import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
from tensorflow.keras.models import load_model
from collections import deque
import time

# Initialize models globally (loaded once)
yolo_model = YOLO("yolov8n-face-lindevs.pt")
arcface_model = load_model('arcface_v3_256D.h5')

# Frame skipping configuration
PROCESS_EVERY_N_FRAMES = 2


class FaceDetector:
    """High-performance face detection and recognition"""
    
    def __init__(self, gallery_embeddings, camera_view="Front", threshold=1.04):
        self.gallery_embeddings = gallery_embeddings
        self.camera_view = camera_view
        self.threshold = threshold
        self.frame_counter = 0
        self.tracker_cache = {}
        
        # Performance optimization: LRU cache for embeddings
        self.embedding_cache = deque(maxlen=100)
        
    def should_process_frame(self):
        """Determine if this frame should be processed"""
        self.frame_counter += 1
        return self.frame_counter % PROCESS_EVERY_N_FRAMES == 0
    
    def is_valid_face_region(self, x1, y1, x2, y2, frame_shape):
        """Check if face bounding box is valid"""
        h_frame, w_frame = frame_shape[:2]
        w, h = x2 - x1, y2 - y1
        
        # Border check
        if x1 <= 2 or y1 <= 2 or x2 >= (w_frame - 2) or y2 >= (h_frame - 2):
            return False, "border"
        
        # Aspect ratio check
        aspect_ratio = w / float(h)
        if aspect_ratio < 0.5 or aspect_ratio > 1.5:
            return False, "aspect"
        
        # Camera view specific checks
        if self.camera_view == 'Front':
            if y2 < 330 and (x1 > 70 or x2 < w_frame - 70):
                return False, "distance"
        elif self.camera_view == 'Right':
            if x2 > w_frame - 210:
                return False, "distance"
        elif self.camera_view == 'Left':
            if x1 < 300 or x2 > w_frame - 150:
                return False, "distance"
        
        return True, None
    
    def is_spoof(self, frame, bbox):
        """Lightweight anti-spoofing check"""
        try:
            x1, y1, x2, y2 = bbox
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return True
            
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Texture check
            focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()
            if focus_measure < 20:
                return True
            
            # Color saturation
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            sat_mean = hsv[:, :, 1].mean()
            if sat_mean < 25:
                return True
            
            # Motion check
            if not hasattr(self, '_last_face'):
                self._last_face = gray
                return False
            
            if self._last_face.shape != gray.shape:
                self._last_face = cv2.resize(self._last_face, (gray.shape[1], gray.shape[0]))
            
            diff = cv2.absdiff(self._last_face, gray)
            motion_score = np.mean(diff)
            self._last_face = gray
            
            if motion_score < 1.5:
                return True
            
            return False
            
        except:
            return False
    
    def preprocess_face(self, face_region):
        """Apply face enhancement"""
        try:
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (112, 112))
            
            gaussian = cv2.GaussianBlur(face_resized, (0, 0), 2.0)
            sharpened = cv2.addWeighted(face_resized, 1.5, gaussian, -0.5, 0)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(sharpened)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            contrast = 1.0 + 0.2
            enhanced = np.clip((enhanced - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
            
            return enhanced
        except:
            return face_region
    
    def compute_embedding(self, face_img):
        """Compute face embedding"""
        try:
            img = DeepFace.extract_faces(
                img_path=face_img,
                align=True,
                detector_backend='retinaface',
                enforce_detection=False
            )[0]["face"]
            
            img = cv2.resize(img, (112, 112))
            img = np.expand_dims(img, 0)
            emb = arcface_model.predict(img, verbose=0)
            return emb[0]
        except:
            return None
    
    def find_best_match(self, embedding):
        """Find best match in gallery"""
        best_score = 20
        best_subject = "Unknown"
        
        for g in self.gallery_embeddings:
            score = np.linalg.norm(g['embedding'] - embedding)
            if score < best_score:
                best_score = score
                best_subject = g["subject"]
        
        if best_score >= self.threshold:
            return "Unknown", best_score
        return best_subject, best_score
    
    def detect_and_recognize(self, frame, frame_id):
        """
        Main detection and recognition function
        Returns: detections list with format:
        [{
            'bbox': (x1, y1, x2, y2),
            'track_id': int,
            'subject': str,
            'score': float,
            'face_crop': np.array,
            'status': str  # 'recognized', 'unknown', 'spoof', 'invalid'
        }]
        """
        detections = []
        h_frame, w_frame = frame.shape[:2]
        
        # Run YOLO detection with tracking
        results = yolo_model.track(frame, persist=True, conf=0.80, imgsz=640, tracker='botsort.yaml')[0]
        
        if results.boxes is None:
            return detections
        
        ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [None] * len(results.boxes)
        
        for box, track_id in zip(results.boxes.xyxy, ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Validate face region
            is_valid, reason = self.is_valid_face_region(x1, y1, x2, y2, frame.shape)
            if not is_valid:
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': track_id,
                    'status': f'invalid_{reason}',
                    'subject': None,
                    'score': None,
                    'face_crop': None
                })
                continue
            
            # Extract face with padding
            face_crop = frame[max(0, y1-35):min(h_frame, y2+25), 
                             max(0, x1-15):min(w_frame, x2+30)]
            
            if face_crop.size == 0:
                continue
            
            # Anti-spoofing check
            if self.is_spoof(frame, (x1, y1, x2, y2)):
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'track_id': track_id,
                    'status': 'spoof',
                    'subject': None,
                    'score': None,
                    'face_crop': face_crop
                })
                continue
            
            # Check tracker cache
            if track_id in self.tracker_cache:
                subject, score, _ = self.tracker_cache[track_id]
            else:
                # Preprocess and recognize
                enhanced_face = self.preprocess_face(face_crop)
                embedding = self.compute_embedding(enhanced_face)
                
                if embedding is None:
                    continue
                
                subject, score = self.find_best_match(embedding)
                self.tracker_cache[track_id] = (subject, score, embedding)
            
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'track_id': track_id,
                'status': 'recognized' if subject != 'Unknown' else 'unknown',
                'subject': subject,
                'score': float(score),
                'face_crop': face_crop
            })
        
        return detections


class FrameAnnotator:
    """Fast frame annotation (runs in main process)"""
    
    @staticmethod
    def annotate_frame(frame, detections, frame_id, camera_view, stats):
        """
        Annotate frame with detection results
        Args:
            frame: input frame
            detections: list of detection dicts
            frame_id: current frame number
            camera_view: camera view name
            stats: dict with total_faces, known_faces, unknown_faces
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            status = det['status']
            
            # Determine color based on status
            if status == 'recognized':
                color = (0, 255, 0)  # Green
                label = f"{det['subject']}: ({det['score']:.2f})"
            elif status == 'unknown':
                color = (0, 0, 255)  # Red
                label = f"Unknown: ({det['score']:.2f})"
            elif status == 'spoof':
                color = (0, 0, 255)  # Red
                label = "Spoof Detected!"
            else:  # invalid
                if 'border' in status:
                    color = (0, 0, 255)  # Red
                elif 'aspect' in status:
                    color = (255, 0, 0)  # Blue
                else:  # distance
                    color = (0, 165, 255)  # Orange
                label = None
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if label:
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add frame info
        cv2.putText(annotated, f"Frame: {frame_id} | View: {camera_view}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add stats
        cv2.putText(annotated, 
                   f"Total: {stats['total']} | Known: {stats['known']} | Unknown: {stats['unknown']}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated