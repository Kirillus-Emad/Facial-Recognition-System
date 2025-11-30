# register_faces.py - Face Registration and Deletion
"""
Handles face registration and deletion operations
Extracted from face_processor_realtime.py
"""
import cv2
import numpy as np
import psycopg2
import json
from ultralytics import YOLO
from deepface import DeepFace
from tensorflow.keras.models import load_model

# Initialize models
yolo_model = YOLO("yolov8n-face-lindevs.pt")
arcface_model = load_model('arcface_v3_256D.h5')

# Database configuration
DB_CONFIG = {
    'host': "ep-lingering-glade-aghpvil3-pooler.c-2.eu-central-1.aws.neon.tech",
    'dbname': "neondb",
    'user': "neondb_owner",
    'password': "npg_AO7fphz9ieod",
    'sslmode': "require"
}


def get_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        dbname=DB_CONFIG['dbname'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        sslmode=DB_CONFIG['sslmode']
    )


def person_exists(person_id):
    """Check if person exists in database"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM persons WHERE id = %s", (person_id,))
        result = cur.fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        raise Exception(f"Database check error: {e}")


def create_person(person_id):
    """Insert new person into database"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO persons (id) VALUES (%s)", (person_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        raise Exception(f"Error inserting new person: {e}")


def insert_embedding(person_id, embedding):
    """Insert or update embedding in database"""
    try:
        emb_list = embedding.tolist()
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO person_embeddings (person_id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (person_id)
            DO UPDATE SET embedding = EXCLUDED.embedding;
        """, (person_id, emb_list))

        conn.commit()
        conn.close()
    except Exception as e:
        raise Exception(f"Database insert error: {e}")


def preprocess_face_enhancement(face_region):
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
    except Exception as e:
        print(f"Face enhancement error: {e}")
        return face_region


def compute_embedding(img, detector_backend='retinaface'):
    """Compute face embedding"""
    try:
        img = DeepFace.extract_faces(
            img_path=img, align=True, detector_backend=detector_backend,
            enforce_detection=False
        )[0]["face"]
        
        img = cv2.resize(img, (112, 112))
        img = np.expand_dims(img, 0)
        emb = arcface_model.predict(img, verbose=0)
        return emb[0]
    except Exception as e:
        print(f"Error computing embedding: {e}")
        return None


def register_multiple_faces_to_database(person_id: str, image_paths: list):
    """
    Register a new person with multiple face images (up to 5).
    Averages embeddings from all provided images.
    """
    try:
        print(f"🔍 Registering person {person_id} with {len(image_paths)} images")
        
        if len(image_paths) == 0:
            return {"success": False, "error": "No images provided"}
        
        if len(image_paths) > 5:
            return {"success": False, "error": "Maximum 5 images allowed"}
        
        try:
            person_id_int = int(person_id)
        except ValueError:
            return {"success": False, "error": "Person ID must be numeric"}
        
        embeddings_list = []
        processed_count = 0
        
        # Process each image
        for idx, image_path in enumerate(image_paths):
            print(f"🔍 Processing image {idx+1}/{len(image_paths)}: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"⚠️ Cannot load image {image_path}")
                continue
            
            # Detect face
            results = yolo_model.predict(img, conf=0.2, imgsz=640)[0]
            
            if results.boxes is None or len(results.boxes) == 0:
                print(f"⚠️ No face detected in {image_path}")
                continue
            
            # Get first face
            box = results.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            h_frame, w_frame, _ = img.shape
            
            # Extract face with padding
            face = img[max(0, y1-35):min(h_frame, y2+25), 
                      max(0, x1-15):min(w_frame, x2+30)]
            
            if face is None or face.size == 0:
                print(f"⚠️ Invalid face region in {image_path}")
                continue
            
            # Apply enhancement
            enhanced_face = preprocess_face_enhancement(face)
            
            # Compute embedding
            embedding = compute_embedding(enhanced_face, 'retinaface')
            
            if embedding is None:
                print(f"⚠️ Failed to compute embedding for {image_path}")
                continue
            
            embeddings_list.append(embedding)
            processed_count += 1
            print(f"✅ Successfully processed image {idx+1}")
        
        if processed_count == 0:
            return {
                "success": False,
                "error": "No valid face embeddings could be extracted"
            }
        
        print(f"✅ Successfully processed {processed_count} images")
        
        # Average embeddings
        embedding_sum = np.sum(embeddings_list, axis=0)
        averaged_embedding = embedding_sum / np.linalg.norm(embedding_sum)
        
        # Create person if not exists
        try:
            if not person_exists(person_id_int):
                print(f"👤 Creating new person with ID: {person_id_int}")
                create_person(person_id_int)
            else:
                print(f"👤 Person {person_id_int} exists, updating embedding")
        except Exception as e:
            return {"success": False, "error": f"Error creating person: {str(e)}"}
        
        # Insert/update embedding
        try:
            print(f"💾 Inserting/updating embedding for person {person_id_int}")
            insert_embedding(person_id_int, averaged_embedding)
        except Exception as e:
            return {"success": False, "error": f"Error inserting embedding: {str(e)}"}
        
        # Reload gallery in face_processor_optimized
        try:
            from face_processor_optimized import load_gallery_embeddings
            print("🔄 Reloading gallery embeddings...")
            load_gallery_embeddings()
        except:
            pass
        
        print(f"✅ Successfully registered person {person_id_int}")
        return {
            "success": True,
            "message": f"Successfully registered person {person_id_int}",
            "person_id": person_id_int,
            "images_processed": processed_count,
            "images_provided": len(image_paths)
        }
        
    except Exception as e:
        print(f"❌ Registration error: {e}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }


def delete_person_from_database(person_id: int):
    """
    Delete a person using CASCADE.
    PostgreSQL handles all related deletions automatically.
    """
    try:
        print(f"🗑️ Attempting to delete person {person_id}")
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Get counts before deletion
        cur.execute("""
            SELECT 
                (SELECT COUNT(*) FROM person_embeddings WHERE person_id = %s) as emb_count,
                (SELECT COUNT(*) FROM attendance WHERE person_id = %s) as att_count
        """, (person_id, person_id))
        counts = cur.fetchone()
        embeddings_count = counts[0] if counts else 0
        attendance_count = counts[1] if counts else 0
        
        # Delete with CASCADE
        cur.execute("DELETE FROM persons WHERE id = %s RETURNING id", (person_id,))
        deleted_person = cur.fetchone()
        
        if not deleted_person:
            cur.close()
            conn.close()
            return {
                "success": False,
                "error": f"Person with ID {person_id} does not exist"
            }
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Reload gallery
        try:
            from face_processor_optimized import load_gallery_embeddings
            print("🔄 Reloading gallery embeddings...")
            load_gallery_embeddings()
        except:
            pass
        
        print(f"✅ Successfully deleted person {person_id}")
        print(f"   - Embeddings deleted: {embeddings_count}")
        print(f"   - Attendance records deleted: {attendance_count}")
        
        return {
            "success": True,
            "message": f"Successfully deleted person {person_id}",
            "person_id": person_id,
            "embeddings_deleted": embeddings_count,
            "attendance_deleted": attendance_count
        }
        
    except Exception as e:
        print(f"❌ Delete error: {e}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }