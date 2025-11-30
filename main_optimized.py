# main_optimized_complete.py - COMPLETE INTEGRATION
"""
Complete integration with all features from face_processor_realtime.py and main.py
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
import asyncio
from typing import Dict, Any, List
from pathlib import Path
import base64
import cv2

# Import optimized modules
from face_processor_optimized import (
    process_video_realtime_frames,
    process_webcam_frames,
    get_dashboard_data,
    initialize_io_process,
    shutdown_io_process,
    load_gallery_embeddings
)

# Import simplified MLOps
from mlops import mlops_simple_monitor
from mlops.retraining_with_finetuning import trigger_manual_retraining

app = FastAPI(title="Face Recognition System - Complete")

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed_videos", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global stream management
active_streams: Dict[str, Any] = {}
webcam_streams: Dict[str, Any] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    print("=" * 80)
    print("🚀 STARTING COMPLETE FACE RECOGNITION SYSTEM")
    print("=" * 80)
    
    # Initialize I/O process
    initialize_io_process()
    
    # Initialize MLOps
    try:
        mlops_simple_monitor.mlops_monitor.initialize_mlflow()
    except Exception as e:
        print(f"⚠️ MLflow initialization warning: {e}")
    
    print("✅ System started successfully")
    print("📊 MLflow Dashboard: http://localhost:5000")
    print("🌐 Web Interface: http://localhost:8000")
    print("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("=" * 80)
    print("🛑 SHUTTING DOWN SYSTEM")
    print("=" * 80)
    
    # Shutdown I/O process
    shutdown_io_process()
    
    print("✅ System shutdown complete")
    print("=" * 80)


# ============ WEB UI ENDPOINTS ============

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve main web interface"""
    index_path = Path("templates/index.html")
    
    if not index_path.exists():
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 50px auto;
            background: white;
            color: #333;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        }
        h1 { color: #667eea; }
        .status { margin: 20px 0; padding: 20px; background: #f0f0f0; border-radius: 10px; }
        .error { background: #fee; color: #c00; }
        a { color: #667eea; text-decoration: none; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Face Recognition System</h1>
        <div class="status error">
            <h3>⚠️ Template File Missing</h3>
            <p>Please ensure <code>templates/index.html</code> exists.</p>
            <p>Copy the provided index.html file to the templates directory.</p>
        </div>
        <div class="status">
            <h3>System Status</h3>
            <p>✅ Backend: Running</p>
            <p>❌ Frontend: Template missing</p>
        </div>
        <div class="status">
            <h3>Quick Links</h3>
            <p><a href="/mlops/status">📊 MLOps Status</a></p>
            <p><a href="/health">💚 Health Check</a></p>
            <p><a href="/docs">📚 API Documentation</a></p>
        </div>
    </div>
</body>
</html>
        """)
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/mlops-dashboard", response_class=HTMLResponse)
async def mlops_dashboard(request: Request):
    """Serve MLOps dashboard"""
    dashboard_path = Path("templates/mlops_dashboard.html")
    
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="MLOps dashboard template not found")
    
    return templates.TemplateResponse("mlops_dashboard.html", {"request": request})


# ============ VIDEO PROCESSING ENDPOINTS ============

@app.post("/upload-video/")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    camera_view: str = Form("Front")
):
    """Upload and process video"""
    try:
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Only video files allowed")
        
        if camera_view not in ["Front", "Left", "Right"]:
            raise HTTPException(status_code=400, detail="Invalid camera view")
        
        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        input_path = f"uploads/{unique_filename}"
        
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialize stream info
        active_streams[unique_filename] = {
            "status": "processing",
            "frame_queue": asyncio.Queue(),
            "total_frames": 0,
            "processed_frames": 0,
            "stop_requested": False,
            "processing_task": None,
            "stream_type": "video",
            "camera_view": camera_view,
            "dashboard_data": {}
        }
        
        # Start processing
        processing_task = asyncio.create_task(
            process_video_realtime_frames(input_path, unique_filename, active_streams, camera_view)
        )
        active_streams[unique_filename]["processing_task"] = processing_task
        
        return {
            "message": "Video uploaded successfully",
            "filename": unique_filename,
            "status": "processing",
            "camera_view": camera_view,
            "stream_url": f"/live-stream/{unique_filename}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/start-webcam/")
async def start_webcam(
    background_tasks: BackgroundTasks,
    camera_view: str = Form("Front")
):
    """Start webcam processing"""
    try:
        if camera_view not in ["Front", "Left", "Right"]:
            raise HTTPException(status_code=400, detail="Invalid camera view")
        
        session_id = f"webcam_{uuid.uuid4()}"
        
        webcam_streams[session_id] = {
            "status": "processing",
            "frame_queue": asyncio.Queue(),
            "processed_frames": 0,
            "stop_requested": False,
            "processing_task": None,
            "stream_type": "webcam",
            "camera_view": camera_view,
            "dashboard_data": {}
        }
        
        processing_task = asyncio.create_task(
            process_webcam_frames(session_id, webcam_streams, camera_view)
        )
        webcam_streams[session_id]["processing_task"] = processing_task
        
        return {
            "message": "Webcam started successfully",
            "session_id": session_id,
            "status": "processing",
            "camera_view": camera_view,
            "stream_url": f"/live-stream/{session_id}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/stop-processing/{stream_id}")
async def stop_processing(stream_id: str):
    """Stop processing"""
    stream_info = None
    
    if stream_id in active_streams:
        stream_info = active_streams[stream_id]
    elif stream_id in webcam_streams:
        stream_info = webcam_streams[stream_id]
    else:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    stream_info["stop_requested"] = True
    stream_info["status"] = "stopped"
    
    # Force webcam release
    if stream_id.startswith("webcam_") and "webcam_capture" in stream_info:
        try:
            cap = stream_info["webcam_capture"]
            if cap:
                cap.release()
            stream_info.pop("webcam_capture", None)
        except:
            pass
    
    # Cancel task
    if stream_info["processing_task"]:
        stream_info["processing_task"].cancel()
        try:
            await stream_info["processing_task"]
        except asyncio.CancelledError:
            pass
    
    return {"message": "Processing stopped", "status": "stopped"}


@app.get("/live-stream/{stream_id}")
async def live_stream_frames(stream_id: str):
    """Stream processed frames"""
    stream_info = None
    if stream_id in active_streams:
        stream_info = active_streams[stream_id]
    elif stream_id in webcam_streams:
        stream_info = webcam_streams[stream_id]
    else:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    async def generate_frames():
        while True:
            try:
                if stream_info.get("stop_requested", False):
                    break
                
                frame_data = await asyncio.wait_for(
                    stream_info["frame_queue"].get(), timeout=5.0
                )
                
                if frame_data == "END":
                    break
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            except asyncio.TimeoutError:
                if stream_info["status"] in ["completed", "stopped", "error"]:
                    break
                continue
            except Exception as e:
                print(f"Stream error: {e}")
                break
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stream-progress/{stream_id}")
async def get_stream_progress(stream_id: str):
    """Get processing progress"""
    stream_info = None
    if stream_id in active_streams:
        stream_info = active_streams[stream_id]
    elif stream_id in webcam_streams:
        stream_info = webcam_streams[stream_id]
    else:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    progress = 0
    if stream_info.get("total_frames", 0) > 0:
        progress = min(100, int((stream_info["processed_frames"] / stream_info["total_frames"]) * 100))
    
    return {
        "status": stream_info["status"],
        "progress": progress,
        "processed_frames": stream_info["processed_frames"],
        "total_frames": stream_info.get("total_frames", 0),
        "can_stop": stream_info["status"] == "processing",
        "stream_type": stream_info.get("stream_type", "video"),
        "camera_view": stream_info.get("camera_view", "Front"),
        "dashboard_data": stream_info.get("dashboard_data", {})
    }


@app.get("/dashboard-data/{stream_id}")
async def get_dashboard_stats(stream_id: str):
    """Get dashboard data"""
    stream_info = None
    if stream_id in active_streams:
        stream_info = active_streams[stream_id]
    elif stream_id in webcam_streams:
        stream_info = webcam_streams[stream_id]
    else:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    return stream_info.get("dashboard_data", {})


@app.get("/known-faces/{stream_id}")
async def get_known_faces(stream_id: str):
    """Get known faces for a stream"""
    stream_info = None
    if stream_id in active_streams:
        stream_info = active_streams[stream_id]
    elif stream_id in webcam_streams:
        stream_info = webcam_streams[stream_id]
    else:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    dashboard_data = stream_info.get("dashboard_data", {})
    known_faces = dashboard_data.get("known_faces", [])
    
    # Convert to proper format
    known_faces_with_images = []
    for face_data in known_faces:
        known_faces_with_images.append({
            "subject": face_data["subject"],
            "face_image": f"data:image/jpeg;base64,{face_data['face_image']}",
            "track_id": face_data.get("track_id", "N/A"),
            "frame_id": face_data.get("frame_id", "N/A"),
            "score": face_data.get("score", "N/A")
        })
    
    return {"known_faces": known_faces_with_images}


@app.get("/unknown-faces/{stream_id}")
async def get_unknown_faces(stream_id: str):
    """Get unknown faces for a stream"""
    stream_info = None
    if stream_id in active_streams:
        stream_info = active_streams[stream_id]
    elif stream_id in webcam_streams:
        stream_info = webcam_streams[stream_id]
    else:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    dashboard_data = stream_info.get("dashboard_data", {})
    unknown_faces = dashboard_data.get("unknown_faces", [])
    
    # Convert to proper format
    unknown_faces_with_images = []
    for face_data in unknown_faces:
        unknown_faces_with_images.append({
            "face_image": f"data:image/jpeg;base64,{face_data['face_image']}",
            "track_id": face_data.get("track_id", "N/A"),
            "frame_id": face_data.get("frame_id", "N/A"),
            "score": face_data.get("score", "N/A")
        })
    
    return {"unknown_faces": unknown_faces_with_images}


@app.get("/video/{filename}")
async def get_processed_video(filename: str):
    """Get processed video file"""
    video_path = f"processed_videos/processed_{filename}"
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"processed_{filename}"
    )


# ============ FACE REGISTRATION & DELETION ============

@app.post("/register-face-multiple/")
async def register_face_multiple(
    person_id: int = Form(...),
    images: List[UploadFile] = File(...)
):
    """Register a new face with multiple images"""
    try:
        if len(images) == 0:
            raise HTTPException(status_code=400, detail="At least one image is required")
        
        if len(images) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 images allowed")
        
        print(f"📥 Received {len(images)} images for person ID: {person_id}")
        
        # Import registration function
        from register_faces import register_multiple_faces_to_database
        
        # Save temporary files
        temp_paths = []
        for idx, image in enumerate(images):
            if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                for temp_path in temp_paths:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Image {idx+1} is not a valid image file"
                )
            
            temp_filename = f"temp_{uuid.uuid4()}_{idx}{os.path.splitext(image.filename)[1]}"
            temp_path = f"uploads/{temp_filename}"
            
            with open(temp_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
            
            temp_paths.append(temp_path)
        
        # Register faces
        result = register_multiple_faces_to_database(str(person_id), temp_paths)
        
        # Cleanup
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if result["success"]:
            return {
                "message": f"Successfully registered person with ID {person_id}",
                "details": result
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Registration failed"))
    
    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        try:
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.delete("/delete-person/{person_id}")
async def delete_person(person_id: int):
    """Delete a person and their embeddings"""
    try:
        print(f"🗑️ Deleting person ID: {person_id}")
        
        # Import deletion function
        from register_faces import delete_person_from_database
        
        result = delete_person_from_database(person_id)
        
        if result["success"]:
            return {
                "message": f"Successfully deleted person with ID {person_id}",
                "details": result
            }
        else:
            raise HTTPException(status_code=404, detail=result.get("error", "Person not found"))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============ ATTENDANCE REPORTS ============

@app.get("/generate-attendance-report/")
async def generate_attendance_report(report_type: str = "monthly"):
    """Generate and download attendance report"""
    try:
        # Import report function
        from attendance_reports import generate_attendance_report_csv
        
        result = generate_attendance_report_csv(report_type)
        
        if not result["success"]:
            raise HTTPException(status_code=404, detail=result.get("error", "Report generation failed"))
        
        return FileResponse(
            result["filepath"],
            media_type="text/csv",
            filename=result["filename"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============ MLOPS ENDPOINTS ============

@app.get("/mlops/status")
async def get_mlops_status():
    """Get MLOps system status"""
    try:
        status = mlops_simple_monitor.get_status()
        status["scheduler_running"] = False
        status["mlflow_active"] = True
        return status
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "scheduler_running": False,
            "mlflow_active": False
        }


@app.get("/mlops/metrics")
async def get_mlops_metrics(hours: int = 24):
    """Get performance metrics"""
    try:
        return mlops_simple_monitor.get_metrics(hours)
    except Exception as e:
        return {"error": str(e)}


@app.get("/mlops/metrics/trends")
async def get_performance_trends(days: int = 7):
    """Get performance trends"""
    try:
        return mlops_simple_monitor.get_trends(days)
    except Exception as e:
        return {"error": str(e), "dates": [], "accuracy": [], "avg_similarity": []}


@app.post("/mlops/retrain")
async def trigger_retraining():
    """Trigger manual retraining"""
    try:
        result = trigger_manual_retraining()
        
        if result.get('success'):
            return {
                "status": "success",
                "message": "Retraining completed successfully",
                "results": result
            }
        else:
            return {
                "status": "error",
                "message": result.get('error', 'Retraining failed')
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/mlops/report")
async def get_performance_report():
    """Get detailed performance report"""
    try:
        report = mlops_simple_monitor.generate_report()
        return {"report": report}
    except Exception as e:
        return {"report": f"Error generating report: {str(e)}"}


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "message": "System operational",
        "active_streams": len(active_streams),
        "active_webcams": len(webcam_streams)
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 STARTING COMPLETE FACE RECOGNITION SYSTEM")
    print("="*80)
    print("\n📋 Checking requirements...")
    
    # Check if templates exist
    if not Path("templates/index.html").exists():
        print("⚠️  WARNING: templates/index.html not found!")
    
    if not Path("static/script.js").exists():
        print("⚠️  WARNING: static/script.js not found!")
    
    if not Path("static/style.css").exists():
        print("⚠️  WARNING: static/style.css not found!")
    
    print("\n🌐 Starting web server...")
    print("   URL: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    print("\n   Press Ctrl+C to stop\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )