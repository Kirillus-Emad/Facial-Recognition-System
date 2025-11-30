# start_optimized_system.py
"""
Quick start script for optimized Face Recognition System
Starts MLflow + FastAPI application
"""
import subprocess
import sys
import time
import logging
from pathlib import Path
import signal
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def kill_process_on_port(port):
    """Kill any process using the specified port"""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections():
                    if conn.laddr.port == port:
                        logger.info(f"Killing process {proc.pid} on port {port}")
                        proc.kill()
                        time.sleep(1)
            except:
                pass
    except Exception as e:
        logger.warning(f"Error checking port {port}: {e}")


def start_mlflow_server():
    """Start MLflow tracking server"""
    try:
        # Kill any existing MLflow process on port 5000
        kill_process_on_port(5000)
        
        logger.info("Starting MLflow tracking server...")
        
        # Create mlruns directory
        Path("mlruns").mkdir(exist_ok=True)
        
        mlflow_process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "server",
            "--host", "127.0.0.1",
            "--port", "5000",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for MLflow to start
        time.sleep(5)
        
        logger.info("✅ MLflow server started at http://localhost:5000")
        return mlflow_process
        
    except Exception as e:
        logger.error(f"❌ Failed to start MLflow server: {e}")
        return None


def setup_database():
    """Setup database schema"""
    try:
        logger.info("Setting up database schema...")
        from database.database_setup import setup_database_schema
        
        if setup_database_schema():
            logger.info("✅ Database schema setup completed")
            return True
        else:
            logger.error("❌ Database schema setup failed")
            return False
    except Exception as e:
        logger.error(f"❌ Database setup error: {e}")
        return False


def start_fastapi_app():
    """Start FastAPI application"""
    try:
        logger.info("Starting FastAPI application...")
        
        from main_optimized import app
        import uvicorn
        
        # Run the server
        uvicorn.run(
            app,
            host="127.0.0.1",  # ← CHANGED
            port=8000,
            log_level="info"
        )
        return True
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
        return True
    except Exception as e:
        logger.error(f"❌ FastAPI application error: {e}")
        return False


def display_startup_info():
    """Display system startup information"""
    print("\n" + "="*80)
    print("🚀 OPTIMIZED FACE RECOGNITION SYSTEM")
    print("="*80)
    print("\n📋 SYSTEM INFORMATION:")
    print(f"   Version: 2.0 (Optimized)")
    print(f"   Architecture: Multi-process (Detection + I/O)")
    print(f"   MLOps: Simplified monitoring + Fine-tuning")
    
    print("\n🔗 ACCESS POINTS:")
    print(f"   🌐 Web Interface: http://localhost:8000")
    print(f"   📊 MLflow Dashboard: http://localhost:5000")
    print(f"   📈 MLOps Status: http://localhost:8000/mlops/status")
    print(f"   🔄 Trigger Retraining: http://localhost:8000/mlops/retrain (POST)")
    
    print("\n⚡ PERFORMANCE:")
    print(f"   Frame Processing: 25-30 FPS (Real-time)")
    print(f"   I/O Operations: Background process (non-blocking)")
    print(f"   Frame Skipping: Every 2 frames (configurable)")
    
    print("\n📊 MONITORING:")
    print(f"   Metrics: Accuracy, FAR, FRR")
    print(f"   Thresholds: Accuracy>95%, FAR<1%, FRR<5%")
    print(f"   Logging: MLflow + Database")
    
    print("\n🎓 MLOPS FEATURES:")
    print(f"   ✅ Real-time monitoring")
    print(f"   ✅ Performance alerts")
    print(f"   ✅ Manual retraining with fine-tuning")
    print(f"   ✅ Triplet loss optimization")
    print(f"   ✅ Threshold auto-tuning")
    
    print("\n💡 QUICK TIPS:")
    print(f"   - Use Front/Left/Right camera views for better accuracy")
    print(f"   - Monitor MLflow for real-time metrics")
    print(f"   - Trigger retraining when accuracy drops")
    print(f"   - Check logs/ directory for detailed logs")
    
    print("\n⌨️  KEYBOARD SHORTCUTS:")
    print(f"   Ctrl+C: Stop system gracefully")
    
    print("\n" + "="*80)
    print("🟢 SYSTEM READY - Press Ctrl+C to stop")
    print("="*80 + "\n")


def main():
    """Main startup function"""
    logger.info("="*80)
    logger.info("STARTING OPTIMIZED SYSTEM")
    logger.info("="*80)
    
    # Setup database first
    if not setup_database():
        logger.error("Database setup failed. Exiting.")
        return
    
    # Start MLflow server
    mlflow_process = start_mlflow_server()
    if not mlflow_process:
        logger.error("MLflow server failed to start. Exiting.")
        return
    
    try:
        # Display startup info
        display_startup_info()
        
        # Start FastAPI app (blocking)
        start_fastapi_app()
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        # Cleanup
        logger.info("Shutting down...")
        
        if mlflow_process:
            try:
                # Kill MLflow process and children
                parent = psutil.Process(mlflow_process.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.terminate()
                parent.terminate()
                
                # Wait for termination
                gone, still_alive = psutil.wait_procs(children + [parent], timeout=5)
                for p in still_alive:
                    p.kill()
                
                logger.info("✅ MLflow server stopped")
            except Exception as e:
                logger.error(f"Error stopping MLflow: {e}")
        
        logger.info("="*80)
        logger.info("✅ SYSTEM SHUTDOWN COMPLETE")
        logger.info("="*80)


if __name__ == "__main__":
    main()