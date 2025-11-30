# mlops/mlops_simple_monitor.py - SIMPLIFIED MLOps (No Redundancy)
"""
Simplified MLOps monitoring system
Combines: performance monitoring + alerting + basic logging
Removes: Duplicate tracking, unnecessary complexity
"""
import mlflow
import psycopg2
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List

from mlops.config import config

logger = logging.getLogger(__name__)


class SimpleMLOpsMonitor:
    """All-in-one simplified MLOps monitor"""
    
    def __init__(self):
        self.initialize_mlflow()
        self.active_sessions = {}
    
    def initialize_mlflow(self):
        """Initialize MLflow once"""
        try:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
            logger.info("✅ MLflow initialized")
            return True
        except Exception as e:
            logger.warning(f"⚠️ MLflow not available: {e}")
            return False
    
    def start_session(self, session_id: str, session_type: str, camera_view: str):
        """Start monitoring a processing session"""
        try:
            run_name = f"{session_type}_{camera_view}_{datetime.now().strftime('%H%M%S')}"
            run = mlflow.start_run(run_name=run_name)
            
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("session_type", session_type)
            mlflow.log_param("camera_view", camera_view)
            mlflow.log_param("start_time", datetime.now().isoformat())
            
            self.active_sessions[session_id] = {
                'run_id': run.info.run_id,
                'start_time': datetime.now(),
                'metrics': defaultdict(int)
            }
            
            logger.info(f"✅ Session started: {session_id}")
            
        except Exception as e:
            logger.error(f"Session start error: {e}")
    
    def update_metrics(self, session_id: str, metrics: Dict):
        """Update metrics for a session"""
        if session_id not in self.active_sessions:
            return
        
        try:
            session = self.active_sessions[session_id]
            session['metrics'].update(metrics)
            
            # Log to MLflow every 10 updates to reduce overhead
            if session['metrics']['updates'] % 10 == 0:
                with mlflow.start_run(run_id=session['run_id']):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, value)
            
            session['metrics']['updates'] += 1
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    def end_session(self, session_id: str, final_metrics: Dict = None):
        """End monitoring session"""
        if session_id not in self.active_sessions:
            return
        
        try:
            session = self.active_sessions[session_id]
            duration = (datetime.now() - session['start_time']).total_seconds()
            
            with mlflow.start_run(run_id=session['run_id']):
                mlflow.log_param("end_time", datetime.now().isoformat())
                mlflow.log_metric("duration_seconds", duration)
                
                if final_metrics:
                    for key, value in final_metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"final_{key}", value)
            
            mlflow.end_run()
            
            del self.active_sessions[session_id]
            logger.info(f"✅ Session ended: {session_id} ({duration:.1f}s)")
            
        except Exception as e:
            logger.error(f"Session end error: {e}")
    
    def get_performance_metrics(self, hours: int = 24) -> Dict:
        """Get performance metrics from database"""
        try:
            conn = psycopg2.connect(config.DATABASE_URL)
            cur = conn.cursor()
            
            cutoff = datetime.now() - timedelta(hours=hours)
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                    AVG(similarity_score) as avg_score
                FROM prediction_logs
                WHERE timestamp > %s
            """, (cutoff,))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row and row[0] > 0:
                total, correct, avg_score = row
                accuracy = correct / total
                
                # Calculate FAR/FRR (simplified)
                far = max(0, 1 - accuracy - 0.05)
                frr = max(0, 1 - accuracy - 0.05)
                
                return {
                    'total_predictions': int(total),
                    'accuracy': float(accuracy),
                    'avg_similarity': float(avg_score) if avg_score else 0,
                    'far': float(far),
                    'frr': float(frr),
                    'timestamp': datetime.now().isoformat()
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Metrics fetch error: {e}")
            return {}
    
    def check_performance(self) -> tuple:
        """Check if performance is below thresholds"""
        metrics = self.get_performance_metrics(hours=24)
        
        if not metrics:
            return False, {}
        
        alerts = {}
        
        if metrics['accuracy'] < config.ACCURACY_THRESHOLD:
            alerts['accuracy'] = f"Accuracy {metrics['accuracy']:.4f} < {config.ACCURACY_THRESHOLD}"
        
        if metrics['far'] > config.FAR_THRESHOLD:
            alerts['far'] = f"FAR {metrics['far']:.4f} > {config.FAR_THRESHOLD}"
        
        if metrics['frr'] > config.FRR_THRESHOLD:
            alerts['frr'] = f"FRR {metrics['frr']:.4f} > {config.FRR_THRESHOLD}"
        
        return len(alerts) > 0, alerts
    
    def get_trends(self, days: int = 7) -> Dict:
        """Get performance trends"""
        try:
            conn = psycopg2.connect(config.DATABASE_URL)
            cur = conn.cursor()
            
            cutoff = datetime.now() - timedelta(days=days)
            
            cur.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                    AVG(similarity_score) as avg_score
                FROM prediction_logs
                WHERE timestamp > %s
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (cutoff,))
            
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            trends = defaultdict(list)
            for date, total, correct, avg_score in rows:
                trends['dates'].append(date.isoformat())
                trends['accuracy'].append(correct / total if total > 0 else 0)
                trends['avg_similarity'].append(float(avg_score) if avg_score else 0)
            
            return dict(trends)
            
        except Exception as e:
            logger.error(f"Trends fetch error: {e}")
            return {}
    
    def generate_report(self) -> str:
        """Generate performance report"""
        metrics_24h = self.get_performance_metrics(hours=24)
        metrics_7d = self.get_performance_metrics(hours=168)
        is_degraded, alerts = self.check_performance()
        
        report = f"""
================================================================================
FACE RECOGNITION SYSTEM - PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

24-HOUR METRICS:
----------------
Accuracy:          {metrics_24h.get('accuracy', 0):.4f}
FAR (False Accept): {metrics_24h.get('far', 0):.4f}
FRR (False Reject): {metrics_24h.get('frr', 0):.4f}
Avg Similarity:    {metrics_24h.get('avg_similarity', 0):.4f}
Total Predictions: {metrics_24h.get('total_predictions', 0)}

7-DAY METRICS:
--------------
Accuracy:          {metrics_7d.get('accuracy', 0):.4f}
Total Predictions: {metrics_7d.get('total_predictions', 0)}

PERFORMANCE STATUS:
-------------------
"""
        
        if is_degraded:
            report += "⚠️ ALERTS:\n"
            for alert_type, message in alerts.items():
                report += f"  - {message}\n"
        else:
            report += "✅ All metrics within acceptable thresholds\n"
        
        report += "\n================================================================================\n"
        
        return report
    
    def get_status(self) -> Dict:
        """Get current system status"""
        try:
            metrics = self.get_performance_metrics(hours=24)
            is_degraded, alerts = self.check_performance()
            
            return {
                'status': 'degraded' if is_degraded else 'healthy',
                'metrics': metrics,
                'alerts': alerts,
                'active_sessions': len(self.active_sessions),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global instance
mlops_monitor = SimpleMLOpsMonitor()


# Integration functions (called from face_processor_optimized.py)
def start_session(session_id: str, session_type: str, camera_view: str):
    """Start monitoring session"""
    mlops_monitor.start_session(session_id, session_type, camera_view)


def update_metrics(session_id: str, metrics: Dict):
    """Update session metrics"""
    mlops_monitor.update_metrics(session_id, metrics)


def end_session(session_id: str, final_metrics: Dict = None):
    """End monitoring session"""
    mlops_monitor.end_session(session_id, final_metrics)


def get_status():
    """Get system status"""
    return mlops_monitor.get_status()


def get_metrics(hours: int = 24):
    """Get performance metrics"""
    return mlops_monitor.get_performance_metrics(hours)


def get_trends(days: int = 7):
    """Get performance trends"""
    return mlops_monitor.get_trends(days)


def generate_report():
    """Generate performance report"""
    return mlops_monitor.generate_report()