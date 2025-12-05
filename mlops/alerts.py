# mlops/alerts.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging
from typing import List
import json
import os

from mlops.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSystem:
    """Alert and notification system for MLOps monitoring"""
    
    def __init__(self):
        self.alert_history = []
        
    def send_email_alert(self, subject: str, message: str, recipients: List[str] = None):
        """Send email alert"""
        if not config.SMTP_USERNAME or not config.SMTP_PASSWORD:
            logger.warning("SMTP credentials not configured, skipping email alert")
            return False
        
        if recipients is None:
            recipients = [config.ALERT_EMAIL]
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config.SMTP_USERNAME
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[Face Recognition MLOps] {subject}"
            
            timestamped_message = f"""
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{message}

---
This is an automated alert from the Face Recognition Attendance System.
            """
            
            msg.attach(MIMEText(timestamped_message, 'plain'))
            
            with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
                server.starttls()
                server.login(config.SMTP_USERNAME, config.SMTP_PASSWORD)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def log_alert(self, subject: str, message: str, alert_type: str = 'info'):
        """Log alert to file"""
        alert_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'subject': subject,
            'message': message
        }
        
        self.alert_history.append(alert_entry)
        
        alert_file = config.LOGS_DIR / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        
        try:
            if alert_file.exists():
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            else:
                alerts = []
            
            alerts.append(alert_entry)
            
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def send_alert(self, subject: str, message: str, 
                   alert_type: str = 'info', send_email: bool = True):
        """Send alert through all configured channels"""
        
        self.log_alert(subject, message, alert_type)
        
        if alert_type == 'error':
            formatted_subject = f"🔴 ERROR: {subject}"
        elif alert_type == 'warning':
            formatted_subject = f"⚠️  WARNING: {subject}"
        elif alert_type == 'success':
            formatted_subject = f"✅ SUCCESS: {subject}"
        else:
            formatted_subject = f"ℹ️  INFO: {subject}"
        
        if send_email and config.SMTP_USERNAME and config.SMTP_PASSWORD:
            self.send_email_alert(formatted_subject, message)
        
        logger.info(f"Alert sent: {formatted_subject}")


class PerformanceAlertMonitor:
    """Monitor performance and trigger alerts"""
    
    def __init__(self):
        self.alert_system = AlertSystem()
        self.alert_cooldown = {}
    
    def check_and_alert(self, metrics: dict, alerts: dict):
        """Check metrics and send alerts if needed"""
        current_time = datetime.now()
        
        for metric_name, alert_message in alerts.items():
            last_alert_time = self.alert_cooldown.get(metric_name)
            
            if last_alert_time:
                time_diff = (current_time - last_alert_time).total_seconds()
                if time_diff < 3600:  # 1 hour cooldown
                    continue
            
            self.alert_system.send_alert(
                subject=f"Performance Degradation: {metric_name}",
                message=f"""
Performance degradation detected in the Face Recognition system.

Metric: {metric_name}
Issue: {alert_message}

Current Metrics:
- Accuracy: {metrics.get('accuracy', 0):.4f}
- FAR: {metrics.get('far', 0):.4f}
- FRR: {metrics.get('frr', 0):.4f}
- F1 Score: {metrics.get('f1_score', 0):.4f}

Action Required:
Review the system performance and consider model retraining.

Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
                """,
                alert_type='warning'
            )
            
            self.alert_cooldown[metric_name] = current_time
    
    def send_daily_report(self, metrics: dict, trends: dict):
        """Send daily performance report"""
        message = f"""
Daily Performance Report - Face Recognition Attendance System

Date: {datetime.now().strftime('%Y-%m-%d')}

CURRENT METRICS:
-----------------
Accuracy:    {metrics.get('accuracy', 0):.4f}
Precision:   {metrics.get('precision', 0):.4f}
Recall:      {metrics.get('recall', 0):.4f}
F1 Score:    {metrics.get('f1_score', 0):.4f}

SECURITY METRICS:
------------------
FAR (False Accept): {metrics.get('far', 0):.4f}
FRR (False Reject): {metrics.get('frr', 0):.4f}

USAGE STATISTICS:
------------------
Total Predictions: {metrics.get('total_predictions', 0)}
True Positives:    {metrics.get('true_positives', 0)}
True Negatives:    {metrics.get('true_negatives', 0)}
False Positives:   {metrics.get('false_positives', 0)}
False Negatives:   {metrics.get('false_negatives', 0)}

STATUS:
--------
"""
        
        if (metrics.get('far', 1) <= config.FAR_THRESHOLD and 
            metrics.get('frr', 1) <= config.FRR_THRESHOLD and 
            metrics.get('accuracy', 0) >= config.ACCURACY_THRESHOLD):
            message += "✅ System performing within acceptable thresholds\n"
        else:
            message += "⚠️  System performance requires attention\n"
        
        # Add trends if available
        if trends and 'dates' in trends:
            message += f"\nTRENDS (Last {len(trends['dates'])} days):\n"
            message += f"Accuracy Trend: {trends.get('accuracy', [0])[-1]:.4f}\n"
            message += f"Avg Similarity: {trends.get('avg_similarity', [0])[-1]:.4f}\n"
        
        self.alert_system.send_alert(
            subject="Daily Performance Report",
            message=message,
            alert_type='info',
            send_email=True
        )