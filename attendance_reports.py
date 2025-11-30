# attendance_reports.py - FIXED: Ensure proper CSV extension
"""
Generates attendance reports in CSV format
Fixed: Ensures .csv extension (not csv_)
"""
import os
import csv
import psycopg2
from datetime import datetime

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


def generate_attendance_report_csv(report_type="detailed"):
    """
    Generate attendance report using pure SQL - MUCH faster.
    All aggregation done in PostgreSQL.
    
    Args:
        report_type: Either "monthly" (summary) or "detailed" (all records)
    
    Returns:
        dict: Result with success status and file path
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report_type == "monthly":
            # ============================================
            # MONTHLY REPORT - Pure SQL aggregation
            # ============================================
            
            # Single SQL query that does ALL the work
            cur.execute("""
                SELECT 
                    person_id,
                    TO_CHAR(attendance_date, 'YYYY-MM') as month,
                    COUNT(*) as attendance_count
                FROM attendance
                GROUP BY person_id, TO_CHAR(attendance_date, 'YYYY-MM')
                ORDER BY person_id, month
            """)
            
            records = cur.fetchall()
            
            if not records:
                cur.close()
                conn.close()
                return {"success": False, "error": "No attendance records found"}
            
            # FIXED: Proper CSV filename (not csv_)
            filename = f"monthly_attendance_report_{timestamp}.csv"
            filepath = os.path.join("reports", filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['person_id', 'month', 'attendance_count'])
                writer.writerows(records)
            
            cur.close()
            conn.close()
            
            print(f"✅ Monthly report generated: {filepath}")
            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "records_count": len(records)
            }
            
        else:  # detailed
            # ============================================
            # DETAILED REPORT - SQL with window function
            # ============================================
            
            # Single SQL query with window function for monthly counts
            cur.execute("""
                SELECT 
                    a.id as attendance_id,
                    a.person_id,
                    TO_CHAR(a.attendance_date, 'YYYY-MM-DD') as attendance_date,
                    TO_CHAR(a.attendance_date, 'YYYY-MM') as month,
                    COUNT(*) OVER (
                        PARTITION BY a.person_id, TO_CHAR(a.attendance_date, 'YYYY-MM')
                    ) as monthly_attendance_count
                FROM attendance a
                ORDER BY a.person_id, a.attendance_date
            """)
            
            records = cur.fetchall()
            
            if not records:
                cur.close()
                conn.close()
                return {"success": False, "error": "No attendance records found"}
            
            # FIXED: Proper CSV filename (not csv_)
            filename = f"detailed_attendance_report_{timestamp}.csv"
            filepath = os.path.join("reports", filename)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['attendance_id', 'person_id', 'attendance_date', 'month', 'monthly_attendance_count'])
                writer.writerows(records)
            
            cur.close()
            conn.close()
            
            print(f"✅ Detailed report generated: {filepath}")
            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "records_count": len(records)
            }
            
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }