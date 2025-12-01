"""
LedgerX - Cloud SQL Database Helper
====================================

Database operations for users and invoices
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "34.41.11.190"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "ledgerx"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "LedgerX2025SecurePass!")
}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Connection pool (reuse connections)
_connection = None

def get_db_connection():
    """Get database connection (reuses connection)"""
    global _connection
    
    if _connection is None or _connection.closed:
        _connection = psycopg2.connect(**DB_CONFIG)
        logger.info("[DB] ✅ Connected to Cloud SQL")
    
    return _connection

# ============================================================================
# USER OPERATIONS
# ============================================================================

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Get user from database by username"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT id, username, email, full_name, hashed_password, role, disabled
            FROM users
            WHERE username = %s
        """, (username,))
        
        user = cursor.fetchone()
        cursor.close()
        
        return dict(user) if user else None
        
    except Exception as e:
        logger.error(f"[DB] Error fetching user: {e}")
        return None

def create_user(username: str, email: str, full_name: str, password: str, role: str = "user") -> bool:
    """Create new user in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        hashed_password = pwd_context.hash(password)
        
        cursor.execute("""
            INSERT INTO users (username, email, full_name, hashed_password, role)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, email, full_name, hashed_password, role))
        
        conn.commit()
        cursor.close()
        
        logger.info(f"[DB] ✅ User created: {username}")
        return True
        
    except Exception as e:
        logger.error(f"[DB] Error creating user: {e}")
        conn.rollback()
        return False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

# ============================================================================
# INVOICE OPERATIONS
# ============================================================================

def save_invoice(user_id: int, invoice_data: Dict[str, Any]) -> Optional[int]:
    """Save invoice to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO invoices (
                user_id, invoice_number, vendor_name, total_amount, currency, invoice_date,
                quality_prediction, quality_score, risk_prediction, risk_score,
                file_name, file_type, file_size_kb, ocr_method, ocr_confidence,
                subtotal, tax_amount, discount_amount, raw_data
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) RETURNING id
        """, (
            user_id,
            invoice_data.get('invoice_number'),
            invoice_data.get('vendor_name'),
            invoice_data.get('total_amount'),
            invoice_data.get('currency'),
            invoice_data.get('invoice_date'),
            invoice_data.get('quality_prediction'),
            invoice_data.get('quality_score'),
            invoice_data.get('risk_prediction'),
            invoice_data.get('risk_score'),
            invoice_data.get('file_name'),
            invoice_data.get('file_type'),
            invoice_data.get('file_size_kb'),
            invoice_data.get('ocr_method'),
            invoice_data.get('ocr_confidence'),
            invoice_data.get('subtotal'),
            invoice_data.get('tax_amount'),
            invoice_data.get('discount_amount'),
            psycopg2.extras.Json(invoice_data)
        ))
        
        invoice_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        
        logger.info(f"[DB] ✅ Invoice saved: ID={invoice_id}")
        return invoice_id
        
    except Exception as e:
        logger.error(f"[DB] Error saving invoice: {e}")
        conn.rollback()
        return None

def get_user_invoices(user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
    """Get all invoices for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT * FROM invoices
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))
        
        invoices = cursor.fetchall()
        cursor.close()
        
        return [dict(inv) for inv in invoices]
        
    except Exception as e:
        logger.error(f"[DB] Error fetching invoices: {e}")
        return []

def delete_invoice(invoice_id: int, user_id: int) -> bool:
    """Delete invoice (only if belongs to user)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM invoices
            WHERE id = %s AND user_id = %s
        """, (invoice_id, user_id))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        cursor.close()
        
        if deleted:
            logger.info(f"[DB] ✅ Invoice deleted: ID={invoice_id}")
        
        return deleted
        
    except Exception as e:
        logger.error(f"[DB] Error deleting invoice: {e}")
        conn.rollback()
        return False

# ============================================================================
# BILLING TRACKING
# ============================================================================

def track_document_ai_usage():
    """Increment Document AI usage counter for today"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        today = datetime.utcnow().date()
        
        cursor.execute("""
            INSERT INTO billing_usage (service_name, usage_date, usage_count, cost_usd)
            VALUES ('document_ai', %s, 1, 0.01)
            ON CONFLICT (service_name, usage_date)
            DO UPDATE SET 
                usage_count = billing_usage.usage_count + 1,
                cost_usd = billing_usage.cost_usd + 0.01
        """, (today,))
        
        conn.commit()
        cursor.close()
        
    except Exception as e:
        logger.error(f"[DB] Error tracking usage: {e}")

def get_monthly_document_ai_usage() -> int:
    """Get Document AI usage for current month"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get first day of current month
        now = datetime.utcnow()
        first_day = now.replace(day=1).date()
        
        cursor.execute("""
            SELECT COALESCE(SUM(usage_count), 0)
            FROM billing_usage
            WHERE service_name = 'document_ai'
            AND usage_date >= %s
        """, (first_day,))
        
        count = cursor.fetchone()[0]
        cursor.close()
        
        return int(count)
        
    except Exception as e:
        logger.error(f"[DB] Error getting usage: {e}")
        return 0

def get_billing_stats() -> Dict[str, Any]:
    """Get billing statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get current month usage
        now = datetime.utcnow()
        first_day = now.replace(day=1).date()
        
        cursor.execute("""
            SELECT 
                service_name,
                SUM(usage_count) as total_count,
                SUM(cost_usd) as total_cost
            FROM billing_usage
            WHERE usage_date >= %s
            GROUP BY service_name
        """, (first_day,))
        
        stats = cursor.fetchall()
        cursor.close()
        
        return {row['service_name']: dict(row) for row in stats}
        
    except Exception as e:
        logger.error(f"[DB] Error getting billing stats: {e}")
        return {}