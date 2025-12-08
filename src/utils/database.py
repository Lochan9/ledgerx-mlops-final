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
from psycopg2.extras import RealDictCursor, Json
from passlib.context import CryptContext
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# No connection pooling - create fresh each time to avoid transaction errors
# This is the CRITICAL FIX for "transaction aborted" errors

def get_db_connection():
    """
    Get FRESH database connection (no pooling/reuse)
    Creates new connection each time to prevent "transaction aborted" state
    """
    # Always create new connection (don't reuse)
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        # Parse the DATABASE_URL for Cloud Run Unix socket
        logger.info("[DB] Using DATABASE_URL from Secret Manager")
        
        if "/cloudsql/" in database_url:
            parts = database_url.split("@")
            user_pass = parts[0].replace("postgresql://", "")
            socket_path = parts[1]
            
            user, password = user_pass.split(":")
            db_parts = socket_path.split("/")
            connection_name = db_parts[2]
            database = db_parts[3] if len(db_parts) > 3 else "ledgerx"
            
            conn = psycopg2.connect(
                host=f"/cloudsql/{connection_name}",
                database=database,
                user=user,
                password=password
            )
            logger.info(f"[DB] ✅ Connected via Unix socket: /cloudsql/{connection_name}")
            return conn
        else:
            conn = psycopg2.connect(database_url)
            logger.info("[DB] ✅ Connected via TCP")
            return conn
    else:
        # Fallback to environment variables
        logger.info("[DB] Using legacy config from env vars")
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "ledgerx"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
        logger.info("[DB] ✅ Connected via legacy config")
        return conn

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
        logger.error(f"[DB] ? Error fetching user {username}: {e}")
        # CRITICAL: Rollback failed transaction
        try:
            conn = get_db_connection()
            conn.rollback()
            logger.info("[DB] Transaction rolled back")
        except:
            pass
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

        logger.info(f"[DB] ? User created: {username}")
        return True

    except Exception as e:
        logger.error(f"[DB] ? Error creating user: {e}")
        if conn:
            conn.rollback()
        return False

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

# ============================================================================
# INVOICE OPERATIONS (keeping your existing code)
# ============================================================================

def save_invoice(user_id: int, invoice_data: Dict[str, Any]) -> Optional[int]:
    """Save invoice to database with all fields"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO invoices (
                user_id, invoice_number, vendor_name, total_amount, currency, invoice_date,
                quality_prediction, quality_score, risk_prediction, risk_score,
                file_name, file_type, file_size_kb, ocr_method, ocr_confidence,
                subtotal, tax_amount, discount_amount
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            user_id,
            invoice_data.get('invoice_number', 'N/A'),
            invoice_data.get('vendor_name', 'Unknown'),
            invoice_data.get('total_amount', 0),
            invoice_data.get('currency', 'USD'),
            invoice_data.get('invoice_date', datetime.now().date()),
            invoice_data.get('quality_prediction', 'unknown'),
            invoice_data.get('quality_score', 0),
            invoice_data.get('risk_prediction', 'unknown'),
            invoice_data.get('risk_score', 0),  # FIXED: was failure_risk
            invoice_data.get('file_name', 'unknown'),
            invoice_data.get('file_type', 'IMAGE'),
            invoice_data.get('file_size_kb', 0),
            invoice_data.get('ocr_method', 'document_ai'),
            invoice_data.get('ocr_confidence', 0),
            invoice_data.get('subtotal', 0),
            invoice_data.get('tax_amount', 0),
            invoice_data.get('discount_amount', 0)
        ))

        invoice_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(f"[DB] ✅ Invoice saved: ID={invoice_id}")
        return invoice_id

    except Exception as e:
        logger.error(f"[DB] ❌ Error saving invoice: {e}")
        try:
            if conn:
                conn.rollback()
        except:
            pass
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
        logger.error(f"[DB] ? Error fetching invoices: {e}")
        return []

def delete_invoice(invoice_id: int, user_id: int) -> bool:
    """Delete invoice (with ownership check)"""
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
            logger.info(f"[DB] ? Invoice deleted: ID={invoice_id}")
        return deleted

    except Exception as e:
        logger.error(f"[DB] ? Error deleting invoice: {e}")
        if conn:
            conn.rollback()
        return False

# ============================================================================
# API USAGE TRACKING
# ============================================================================

def track_document_ai_usage(user_id: int, pages: int = 1):
    """Track Document AI usage for cost monitoring"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO api_usage (user_id, endpoint, method, status_code)
            VALUES (%s, 'document_ai', 'OCR', 200)
        """, (user_id,))

        conn.commit()
        cursor.close()
        logger.info(f"[DB] ? Tracked Document AI usage for user {user_id}")

    except Exception as e:
        logger.error(f"[DB] ? Error tracking usage: {e}")

def get_monthly_document_ai_usage(user_id: Optional[int] = None) -> int:
    """Get monthly Document AI page count"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        if user_id:
            cursor.execute("""
                SELECT COUNT(*) FROM api_usage
                WHERE user_id = %s
                AND endpoint = 'document_ai'
                AND created_at >= date_trunc('month', CURRENT_DATE)
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM api_usage
                WHERE endpoint = 'document_ai'
                AND created_at >= date_trunc('month', CURRENT_DATE)
            """)

        count = cursor.fetchone()[0]
        cursor.close()
        return count

    except Exception as e:
        logger.error(f"[DB] ? Error getting usage: {e}")
        return 0