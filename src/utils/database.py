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

# Connection pool
_connection = None

def get_db_connection():
    """Get database connection using DATABASE_URL (Cloud Run compatible)"""
    global _connection

    if _connection is None or _connection.closed:
        # Use DATABASE_URL from Secret Manager (Cloud Run)
        database_url = os.getenv("DATABASE_URL")
        
        if database_url:
            # Parse the DATABASE_URL for Cloud Run Unix socket
            logger.info("[DB] Using DATABASE_URL from Secret Manager")
            
            # For Cloud Run, DATABASE_URL format is:
            # postgresql://user:pass@/cloudsql/project:region:instance/dbname
            if "/cloudsql/" in database_url:
                # Extract components for Unix socket connection
                # Format: postgresql://postgres:password@/cloudsql/connection-name/database
                parts = database_url.split("@")
                user_pass = parts[0].replace("postgresql://", "")
                socket_path = parts[1]
                
                user, password = user_pass.split(":")
                # socket_path is like: /cloudsql/project:region:instance/dbname
                db_parts = socket_path.split("/")
                connection_name = db_parts[2]  # project:region:instance
                database = db_parts[3] if len(db_parts) > 3 else "ledgerx_db"
                
                _connection = psycopg2.connect(
                    host=f"/cloudsql/{connection_name}",
                    database=database,
                    user=user,
                    password=password
                )
                logger.info(f"[DB] ? Connected via Unix socket: /cloudsql/{connection_name}")
            else:
                # Regular TCP connection (local development)
                _connection = psycopg2.connect(database_url)
                logger.info("[DB] ? Connected via TCP (development)")
        else:
            # Fallback to old environment variables (legacy)
            logger.warning("[DB] ??  DATABASE_URL not found, using legacy config")
            _connection = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME", "ledgerx_db"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "")
            )
            logger.info("[DB] ? Connected via legacy config")

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
    """Save invoice to database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO invoices (
                user_id, invoice_number, vendor_name, total_amount,
                quality_score, failure_risk
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            user_id,
            invoice_data.get('invoice_number'),
            invoice_data.get('vendor_name'),
            invoice_data.get('total_amount'),
            invoice_data.get('quality_score'),
            invoice_data.get('failure_risk')
        ))

        invoice_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()

        logger.info(f"[DB] ? Invoice saved: ID={invoice_id}")
        return invoice_id

    except Exception as e:
        logger.error(f"[DB] ? Error saving invoice: {e}")
        if conn:
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