"""
Cloud SQL Database Integration - Production
Enables invoice history sync across all devices
"""

import os
import logging
from typing import Optional, List, Dict
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    import psycopg2.pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    logging.warning("psycopg2 not available - Cloud SQL disabled")

logger = logging.getLogger(__name__)

# Connection pool for production
connection_pool = None

def init_connection_pool():
    """Initialize PostgreSQL connection pool"""
    global connection_pool
    
    if not POSTGRES_AVAILABLE:
        return None
    
    try:
        # Check if running in Cloud Run (use Unix socket)
        if os.getenv('K_SERVICE'):
            # Cloud Run - use Unix socket
            instance_connection = os.getenv('INSTANCE_CONNECTION_NAME', 'ledgerx-mlops:us-central1:ledgerx-postgres')
            
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,
                host=f'/cloudsql/{instance_connection}',
                database='ledgerx_db',
                user='postgres',
                password=os.getenv('DB_PASSWORD', '')
            )
        else:
            # Local development - use TCP
            connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,
                host=os.getenv('DB_HOST', '34.41.11.190'),
                port=int(os.getenv('DB_PORT', '5432')),
                database=os.getenv('DB_NAME', 'ledgerx_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
        
        logger.info("✅ Cloud SQL connection pool initialized")
        return connection_pool
    except Exception as e:
        logger.error(f"❌ Failed to initialize Cloud SQL pool: {e}")
        return None

def get_db_connection():
    """Get connection from pool"""
    global connection_pool
    
    if not connection_pool:
        connection_pool = init_connection_pool()
    
    if connection_pool:
        return connection_pool.getconn()
    return None

def release_db_connection(conn):
    """Return connection to pool"""
    if connection_pool and conn:
        connection_pool.putconn(conn)

# ============================================================================
# USER OPERATIONS
# ============================================================================

def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user from database"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, hashed_password, email, full_name, role, disabled FROM users WHERE username = %s",
                (username,)
            )
            user = cur.fetchone()
            return dict(user) if user else None
    except Exception as e:
        logger.error(f"Error fetching user {username}: {e}")
        return None
    finally:
        if conn:
            release_db_connection(conn)

def create_user(username: str, hashed_password: str, email: str, full_name: str, role: str = 'user') -> Optional[int]:
    """Create new user"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO users (username, hashed_password, email, full_name, role) 
                   VALUES (%s, %s, %s, %s, %s) 
                   RETURNING id""",
                (username, hashed_password, email, full_name, role)
            )
            user_id = cur.fetchone()[0]
            conn.commit()
            logger.info(f"✅ Created user: {username} (ID: {user_id})")
            return user_id
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            release_db_connection(conn)

# ============================================================================
# INVOICE OPERATIONS
# ============================================================================

def save_invoice_to_db(user_id: int, invoice_data: Dict) -> Optional[int]:
    """
    Save invoice to Cloud SQL
    Enables cross-device history sync
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("Cloud SQL not available - invoice not saved to database")
            return None
        
        # Prepare metadata as JSONB
        metadata = {
            "file_name": invoice_data.get("file_name"),
            "file_type": invoice_data.get("file_type"),
            "file_size_kb": invoice_data.get("file_size_kb"),
            "ocr_method": invoice_data.get("ocr_method", "document_ai"),
            "ocr_confidence": invoice_data.get("ocr_confidence", 0.0),
            "currency": invoice_data.get("currency", "USD"),
            "invoice_date": invoice_data.get("invoice_date"),
            "subtotal": invoice_data.get("subtotal", 0.0),
            "tax_amount": invoice_data.get("tax_amount", 0.0),
            "discount_amount": invoice_data.get("discount_amount", 0.0),
            "quality_prediction": invoice_data.get("quality_prediction"),
            "risk_prediction": invoice_data.get("risk_prediction"),
            "processed_at": datetime.now().isoformat()
        }
        
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO invoices 
                   (user_id, invoice_number, vendor_name, total_amount, quality_score, failure_risk, metadata, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                   RETURNING id""",
                (
                    user_id,
                    invoice_data.get("invoice_number"),
                    invoice_data.get("vendor_name"),
                    float(invoice_data.get("total_amount", 0)),
                    float(invoice_data.get("quality_score", 0)),
                    float(invoice_data.get("risk_score", 0)),
                    psycopg2.extras.Json(metadata)
                )
            )
            invoice_id = cur.fetchone()[0]
            conn.commit()
            
            logger.info(f"✅ Saved invoice {invoice_data.get('invoice_number')} to Cloud SQL (ID: {invoice_id})")
            return invoice_id
            
    except Exception as e:
        logger.error(f"❌ Error saving invoice to Cloud SQL: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            release_db_connection(conn)

def get_user_invoices(user_id: int, limit: int = 1000) -> List[Dict]:
    """
    Get all invoices for a user from Cloud SQL
    Returns invoices across ALL devices
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.warning("Cloud SQL not available - returning empty list")
            return []
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT 
                    id,
                    invoice_number,
                    vendor_name,
                    total_amount,
                    quality_score,
                    failure_risk,
                    metadata,
                    created_at,
                    updated_at
                   FROM invoices 
                   WHERE user_id = %s 
                   ORDER BY created_at DESC 
                   LIMIT %s""",
                (user_id, limit)
            )
            invoices = cur.fetchall()
            
            # Convert to list of dicts
            result = []
            for inv in invoices:
                invoice_dict = dict(inv)
                # Merge metadata into main dict
                if invoice_dict.get('metadata'):
                    metadata = invoice_dict.pop('metadata')
                    invoice_dict.update(metadata)
                result.append(invoice_dict)
            
            logger.info(f"✅ Loaded {len(result)} invoices for user {user_id} from Cloud SQL")
            return result
            
    except Exception as e:
        logger.error(f"❌ Error loading invoices: {e}")
        return []
    finally:
        if conn:
            release_db_connection(conn)

def delete_invoice_from_db(invoice_id: int, user_id: int) -> bool:
    """Delete invoice (user can only delete their own)"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return False
        
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM invoices WHERE id = %s AND user_id = %s",
                (invoice_id, user_id)
            )
            deleted = cur.rowcount > 0
            conn.commit()
            
            if deleted:
                logger.info(f"✅ Deleted invoice {invoice_id} for user {user_id}")
            return deleted
            
    except Exception as e:
        logger.error(f"Error deleting invoice: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            release_db_connection(conn)

# ============================================================================
# ANALYTICS
# ============================================================================

def get_user_stats(user_id: int) -> Dict:
    """Get statistics for user's invoices"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return {}
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """SELECT 
                    COUNT(*) as total_invoices,
                    SUM(total_amount) as total_amount,
                    AVG(quality_score) as avg_quality,
                    AVG(failure_risk) as avg_risk,
                    COUNT(CASE WHEN metadata->>'quality_prediction' = 'good' THEN 1 END) as good_quality_count,
                    COUNT(CASE WHEN metadata->>'quality_prediction' = 'bad' THEN 1 END) as bad_quality_count
                   FROM invoices 
                   WHERE user_id = %s""",
                (user_id,)
            )
            stats = cur.fetchone()
            return dict(stats) if stats else {}
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {}
    finally:
        if conn:
            release_db_connection(conn)

# ============================================================================
# INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database with schema"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Cannot initialize database - no connection")
            return False
        
        # Read and execute schema
        schema_path = Path(__file__).parent.parent.parent / "schema.sql"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            with conn.cursor() as cur:
                cur.execute(schema_sql)
                conn.commit()
            
            logger.info("✅ Database schema initialized")
            return True
        else:
            logger.warning("schema.sql not found")
            return False
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            release_db_connection(conn)
