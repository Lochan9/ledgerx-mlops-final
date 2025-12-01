"""
LedgerX - Cloud SQL Database Migration
========================================

Creates tables and initial data in Cloud SQL
"""

import os
import sys
import logging
from pathlib import Path
import psycopg2
from psycopg2 import sql
from passlib.context import CryptContext

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ledgerx_migration")

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),  # Use Cloud SQL proxy
    "port": int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME", "ledgerx"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "LedgerX2025SecurePass!")
}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ============================================================================
# SQL SCHEMAS
# ============================================================================

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    full_name VARCHAR(100),
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    disabled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INVOICES_TABLE = """
CREATE TABLE IF NOT EXISTS invoices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    -- Invoice data
    invoice_number VARCHAR(100),
    vendor_name VARCHAR(200),
    total_amount DECIMAL(12,2),
    currency VARCHAR(3),
    invoice_date DATE,
    
    -- ML predictions
    quality_prediction VARCHAR(20),
    quality_score DECIMAL(5,4),
    risk_prediction VARCHAR(20),
    risk_score DECIMAL(5,4),
    
    -- File metadata
    file_name VARCHAR(255),
    file_type VARCHAR(20),
    file_size_kb DECIMAL(10,2),
    ocr_method VARCHAR(50),
    ocr_confidence DECIMAL(5,4),
    
    -- Additional data
    subtotal DECIMAL(12,2),
    tax_amount DECIMAL(12,2),
    discount_amount DECIMAL(12,2),
    
    -- Full invoice data (JSON)
    raw_data JSONB,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_BILLING_USAGE_TABLE = """
CREATE TABLE IF NOT EXISTS billing_usage (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(50) NOT NULL,
    usage_date DATE NOT NULL,
    usage_count INTEGER DEFAULT 0,
    cost_usd DECIMAL(10,6) DEFAULT 0.0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(service_name, usage_date)
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_invoices_user_id ON invoices(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_invoices_created_at ON invoices(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_invoices_vendor ON invoices(vendor_name);",
    "CREATE INDEX IF NOT EXISTS idx_billing_usage_date ON billing_usage(usage_date);",
]

# ============================================================================
# SEED DATA
# ============================================================================

def get_seed_users():
    """Get initial users to insert"""
    return [
        {
            "username": "admin",
            "email": "admin@ledgerx.com",
            "full_name": "Admin User",
            "hashed_password": pwd_context.hash("admin123"),
            "role": "admin",
            "disabled": False
        },
        {
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "hashed_password": pwd_context.hash("password123"),
            "role": "user",
            "disabled": False
        }
    ]

# ============================================================================
# MIGRATION FUNCTIONS
# ============================================================================

def connect_to_db():
    """Connect to Cloud SQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info(f"[DB] ✅ Connected to Cloud SQL: {DB_CONFIG['database']}")
        return conn
    except Exception as e:
        logger.error(f"[DB] ❌ Connection failed: {e}")
        logger.error(f"[DB] Config: host={DB_CONFIG['host']}, database={DB_CONFIG['database']}")
        raise

def create_tables(conn):
    """Create all tables"""
    cursor = conn.cursor()
    
    try:
        logger.info("[MIGRATION] Creating tables...")
        
        # Create tables
        cursor.execute(CREATE_USERS_TABLE)
        logger.info("[MIGRATION] ✅ Created users table")
        
        cursor.execute(CREATE_INVOICES_TABLE)
        logger.info("[MIGRATION] ✅ Created invoices table")
        
        cursor.execute(CREATE_BILLING_USAGE_TABLE)
        logger.info("[MIGRATION] ✅ Created billing_usage table")
        
        # Create indexes
        for idx_sql in CREATE_INDEXES:
            cursor.execute(idx_sql)
        logger.info("[MIGRATION] ✅ Created indexes")
        
        conn.commit()
        logger.info("[MIGRATION] ✅ All tables created successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"[MIGRATION] ❌ Table creation failed: {e}")
        raise
    finally:
        cursor.close()

def seed_users(conn):
    """Insert initial users"""
    cursor = conn.cursor()
    
    try:
        logger.info("[SEED] Inserting initial users...")
        
        for user in get_seed_users():
            cursor.execute("""
                INSERT INTO users (username, email, full_name, hashed_password, role, disabled)
                VALUES (%(username)s, %(email)s, %(full_name)s, %(hashed_password)s, %(role)s, %(disabled)s)
                ON CONFLICT (username) DO NOTHING
            """, user)
        
        conn.commit()
        logger.info("[SEED] ✅ Users inserted")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"[SEED] ❌ User seed failed: {e}")
        raise
    finally:
        cursor.close()

def verify_migration(conn):
    """Verify migration was successful"""
    cursor = conn.cursor()
    
    try:
        # Check tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"[VERIFY] Tables created: {tables}")
        
        # Check user count
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        logger.info(f"[VERIFY] Users in database: {user_count}")
        
        logger.info("[VERIFY] ✅ Migration verified successfully")
        
    except Exception as e:
        logger.error(f"[VERIFY] ❌ Verification failed: {e}")
        raise
    finally:
        cursor.close()

# ============================================================================
# MAIN
# ============================================================================

def run_migration():
    """Run complete database migration"""
    logger.info("="*70)
    logger.info("LEDGERX CLOUD SQL MIGRATION - START")
    logger.info("="*70)
    
    conn = None
    
    try:
        # Connect
        conn = connect_to_db()
        
        # Create tables
        create_tables(conn)
        
        # Seed data
        seed_users(conn)
        
        # Verify
        verify_migration(conn)
        
        logger.info("="*70)
        logger.info("✅ MIGRATION COMPLETE!")
        logger.info("="*70)
        logger.info("Database is ready for use:")
        logger.info(f"  - Host: {DB_CONFIG['host']}")
        logger.info(f"  - Database: {DB_CONFIG['database']}")
        logger.info(f"  - Users: admin, john_doe")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error("="*70)
        logger.error("❌ MIGRATION FAILED!")
        logger.error("="*70)
        logger.exception(e)
        return False
        
    finally:
        if conn:
            conn.close()
            logger.info("[DB] Connection closed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LedgerX Database Migration")
    parser.add_argument("--reset", action="store_true", help="Drop all tables first (DESTRUCTIVE)")
    
    args = parser.parse_args()
    
    if args.reset:
        response = input("⚠️  This will DELETE all data. Are you sure? (yes/no): ")
        if response.lower() != "yes":
            logger.info("Migration cancelled")
            sys.exit(0)
        
        conn = connect_to_db()
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS invoices CASCADE")
        cursor.execute("DROP TABLE IF EXISTS users CASCADE")
        cursor.execute("DROP TABLE IF EXISTS billing_usage CASCADE")
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("[RESET] ✅ All tables dropped")
    
    success = run_migration()
    sys.exit(0 if success else 1)