-- Users table with authentication
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',
    disabled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);

-- Insert default users (password: admin123, password123, viewer123)
-- These are bcrypt hashes of the passwords
INSERT INTO users (username, hashed_password, email, full_name, role, disabled) VALUES
('admin', '\\\/LewY5GyYzpLaEMUFC', 'admin@ledgerx.ai', 'System Administrator', 'admin', false),
('john_doe', '\\\.dU.CqAGVi.p8F5p5Lw5L8.eFQKqWNm', 'john@ledgerx.ai', 'John Doe', 'user', false),
('jane_viewer', '\\\', 'jane@ledgerx.ai', 'Jane Viewer', 'readonly', false)
ON CONFLICT (username) DO NOTHING;

-- Invoices table
CREATE TABLE IF NOT EXISTS invoices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    invoice_number VARCHAR(100) NOT NULL,
    vendor_name VARCHAR(200),
    total_amount DECIMAL(10, 2),
    quality_score DECIMAL(5, 4),
    failure_risk DECIMAL(5, 4),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_invoices_user_id ON invoices(user_id);
CREATE INDEX idx_invoices_created_at ON invoices(created_at);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    endpoint VARCHAR(100),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms INTEGER,
    ip_address VARCHAR(45),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX idx_api_usage_created_at ON api_usage(created_at);
