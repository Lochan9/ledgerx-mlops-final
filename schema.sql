CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    full_name VARCHAR(100),
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS invoices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    invoice_number VARCHAR(100),
    vendor_name VARCHAR(200),
    total_amount DECIMAL(10,2),
    currency VARCHAR(3),
    invoice_date DATE,
    quality_prediction VARCHAR(20),
    quality_score DECIMAL(5,4),
    risk_prediction VARCHAR(20),
    risk_score DECIMAL(5,4),
    file_name VARCHAR(255),
    file_type VARCHAR(20),
    ocr_method VARCHAR(50),
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
