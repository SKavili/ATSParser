-- Create resume_metadata table
CREATE TABLE IF NOT EXISTS resume_metadata (
    id INT PRIMARY KEY AUTO_INCREMENT,
    candidatename VARCHAR(255),
    jobrole VARCHAR(255),
    experience VARCHAR(100),
    domain VARCHAR(255),
    mobile VARCHAR(50),
    email VARCHAR(255),
    education TEXT,
    filename VARCHAR(512) NOT NULL,
    skillset TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create index on filename for faster lookups
CREATE INDEX idx_filename ON resume_metadata(filename);

-- Create index on email for duplicate detection
CREATE INDEX idx_email ON resume_metadata(email);

