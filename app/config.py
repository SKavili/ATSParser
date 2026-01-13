"""Configuration management using Pydantic Settings."""
import os
from typing import Optional
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # MySQL Configuration
    mysql_host: str = Field(..., alias="MYSQL_HOST")
    mysql_user: str = Field(..., alias="MYSQL_USER")
    mysql_password: str = Field(..., alias="MYSQL_PASSWORD")
    mysql_database: str = Field(..., alias="MYSQL_DATABASE")
    mysql_port: int = Field(3306, alias="MYSQL_PORT")
    
    # Chunking Configuration
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")
    top_k_results: int = Field(5, alias="TOP_K_RESULTS")
    similarity_threshold: float = Field(0.5, alias="SIMILARITY_THRESHOLD")
    embedding_dimension: int = Field(768, alias="EMBEDDING_DIMENSION")
    
    # Pinecone Configuration
    pinecone_api_key: Optional[str] = Field(None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("ats", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field("aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field("us-east-1", alias="PINECONE_REGION")
    
    # OLLAMA Configuration
    ollama_host: str = Field("http://localhost:11434", alias="OLLAMA_HOST")
    ollama_api_key: Optional[str] = Field(None, alias="OLLAMA_API_KEY")
    
    # Memory Optimization Settings
    embedding_batch_size: int = Field(5, alias="EMBEDDING_BATCH_SIZE")
    max_file_size_mb: int = Field(10, alias="MAX_FILE_SIZE_MB")
    max_resume_text_length: int = Field(50000, alias="MAX_RESUME_TEXT_LENGTH")
    job_cache_max_size: int = Field(100, alias="JOB_CACHE_MAX_SIZE")
    enable_memory_cleanup: bool = Field(True, alias="ENABLE_MEMORY_CLEANUP")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(None, alias="SENTRY_DSN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    
    # SQL Logging (for debugging)
    sql_echo: bool = Field(False, alias="SQL_ECHO")  # Enable SQL query logging
    sql_log_level: str = Field("INFO", alias="SQL_LOG_LEVEL")  # SQL log level: DEBUG, INFO, WARNING
    
    @field_validator("mysql_host", "mysql_user", "mysql_password", "mysql_database")
    @classmethod
    def validate_mysql_fields(cls, v: str) -> str:
        """Validate critical MySQL fields are not empty."""
        if not v or not v.strip():
            raise ValueError("MySQL configuration fields cannot be empty")
        return v.strip()
    
    @property
    def mysql_url(self) -> str:
        """Generate MySQL connection URL."""
        
        # URL encode username and password to handle special characters
        encoded_user = quote_plus(self.mysql_user)
        encoded_password = quote_plus(self.mysql_password) if self.mysql_password else ""
        
        # Build connection string
        if encoded_password:
            auth = f"{encoded_user}:{encoded_password}"
        else:
            auth = encoded_user
            
        return (
            f"mysql+aiomysql://{auth}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            "?charset=utf8mb4"
        )
    @property
    def use_pinecone(self) -> bool:
        """Check if Pinecone should be used."""
        return bool(self.pinecone_api_key and self.pinecone_api_key.strip())
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()

