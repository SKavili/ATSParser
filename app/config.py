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

    # LLM provider: "OpenAI" or "OLLAMA" (used for metadata extraction and AI search; embeddings unchanged)
    llm_model: str = Field("OLLAMA", alias="LLM_MODEL")
    # OpenAI (when LLM_MODEL=OpenAI)
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", alias="OPENAI_MODEL")
    
    # Memory Optimization Settings
    embedding_batch_size: int = Field(5, alias="EMBEDDING_BATCH_SIZE")
    max_file_size_mb: int = Field(10, alias="MAX_FILE_SIZE_MB")
    max_resume_text_length: int = Field(50000, alias="MAX_RESUME_TEXT_LENGTH")
    job_cache_max_size: int = Field(100, alias="JOB_CACHE_MAX_SIZE")
    enable_memory_cleanup: bool = Field(True, alias="ENABLE_MEMORY_CLEANUP")
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(None, alias="SENTRY_DSN")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    
    # AI Search (optional overrides; defaults keep current behavior)
    ai_search_query_max_length: int = Field(2000, alias="AI_SEARCH_QUERY_MAX_LENGTH")
    ai_search_top_k_max: int = Field(200, alias="AI_SEARCH_TOP_K_MAX")
    ai_search_name_result_cap: int = Field(100, alias="AI_SEARCH_NAME_RESULT_CAP")
    ai_search_hybrid_semantic_weight: float = Field(0.7, alias="AI_SEARCH_HYBRID_SEMANTIC_WEIGHT")  # 0.7 semantic + 0.3 keyword
    ai_search_skill_weight: float = Field(40.0, alias="AI_SEARCH_SKILL_WEIGHT")
    ai_search_group_skill_weight: float = Field(30.0, alias="AI_SEARCH_GROUP_SKILL_WEIGHT")
    ai_search_designation_match_weight: float = Field(50.0, alias="AI_SEARCH_DESIGNATION_MATCH_WEIGHT")
    ai_search_designation_mismatch_penalty: float = Field(40.0, alias="AI_SEARCH_DESIGNATION_MISMATCH_PENALTY")
    ai_search_experience_exact_weight: float = Field(18.0, alias="AI_SEARCH_EXPERIENCE_EXACT_WEIGHT")
    ai_search_domain_match_weight: float = Field(12.0, alias="AI_SEARCH_DOMAIN_MATCH_WEIGHT")

    # ai-search-2: broad namespace fanout control
    # Max results Pinecone returns per namespace during ai-search-2 broad search.
    ai_search_2_top_k_per_namespace_cap: int = Field(
        100,
        alias="AI_SEARCH_2_TOP_K_PER_NAMESPACE_CAP",
        description="Max Pinecone matches fetched per namespace for ai-search-2 broad search",
    )
    
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

