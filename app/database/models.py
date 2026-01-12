"""SQLAlchemy database models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.sql import func

from app.database.connection import Base


class ResumeMetadata(Base):
    """Database model for resume metadata."""
    __tablename__ = "resume_metadata"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    mastercategory = Column(String(255), nullable=True)  # IT or NON_IT
    category = Column(String(255), nullable=True)
    candidatename = Column(String(255), nullable=True)
    jobrole = Column(String(255), nullable=True)
    designation = Column(String(255), nullable=True)  # Current or most recent job title
    experience = Column(String(100), nullable=True)
    domain = Column(String(255), nullable=True)
    mobile = Column(String(50), nullable=True)
    email = Column(String(255), nullable=True)
    education = Column(Text, nullable=True)
    filename = Column(String(512), nullable=False)
    skillset = Column(Text, nullable=True)
    status = Column(String(50), nullable=True, server_default="pending")  # Processing status
    resume_text = Column(Text, nullable=True)  # Full extracted resume text
    pinecone_status = Column(Integer, nullable=True, server_default="0")  # Pinecone indexing status: 0 = not indexed, 1 = indexed
    created_at = Column(TIMESTAMP, nullable=True, server_default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP,
        nullable=True,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    def __repr__(self) -> str:
        return f"<ResumeMetadata(id={self.id}, candidatename={self.candidatename}, designation={self.designation}, filename={self.filename}, status={self.status})>"


class Prompt(Base):
    """Database model for skills extraction prompts."""
    __tablename__ = "prompts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    mastercategory = Column(String(255), nullable=True)  # IT or NON_IT
    category = Column(String(255), nullable=True)  # Specific category name
    prompt = Column(Text, nullable=False)  # The prompt text for skills extraction
    created_at = Column(TIMESTAMP, nullable=True, server_default=func.current_timestamp())
    updated_at = Column(
        TIMESTAMP,
        nullable=True,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp()
    )
    
    def __repr__(self) -> str:
        return f"<Prompt(id={self.id}, mastercategory={self.mastercategory}, category={self.category})>"

