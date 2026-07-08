import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5435/fraud_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    amount = Column(Float)
    merchant = Column(String)
    category = Column(String)
    gender = Column(String)
    city = Column(String)
    state = Column(String)
    lat = Column(Float)
    long = Column(Float)
    city_pop = Column(Integer)
    job = Column(String)
    dob = Column(String)
    merch_lat = Column(Float)
    merch_long = Column(Float)
    
    # Model tracking fields
    fraud_probability = Column(Float)
    is_fraud = Column(Integer)
    explanation = Column(JSON)  # Stores serialized SHAP reasons
    timestamp = Column(DateTime, default=datetime.utcnow)