import json
import redis
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List


from app.database import engine, Base, get_db, REDIS_URL, Transaction
from app.ml import process_and_predict


Base.metadata.create_all(bind=engine)

app = FastAPI(title="Real-Time Fraud Detection Engine", version="1.0.0")
cache = redis.from_url(REDIS_URL, decode_responses=True)

class TransactionIn(BaseModel):
    user_id: int = Field(..., example=101)
    amount: float = Field(..., example=4200.0)
    merchant: str = Field(..., example="Amazon")
    category: str = Field(..., example="online_shopping")
    gender: str = Field(..., example="F")
    city: str = Field(..., example="Mumbai")
    state: str = Field(..., example="MH")
    lat: float = Field(..., example=18.96)
    long: float = Field(..., example=72.82)
    city_pop: int = Field(..., example=18400000)
    job: str = Field(..., example="Software Developer")
    dob: str = Field(..., example="1992-04-15")
    merch_lat: float = Field(..., example=19.12)
    merch_long: float = Field(..., example=72.90)

class TransactionOut(TransactionIn):
    id: int
    fraud_probability: float
    is_fraud: int
    explanation: List[str]
    timestamp: datetime

    class Config:
        from_attributes = True





@app.post("/predict", response_model=TransactionOut, status_code=status.HTTP_201_CREATED)
def predict_transaction(payload: TransactionIn, db: Session = Depends(get_db)):
    #  Caching Check 
    cache_key = f"txn:{payload.user_id}:{payload.amount}:{payload.merchant}"
    cached_data = cache.get(cache_key)
    if cached_data:
        return json.loads(cached_data)

    #  Extract, predict, and explain
    probability, explanation = process_and_predict(payload.model_dump(),cache)
    is_fraud = 1 if probability > 0.75 else 0

    #  Persist to DB
    db_txn = Transaction(
        **payload.model_dump(),
        fraud_probability=probability,
        is_fraud=is_fraud,
        explanation=explanation
    )
    db.add(db_txn)
    db.commit()
    db.refresh(db_txn)

    # Cache transaction result (Expire in 60s)
    serialized_response = TransactionOut.model_validate(db_txn).model_dump_json()
    cache.setex(cache_key, 60, serialized_response)

    return db_txn

@app.get("/stats")
def fetch_system_stats(db: Session = Depends(get_db)):
    # key telemetry
    total = db.query(Transaction).count()
    flagged = db.query(Transaction).filter(Transaction.is_fraud == 1).count()
    
    return {
        "status": "online",
        "total_evaluated_transactions": total,
        "total_flagged_frauds": flagged,
        "current_fraud_ratio": (flagged / total * 100) if total > 0 else 0.0
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}