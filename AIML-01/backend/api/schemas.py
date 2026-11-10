"""Request and response schemas for the fraud scoring API."""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """A transaction to score using the realtime-safe fraud model."""

    transaction_id: str = Field(..., min_length=1, examples=["txn_000001"])
    step: int = Field(..., ge=1, examples=[100])
    type: str = Field(..., examples=["TRANSFER"])
    amount: float = Field(..., ge=0, examples=[150000.0])
    nameOrig: str = Field(..., min_length=1, examples=["C1000001"])
    oldbalanceOrg: float = Field(..., ge=0, examples=[200000.0])
    newbalanceOrig: float = Field(..., ge=0, examples=[50000.0])
    nameDest: str = Field(..., min_length=1, examples=["C2000001"])
    oldbalanceDest: float = Field(..., ge=0, examples=[0.0])
    newbalanceDest: float = Field(..., ge=0, examples=[150000.0])


class PredictionResponse(BaseModel):
    """Fraud-risk prediction returned by the scoring API."""

    transaction_id: str
    fraud_probability: float
    risk_score: float
    is_alert: bool
    model_name: str
    threshold: float
    recommended_action: str


class CaseUpdate(BaseModel):
    """Payload to update an analyst case review status."""

    case_status: str
    analyst_feedback: str | None = None