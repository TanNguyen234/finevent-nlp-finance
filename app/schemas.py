"""Pydantic schemas for FinEvent API and data validation."""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

EventTypeLiteral = Literal[
    "GUIDANCE_REVISION",
    "EARNINGS_RESULT",
    "MANAGEMENT_CHANGE",
    "OTHER",
]

RevisionDirectionLiteral = Literal[
    "UPWARD_REVISION",
    "DOWNWARD_REVISION",
    "REAFFIRMED",
    "UNKNOWN",
]


class FinancialEvent(BaseModel):
    """Pydantic representation of a classified financial event."""

    id: str = Field(..., description="Unique event identifier")
    cik: str = Field(..., description="10-digit SEC Central Index Key")
    company_name: str = Field(..., description="Full registrant company name")
    filing_date: str = Field(..., description="SEC filing date (YYYY-MM-DD)")
    fiscal_period: str = Field(..., description="Fiscal reporting period (e.g., Q1 2024)")
    accession_number: str = Field(..., description="SEC EDGAR accession number")
    source_url: str = Field(..., description="Verifiable SEC EDGAR URL")
    event_type: EventTypeLiteral = Field(..., description="One of the 4 standard event types")
    text: str = Field(..., description="Raw paragraph snippet text")
    metric: Optional[str] = Field(None, description="Primary financial metric, if any")
    old_low: Optional[float] = Field(None, description="Prior guidance low boundary")
    old_high: Optional[float] = Field(None, description="Prior guidance high boundary")
    new_low: Optional[float] = Field(None, description="Revised/reported low boundary")
    new_high: Optional[float] = Field(None, description="Revised/reported high boundary")
    unit: Optional[str] = Field(None, description="Financial unit (USD_B, USD_M, USD, %, persons)")
    evidence: str = Field(..., description="Verbatim evidence substring")
    char_start: int = Field(..., ge=0, description="Start character offset")
    char_end: int = Field(..., ge=0, description="End character offset")
    status: str = Field("reviewed", description="Annotation or audit status")

    @field_validator("cik")
    @classmethod
    def validate_cik(cls, v: str) -> str:
        clean = v.strip()
        if not clean.isdigit() or len(clean) != 10:
            raise ValueError(f"CIK must be a 10-digit numeric string, got: {v}")
        return clean


class GuidanceChange(BaseModel):
    """Delta comparison between guidance periods."""

    cik: str
    company_name: str
    metric: str
    fiscal_period: str
    unit: str
    old_low: Optional[float] = None
    old_high: Optional[float] = None
    new_low: Optional[float] = None
    new_high: Optional[float] = None
    delta_low: Optional[float] = None
    delta_high: Optional[float] = None
    delta_mid: Optional[float] = None
    direction: RevisionDirectionLiteral
    evidence: str
    source_url: str
    filing_date_prior: Optional[str] = None
    filing_date_current: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictEventRequest(BaseModel):
    """Input payload for live event prediction."""

    text: str = Field(..., min_length=5, description="Paragraph text from SEC filing")
    cik: Optional[str] = Field(None, description="Optional 10-digit CIK")
    company_name: Optional[str] = Field(None, description="Optional company name")
    filing_date: Optional[str] = Field(None, description="Filing date YYYY-MM-DD")
    fiscal_period: Optional[str] = Field(None, description="Fiscal period")


class PredictEventResponse(BaseModel):
    """Prediction output containing classification and extracted arguments."""

    event_type: EventTypeLiteral
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    arguments: dict
    evidence_span: str
    model_version: str
