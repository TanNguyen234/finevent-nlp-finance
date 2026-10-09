"""FastAPI application entrypoint for FinEvent system."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
    FinancialEvent,
    GuidanceChange,
    PredictEventRequest,
    PredictEventResponse,
)
from app.service import service

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinEvent API — Financial Event & Guidance Revision Engine",
    description="NLP-powered extraction of evidence-anchored corporate events and guidance revisions from SEC EDGAR filings.",
    version="1.0.0",
)

# Enable CORS for frontend applications (Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming API availability and model status."""
    return {
        "status": "healthy",
        "service": "FinEvent API",
        "model_loaded": service.model is not None,
        "total_events_indexed": len(service.events),
    }


@app.get("/companies", tags=["Companies"])
def list_companies() -> List[Dict[str, str]]:
    """List all tracked companies and their 10-digit CIK identifiers."""
    return service.list_companies()


@app.get("/companies/{cik}/events", response_model=List[FinancialEvent], tags=["Events"])
def get_company_events(
    cik: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
) -> List[Dict[str, Any]]:
    """Retrieve historical disclosure events for a specific CIK."""
    events = service.get_events(cik=cik, limit=limit)
    if not events:
        clean_cik = cik.strip().zfill(10)
        # Check if CIK exists
        known_ciks = [c["cik"] for c in service.list_companies()]
        if clean_cik not in known_ciks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Company with CIK {cik} not found in database.",
            )
    return events


@app.get("/events/{event_id}", response_model=FinancialEvent, tags=["Events"])
def get_event_by_id(event_id: str) -> Dict[str, Any]:
    """Retrieve a single event by its unique identifier."""
    event = service.get_event_by_id(event_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID '{event_id}' not found.",
        )
    return event


@app.get("/companies/{cik}/changes", response_model=List[GuidanceChange], tags=["Changes"])
def get_company_changes(
    cik: str,
    from_date: str = Query("2019-01-01", alias="from", description="Start date (YYYY-MM or YYYY-MM-DD)"),
    to_date: str = Query("2026-12-31", alias="to", description="End date (YYYY-MM or YYYY-MM-DD)"),
) -> List[Dict[str, Any]]:
    """Retrieve guidance revisions and metric diffs between reporting periods."""
    changes = service.get_changes(cik=cik, from_date=from_date, to_date=to_date)
    return changes


@app.post("/predict/event", response_model=PredictEventResponse, tags=["Inference"])
def predict_event(payload: PredictEventRequest) -> Dict[str, Any]:
    """Classify an SEC disclosure paragraph and extract financial arguments."""
    result = service.predict_event(payload.text)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
