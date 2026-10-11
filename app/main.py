"""FastAPI application entrypoint for FinEvent system."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import (
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint confirming API availability, configured backend, and cloud weights status."""
    return service.get_service_status()


@app.get("/companies", tags=["Companies"])
def list_companies() -> List[Dict[str, str]]:
    """List all tracked companies and their 10-digit CIK identifiers."""
    return service.list_companies()


@app.get("/companies/{cik}/events", tags=["Events"])
def get_company_events(
    cik: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum events to return"),
) -> List[Dict[str, Any]]:
    """Retrieve historical disclosure events for a specific CIK."""
    clean_cik = cik.strip().zfill(10)
    events = service.get_events(cik=clean_cik, limit=limit)
    if not events:
        known_ciks = [c["cik"] for c in service.list_companies()]
        if clean_cik not in known_ciks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Company with CIK {cik} not found in database.",
            )
    return events


@app.get("/events/{event_id}", tags=["Events"])
def get_event_by_id(event_id: str) -> Dict[str, Any]:
    """Retrieve a single event by its unique identifier."""
    event = service.get_event_by_id(event_id)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID '{event_id}' not found.",
        )
    return event


@app.get("/companies/{cik}/changes", tags=["Changes"])
def get_company_changes(
    cik: str,
    from_date: str = Query("2019-01-01", alias="from", description="Start date (YYYY-MM or YYYY-MM-DD)"),
    to_date: str = Query("2026-12-31", alias="to", description="End date (YYYY-MM or YYYY-MM-DD)"),
) -> List[Dict[str, Any]]:
    """Retrieve guidance revisions between reporting periods with date bounds validation."""
    if from_date > to_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date range: 'from' ({from_date}) must be prior to or equal to 'to' ({to_date}).",
        )
    changes = service.get_changes(cik=cik, from_date=from_date, to_date=to_date)
    return changes


@app.post("/predict/event", tags=["Inference"])
def predict_event(payload: PredictEventRequest) -> Dict[str, Any]:
    """Classify an SEC disclosure paragraph and extract financial arguments."""
    result = service.predict_event(payload.text)
    return result
