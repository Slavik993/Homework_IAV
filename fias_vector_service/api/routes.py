"""FastAPI routes for FIAS Vector Service."""
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from models.schemas import (
    AddressInput,
    AddressCreateResponse,
    AddressMatch,
    AddressMatchResult,
    AddressRecordOut,
)
from services.fias_vector import FIASVectorService
from api.deps import get_fias_service

router = APIRouter(prefix="/addresses", tags=["addresses"])


class AddressNormalizeRequest(BaseModel):
    """Request to normalize an address."""
    raw_address: str = Field(..., description="Raw address string to normalize")


class AddressNormalizeResponse(BaseModel):
    """Response with normalized address."""
    success: bool
    normalized: dict
    message: str


class AddressSearchRequest(BaseModel):
    """Request to search addresses."""
    query: str = Field(..., description="Search query")
    region: Optional[str] = Field(None, description="Filter by region")
    city: Optional[str] = Field(None, description="Filter by city")
    top_k: int = Field(10, ge=1, le=100, description="Number of results")


class AddressSearchResponse(BaseModel):
    """Response with search results."""
    query: str
    results: List[AddressMatch]
    total: int


@router.post("/add", response_model=AddressCreateResponse, status_code=status.HTTP_200_OK)
async def add_address(
    address_input: AddressInput,
    auto_create: bool = Query(True, description="Auto-create if no exact match"),
    service: FIASVectorService = Depends(get_fias_service),
) -> AddressCreateResponse:
    """
    Add a new address with smart deduplication.
    
    This endpoint:
    1. Normalizes the address using FIAS
    2. Searches for similar addresses (semantic search)
    3. Returns existing match if similarity >= 0.82
    4. Creates new record if no match found
    
    ## Example
    ```json
    {
        "raw_address": "г Москва, ул Ленина, д 10, кв 5",
        "region_hint": "Москва"
    }
    ```
    """
    try:
        result = await service.smart_upsert(address_input, auto_create=auto_create)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process address: {str(e)}"
        )


@router.post("/normalize", response_model=AddressNormalizeResponse)
async def normalize_address(
    request: AddressNormalizeRequest,
    service: FIASVectorService = Depends(get_fias_service),
) -> AddressNormalizeResponse:
    """
    Normalize an address without storing it.
    
    Returns normalized components and FIAS IDs where found.
    """
    try:
        normalized = await service.normalize_address(request.raw_address)
        return AddressNormalizeResponse(
            success=True,
            normalized=normalized.model_dump(),
            message="Address normalized successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to normalize address: {str(e)}"
        )


@router.post("/match", response_model=AddressMatchResult)
async def match_address(
    address_input: AddressInput,
    service: FIASVectorService = Depends(get_fias_service),
) -> AddressMatchResult:
    """
    Find matching addresses without creating a record.
    
    Returns match quality and suggestions:
    - **exact** (>=0.95): Use existing address
    - **high** (0.82-0.95): High confidence match
    - **medium** (0.75-0.82): Manual review recommended
    - **low** (<0.75): Create new address
    - **new**: No similar addresses found
    """
    try:
        # Normalize first
        normalized = await service.normalize_address(address_input.raw_address)
        
        # Find matches
        result = await service.find_best_match(normalized)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to match address: {str(e)}"
        )


@router.post("/search", response_model=AddressSearchResponse)
async def search_addresses(
    request: AddressSearchRequest,
    service: FIASVectorService = Depends(get_fias_service),
) -> AddressSearchResponse:
    """
    Search addresses by semantic similarity.
    
    Uses vector search to find addresses similar to the query.
    Supports filtering by region and city.
    """
    try:
        matches = await service.search_addresses(
            query=request.query,
            region_filter=request.region,
            city_filter=request.city,
            top_k=request.top_k,
        )
        return AddressSearchResponse(
            query=request.query,
            results=matches,
            total=len(matches),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/{record_id}", response_model=AddressRecordOut)
async def get_address(
    record_id: int,
    service: FIASVectorService = Depends(get_fias_service),
) -> AddressRecordOut:
    """Get address record by ID."""
    record = await service.get_address_by_id(record_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Address record {record_id} not found"
        )
    return record


@router.get("/health", response_model=dict)
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "fias-vector"}
