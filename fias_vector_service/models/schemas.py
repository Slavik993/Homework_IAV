"""Pydantic schemas for API."""
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field, ConfigDict


class AddressLevel(int, Enum):
    """FIAS Address Object Level."""
    REGION = 1
    AUTONOMOUS_DISTRICT = 2
    DISTRICT = 3
    CITY = 4
    INTRA_CITY = 5
    SETTLEMENT = 6
    STREET = 7
    BUILDING = 8
    PLANNING_STRUCTURE = 9
    LAND_PLOT = 10
    ROOM = 11
    CAR_PLACE = 12


class MatchQuality(str, Enum):
    """Quality of address match."""
    EXACT = "exact"           # similarity >= 0.95
    HIGH = "high"              # similarity 0.82-0.95
    MEDIUM = "medium"          # similarity 0.75-0.82
    LOW = "low"                # similarity < 0.75
    NEW = "new"                # no match found


class AddressComponent(BaseModel):
    """Single address component."""
    name: str
    short_name: Optional[str] = None
    fias_id: Optional[uuid.UUID] = None
    level: Optional[AddressLevel] = None


class AddressInput(BaseModel):
    """Input address for normalization/matching."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "raw_address": "г Москва, ул Ленина, д 10, кв 5",
            "region_hint": "Москва"
        }
    })
    
    raw_address: str = Field(..., description="Raw address string")
    region_hint: Optional[str] = Field(None, description="Optional region hint for filtering")
    city_hint: Optional[str] = Field(None, description="Optional city hint for filtering")
    postal_code: Optional[str] = Field(None, description="Optional postal code")


class NormalizedAddress(BaseModel):
    """Normalized address components."""
    region: str
    region_fias_id: Optional[uuid.UUID] = None
    district: Optional[str] = None
    district_fias_id: Optional[uuid.UUID] = None
    city: Optional[str] = None
    city_fias_id: Optional[uuid.UUID] = None
    settlement: Optional[str] = None
    settlement_fias_id: Optional[uuid.UUID] = None
    street: Optional[str] = None
    street_fias_id: Optional[uuid.UUID] = None
    house: Optional[str] = None
    house_fias_id: Optional[uuid.UUID] = None
    apartment: Optional[str] = None
    
    full_address: str
    postal_code: Optional[str] = None
    
    def to_embedding_text(self) -> str:
        """Generate text for embedding generation."""
        parts = [self.region]
        if self.district:
            parts.append(self.district)
        if self.city:
            parts.append(self.city)
        if self.settlement:
            parts.append(self.settlement)
        if self.street:
            parts.append(self.street)
        if self.house:
            parts.append(f"дом {self.house}")
        if self.apartment:
            parts.append(f"кв {self.apartment}")
        return ", ".join(parts)


class AddressMatch(BaseModel):
    """Single address match result."""
    vector_id: uuid.UUID
    similarity: float = Field(..., ge=0.0, le=1.0)
    address: NormalizedAddress
    ao_guid: Optional[uuid.UUID] = None
    house_guid: Optional[uuid.UUID] = None
    is_fias_verified: bool = False


class AddressMatchResult(BaseModel):
    """Result of address matching."""
    input_address: str
    normalized: NormalizedAddress
    match_quality: MatchQuality
    best_match: Optional[AddressMatch] = None
    alternatives: List[AddressMatch] = Field(default_factory=list)
    suggested_action: str  # "use_existing", "create_new", "manual_review"


class AddressRecordOut(BaseModel):
    """Output schema for stored address."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    vector_id: uuid.UUID
    ao_guid: Optional[uuid.UUID] = None
    house_guid: Optional[uuid.UUID] = None
    region: str
    district: Optional[str] = None
    city: Optional[str] = None
    settlement: Optional[str] = None
    street: Optional[str] = None
    house: Optional[str] = None
    apartment: Optional[str] = None
    full_address: str
    postal_code: Optional[str] = None
    confidence_score: float
    is_verified: bool
    created_at: datetime


class AddressCreateResponse(BaseModel):
    """Response for address creation."""
    success: bool
    record: Optional[AddressRecordOut] = None
    match_result: AddressMatchResult
    message: str


class VectorSearchFilter(BaseModel):
    """Filter for vector search."""
    region: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    ao_level: Optional[int] = None
