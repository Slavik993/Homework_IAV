"""FIAS SQLAlchemy models."""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Integer, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class FiasAddressObject(Base):
    """FIAS Address Objects (ADDROBJ table)."""
    __tablename__ = "fias_address_objects"
    __table_args__ = (
        Index("idx_ao_aoguid", "ao_guid"),
        Index("idx_ao_parentguid", "parent_guid"),
        Index("idx_ao_formalname", "formal_name"),
        Index("idx_ao_aolevel", "ao_level"),
        {"schema": "fias"},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    ao_guid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True, nullable=False)
    parent_guid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    formal_name: Mapped[str] = mapped_column(String(255), nullable=False)
    off_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    short_name: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    ao_level: Mapped[int] = mapped_column(Integer, nullable=False)
    region_code: Mapped[str] = mapped_column(String(2), nullable=False)
    area_code: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    city_code: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    place_code: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    street_code: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    extr_code: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)
    sext_code: Mapped[Optional[str]] = mapped_column(String(3), nullable=True)
    postal_code: Mapped[Optional[str]] = mapped_column(String(6), nullable=True)
    okato: Mapped[Optional[str]] = mapped_column(String(11), nullable=True)
    oktmo: Mapped[Optional[str]] = mapped_column(String(11), nullable=True)
    code: Mapped[Optional[str]] = mapped_column(String(17), nullable=True)
    
    # Hierarchy references
    houses: Mapped[list["FiasHouse"]] = relationship("FiasHouse", back_populates="address_object")


class FiasHouse(Base):
    """FIAS Houses table."""
    __tablename__ = "fias_houses"
    __table_args__ = (
        Index("idx_h_houseguid", "house_guid"),
        Index("idx_h_aoguid", "ao_guid"),
        {"schema": "fias"},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    house_guid: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True, nullable=False)
    ao_guid: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        ForeignKey("fias.fias_address_objects.ao_guid"),
        nullable=False
    )
    house_num: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    build_num: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    struc_num: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    house_type: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    build_type: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    struc_type: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    postal_code: Mapped[Optional[str]] = mapped_column(String(6), nullable=True)
    okato: Mapped[Optional[str]] = mapped_column(String(11), nullable=True)
    oktmo: Mapped[Optional[str]] = mapped_column(String(11), nullable=True)
    
    address_object: Mapped["FiasAddressObject"] = relationship("FiasAddressObject", back_populates="houses")


class AddressRecord(Base):
    """Normalized addresses in the main database (custom table)."""
    __tablename__ = "address_records"
    __table_args__ = (
        Index("idx_ar_vector_id", "vector_id"),
        Index("idx_ar_full_address", "full_address"),
        Index("idx_ar_region", "region"),
        Index("idx_ar_city", "city"),
        {"schema": "public"},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    vector_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    
    # FIAS references (may be null if address not found in FIAS)
    ao_guid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    house_guid: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    
    # Normalized components
    region: Mapped[str] = mapped_column(String(255), nullable=False)
    region_fias_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    district: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    district_fias_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    city_fias_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    settlement: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    settlement_fias_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    street: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    street_fias_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    house: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    house_fias_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    apartment: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Full normalized address
    full_address: Mapped[str] = mapped_column(Text, nullable=False)
    postal_code: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Metadata
    source_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[float] = mapped_column(default=0.0)
    is_verified: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
