"""Qdrant vector database client."""
import uuid
from typing import List, Optional, Dict, Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)

from config.settings import get_settings
from config.logger import get_logger
from models.schemas import AddressMatch, NormalizedAddress, VectorSearchFilter

logger = get_logger(__name__)
settings = get_settings()


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = settings.qdrant_collection
        self.vector_size = settings.qdrant_vector_size

    async def connect(self) -> None:
        """Initialize connection to Qdrant."""
        kwargs = {
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
        }
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key

        self.client = AsyncQdrantClient(**kwargs)
        await self._ensure_collection()
        logger.info("qdrant_connected", host=settings.qdrant_host)

    async def disconnect(self) -> None:
        """Close connection."""
        if self.client:
            await self.client.close()
            logger.info("qdrant_disconnected")

    async def _ensure_collection(self) -> None:
        """Create collection if not exists."""
        collections = await self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for metadata filtering
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="region",
                field_schema="keyword",
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="city",
                field_schema="keyword",
            )
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="district",
                field_schema="keyword",
            )
            logger.info("collection_created", collection=self.collection_name)

    def _build_filter(self, search_filter: Optional[VectorSearchFilter]) -> Optional[Filter]:
        """Build Qdrant filter from search filter."""
        if not search_filter:
            return None

        conditions = []
        if search_filter.region:
            conditions.append(
                FieldCondition(
                    key="region",
                    match=MatchValue(value=search_filter.region)
                )
            )
        if search_filter.city:
            conditions.append(
                FieldCondition(
                    key="city",
                    match=MatchValue(value=search_filter.city)
                )
            )
        if search_filter.district:
            conditions.append(
                FieldCondition(
                    key="district",
                    match=MatchValue(value=search_filter.district)
                )
            )

        if conditions:
            return Filter(must=conditions)
        return None

    async def search(
        self,
        vector: List[float],
        search_filter: Optional[VectorSearchFilter] = None,
        top_k: int = 10,
    ) -> List[AddressMatch]:
        """Search similar addresses by vector."""
        if not self.client:
            raise RuntimeError("Qdrant client not connected")

        qdrant_filter = self._build_filter(search_filter)

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=128),
        )

        matches = []
        for point in results:
            payload = point.payload or {}
            address = NormalizedAddress(
                region=payload.get("region", ""),
                district=payload.get("district"),
                city=payload.get("city"),
                settlement=payload.get("settlement"),
                street=payload.get("street"),
                house=payload.get("house"),
                apartment=payload.get("apartment"),
                full_address=payload.get("full_address", ""),
                postal_code=payload.get("postal_code"),
                region_fias_id=uuid.UUID(payload["region_fias_id"]) if payload.get("region_fias_id") else None,
                district_fias_id=uuid.UUID(payload["district_fias_id"]) if payload.get("district_fias_id") else None,
                city_fias_id=uuid.UUID(payload["city_fias_id"]) if payload.get("city_fias_id") else None,
                settlement_fias_id=uuid.UUID(payload["settlement_fias_id"]) if payload.get("settlement_fias_id") else None,
                street_fias_id=uuid.UUID(payload["street_fias_id"]) if payload.get("street_fias_id") else None,
                house_fias_id=uuid.UUID(payload["house_fias_id"]) if payload.get("house_fias_id") else None,
            )
            
            matches.append(AddressMatch(
                vector_id=uuid.UUID(point.id),
                similarity=point.score,
                address=address,
                ao_guid=uuid.UUID(payload["ao_guid"]) if payload.get("ao_guid") else None,
                house_guid=uuid.UUID(payload["house_guid"]) if payload.get("house_guid") else None,
                is_fias_verified=payload.get("is_fias_verified", False),
            ))

        return matches

    async def upsert(
        self,
        vector_id: uuid.UUID,
        vector: List[float],
        address: NormalizedAddress,
        metadata: Dict[str, Any],
    ) -> None:
        """Insert or update address vector."""
        if not self.client:
            raise RuntimeError("Qdrant client not connected")

        payload = {
            "region": address.region,
            "district": address.district,
            "city": address.city,
            "settlement": address.settlement,
            "street": address.street,
            "house": address.house,
            "apartment": address.apartment,
            "full_address": address.full_address,
            "postal_code": address.postal_code,
            "region_fias_id": str(address.region_fias_id) if address.region_fias_id else None,
            "district_fias_id": str(address.district_fias_id) if address.district_fias_id else None,
            "city_fias_id": str(address.city_fias_id) if address.city_fias_id else None,
            "settlement_fias_id": str(address.settlement_fias_id) if address.settlement_fias_id else None,
            "street_fias_id": str(address.street_fias_id) if address.street_fias_id else None,
            "house_fias_id": str(address.house_fias_id) if address.house_fias_id else None,
            **metadata,
        }

        point = PointStruct(
            id=str(vector_id),
            vector=vector,
            payload={k: v for k, v in payload.items() if v is not None},
        )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )
        logger.debug("vector_upserted", vector_id=str(vector_id))

    async def delete(self, vector_id: uuid.UUID) -> None:
        """Delete vector by ID."""
        if not self.client:
            raise RuntimeError("Qdrant client not connected")

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=[str(vector_id)],
        )

    async def count(self) -> int:
        """Get total count of vectors in collection."""
        if not self.client:
            raise RuntimeError("Qdrant client not connected")

        result = await self.client.count(collection_name=self.collection_name)
        return result.count
