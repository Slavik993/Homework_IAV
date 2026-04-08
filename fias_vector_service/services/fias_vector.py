"""Main FIAS Vector Service with smart upsert logic."""
import uuid
from typing import Optional, List, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from config.logger import get_logger
from models.fias import AddressRecord
from models.schemas import (
    AddressInput,
    NormalizedAddress,
    AddressMatch,
    AddressMatchResult,
    MatchQuality,
    AddressCreateResponse,
    AddressRecordOut,
    VectorSearchFilter,
)
from services.qdrant_client import QdrantService
from services.embeddings import get_embedding_service
from services.normalizer import AddressNormalizer

logger = get_logger(__name__)
settings = get_settings()


class FIASVectorService:
    """
    Smart FIAS address service with vector-based deduplication.
    
    Features:
    - Address normalization via FIAS
    - Semantic vector search for deduplication
    - Hybrid search (vector + metadata filters)
    - Smart upsert with configurable thresholds
    """

    def __init__(
        self,
        db: AsyncSession,
        qdrant: Optional[QdrantService] = None,
    ):
        self.db = db
        self.qdrant = qdrant or QdrantService()
        self.normalizer = AddressNormalizer(db)
        self.embedding_service = get_embedding_service()

    async def initialize(self) -> None:
        """Initialize service connections."""
        await self.qdrant.connect()
        logger.info("fias_vector_service_initialized")

    async def close(self) -> None:
        """Close service connections."""
        await self.qdrant.disconnect()

    async def normalize_address(self, raw_address: str) -> NormalizedAddress:
        """
        Normalize raw address using FIAS database.
        
        Args:
            raw_address: Raw address string
            
        Returns:
            NormalizedAddress with FIAS IDs where found
        """
        logger.debug("normalizing_address", raw_address=raw_address)
        normalized = await self.normalizer.normalize(raw_address)
        logger.info("address_normalized", 
                   full_address=normalized.full_address,
                   region=normalized.region)
        return normalized

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for address text.
        
        Args:
            text: Address text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self.embedding_service.encode_single(text)

    async def semantic_search(
        self,
        address: NormalizedAddress,
        region_filter: Optional[str] = None,
        city_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[AddressMatch]:
        """
        Perform semantic search for similar addresses.
        
        Args:
            address: Normalized address to search
            region_filter: Optional region filter
            city_filter: Optional city filter
            top_k: Number of results to return
            
        Returns:
            List of matching addresses with similarity scores
        """
        # Generate embedding for the address
        embedding_text = address.to_embedding_text()
        vector = self.get_embedding(embedding_text)
        
        # Build search filter
        search_filter = VectorSearchFilter(
            region=region_filter or address.region,
            city=city_filter or address.city,
        )
        
        # Search in vector DB
        matches = await self.qdrant.search(
            vector=vector,
            search_filter=search_filter,
            top_k=top_k,
        )
        
        logger.debug("semantic_search_completed", 
                    query=embedding_text,
                    results_count=len(matches),
                    top_similarity=matches[0].similarity if matches else 0)
        
        return matches

    def _determine_match_quality(self, similarity: float) -> MatchQuality:
        """Determine match quality based on similarity score."""
        if similarity >= settings.similarity_high:
            return MatchQuality.EXACT
        elif similarity >= settings.similarity_medium:
            return MatchQuality.HIGH
        elif similarity >= settings.similarity_low:
            return MatchQuality.MEDIUM
        else:
            return MatchQuality.LOW

    async def find_best_match(
        self,
        address: NormalizedAddress,
    ) -> AddressMatchResult:
        """
        Find best matching address with deduplication logic.
        
        Args:
            address: Normalized address to match
            
        Returns:
            AddressMatchResult with match quality and suggestions
        """
        # Semantic search
        matches = await self.semantic_search(address)
        
        if not matches:
            return AddressMatchResult(
                input_address=address.full_address,
                normalized=address,
                match_quality=MatchQuality.NEW,
                best_match=None,
                alternatives=[],
                suggested_action="create_new",
            )
        
        best_match = matches[0]
        match_quality = self._determine_match_quality(best_match.similarity)
        
        # Determine suggested action
        if match_quality == MatchQuality.EXACT:
            suggested_action = "use_existing"
        elif match_quality == MatchQuality.HIGH:
            suggested_action = "use_existing"  # Could be "manual_review" for strict cases
        elif match_quality == MatchQuality.MEDIUM:
            suggested_action = "manual_review"
        else:
            suggested_action = "create_new"
        
        # Alternatives are other matches (excluding best)
        alternatives = matches[1:5] if len(matches) > 1 else []
        
        return AddressMatchResult(
            input_address=address.full_address,
            normalized=address,
            match_quality=match_quality,
            best_match=best_match,
            alternatives=alternatives,
            suggested_action=suggested_action,
        )

    async def smart_upsert(
        self,
        address_input: AddressInput,
        auto_create: bool = True,
    ) -> AddressCreateResponse:
        """
        Smart upsert: normalize, deduplicate, and store address.
        
        Logic:
        1. Normalize address via FIAS
        2. Generate embedding
        3. Search for similar addresses (vector search)
        4. Based on similarity:
           - >= 0.95: Use existing (exact match)
           - 0.82-0.95: Use existing with high confidence
           - 0.75-0.82: Manual review / suggest alternatives
           - < 0.75: Create new address
        5. Store in PostgreSQL and vector DB
        
        Args:
            address_input: Input address data
            auto_create: Whether to auto-create if no match found
            
        Returns:
            AddressCreateResponse with result and action taken
        """
        try:
            # Step 1: Normalize
            logger.info("smart_upsert_started", raw_address=address_input.raw_address)
            normalized = await self.normalize_address(address_input.raw_address)
            
            # Step 2 & 3: Find best match
            match_result = await self.find_best_match(normalized)
            
            # Step 4: Decide action
            if match_result.match_quality in (MatchQuality.EXACT, MatchQuality.HIGH):
                # Use existing match
                logger.info("using_existing_address", 
                           match_quality=match_result.match_quality,
                           similarity=match_result.best_match.similarity if match_result.best_match else 0)
                
                return AddressCreateResponse(
                    success=True,
                    record=None,  # Not created, using existing
                    match_result=match_result,
                    message=f"Address matched to existing (similarity: {match_result.best_match.similarity:.3f})",
                )
            
            elif match_result.match_quality == MatchQuality.MEDIUM and not auto_create:
                # Require manual review
                return AddressCreateResponse(
                    success=False,
                    record=None,
                    match_result=match_result,
                    message="Similar address found, manual review required",
                )
            
            # Step 5: Create new address
            record = await self._create_address_record(normalized, address_input.raw_address)
            
            logger.info("address_created", 
                       record_id=record.id,
                       vector_id=str(record.vector_id))
            
            return AddressCreateResponse(
                success=True,
                record=AddressRecordOut.model_validate(record),
                match_result=match_result,
                message="New address created successfully",
            )
            
        except Exception as e:
            logger.error("smart_upsert_failed", error=str(e), exc_info=True)
            raise

    async def _create_address_record(
        self,
        normalized: NormalizedAddress,
        source_raw: str,
    ) -> AddressRecord:
        """
        Create new address record in PostgreSQL and vector DB.
        
        Args:
            normalized: Normalized address
            source_raw: Original raw address string
            
        Returns:
            Created AddressRecord
        """
        # Generate vector ID
        vector_id = uuid.uuid4()
        
        # Create DB record
        record = AddressRecord(
            vector_id=vector_id,
            ao_guid=normalized.region_fias_id,  # Using region as primary FIAS ref
            house_guid=normalized.house_fias_id,
            region=normalized.region,
            region_fias_id=normalized.region_fias_id,
            district=normalized.district,
            district_fias_id=normalized.district_fias_id,
            city=normalized.city,
            city_fias_id=normalized.city_fias_id,
            settlement=normalized.settlement,
            settlement_fias_id=normalized.settlement_fias_id,
            street=normalized.street,
            street_fias_id=normalized.street_fias_id,
            house=normalized.house,
            house_fias_id=normalized.house_fias_id,
            apartment=normalized.apartment,
            full_address=normalized.full_address,
            postal_code=normalized.postal_code,
            source_raw=source_raw,
            confidence_score=0.0,  # Will be updated based on FIAS match coverage
            is_verified=bool(normalized.house_fias_id),  # Verified if found in FIAS
        )
        
        self.db.add(record)
        await self.db.flush()  # Get ID without committing
        
        # Store in vector DB
        embedding_text = normalized.to_embedding_text()
        vector = self.get_embedding(embedding_text)
        
        await self.qdrant.upsert(
            vector_id=vector_id,
            vector=vector,
            address=normalized,
            metadata={
                "record_id": record.id,
                "ao_guid": str(normalized.region_fias_id) if normalized.region_fias_id else None,
                "house_guid": str(normalized.house_fias_id) if normalized.house_fias_id else None,
                "is_fias_verified": record.is_verified,
                "created_at": record.created_at.isoformat() if record.created_at else None,
            },
        )
        
        await self.db.commit()
        
        return record

    async def get_address_by_id(self, record_id: int) -> Optional[AddressRecordOut]:
        """Get address record by ID."""
        from sqlalchemy import select
        result = await self.db.execute(
            select(AddressRecord).where(AddressRecord.id == record_id)
        )
        record = result.scalar_one_or_none()
        if record:
            return AddressRecordOut.model_validate(record)
        return None

    async def search_addresses(
        self,
        query: str,
        region_filter: Optional[str] = None,
        city_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[AddressMatch]:
        """
        Search addresses by query string.
        
        Args:
            query: Search query
            region_filter: Optional region filter
            city_filter: Optional city filter
            top_k: Number of results
            
        Returns:
            List of matching addresses
        """
        # Normalize query first
        normalized = await self.normalize_address(query)
        
        # Search with filters
        matches = await self.semantic_search(
            normalized,
            region_filter=region_filter,
            city_filter=city_filter,
            top_k=top_k,
        )
        
        return matches
