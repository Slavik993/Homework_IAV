"""Tests for FIAS Vector Service."""
import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession

from models.schemas import (
    AddressInput, 
    NormalizedAddress, 
    MatchQuality,
    VectorSearchFilter,
)
from services.fias_vector import FIASVectorService


@pytest.fixture
def mock_db():
    """Mock database session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant service."""
    mock = AsyncMock()
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    return mock


@pytest.fixture
def mock_normalizer():
    """Mock address normalizer."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    mock = Mock()
    mock.encode_single.return_value = [0.1] * 1024
    mock.vector_size = 1024
    return mock


class TestFIASVectorService:
    """Test suite for FIASVectorService."""

    @pytest.mark.asyncio
    async def test_normalize_address(self, mock_db):
        """Test address normalization."""
        service = FIASVectorService(mock_db)
        service.normalizer.normalize = AsyncMock(return_value=NormalizedAddress(
            region="Москва",
            city="Москва",
            full_address="г Москва",
        ))
        
        result = await service.normalize_address("Москва")
        
        assert result.region == "Москва"
        assert result.city == "Москва"

    @pytest.mark.asyncio
    async def test_get_embedding(self, mock_db, mock_embedding_service):
        """Test embedding generation."""
        with patch("services.fias_vector.get_embedding_service", return_value=mock_embedding_service):
            service = FIASVectorService(mock_db)
            
            vector = service.get_embedding("test address")
            
            assert len(vector) == 1024
            assert vector[0] == 0.1

    @pytest.mark.asyncio
    async def test_determine_match_quality_exact(self, mock_db):
        """Test exact match quality determination."""
        service = FIASVectorService(mock_db)
        
        quality = service._determine_match_quality(0.96)
        
        assert quality == MatchQuality.EXACT

    @pytest.mark.asyncio
    async def test_determine_match_quality_high(self, mock_db):
        """Test high match quality determination."""
        service = FIASVectorService(mock_db)
        
        quality = service._determine_match_quality(0.90)
        
        assert quality == MatchQuality.HIGH

    @pytest.mark.asyncio
    async def test_determine_match_quality_medium(self, mock_db):
        """Test medium match quality determination."""
        service = FIASVectorService(mock_db)
        
        quality = service._determine_match_quality(0.80)
        
        assert quality == MatchQuality.MEDIUM

    @pytest.mark.asyncio
    async def test_determine_match_quality_low(self, mock_db):
        """Test low match quality determination."""
        service = FIASVectorService(mock_db)
        
        quality = service._determine_match_quality(0.70)
        
        assert quality == MatchQuality.LOW


class TestSmartUpsert:
    """Test smart upsert logic."""

    @pytest.mark.asyncio
    async def test_smart_upsert_exact_match(self, mock_db, mock_qdrant):
        """Test upsert with exact match - should use existing."""
        service = FIASVectorService(mock_db, mock_qdrant)
        service.normalizer.normalize = AsyncMock(return_value=NormalizedAddress(
            region="Москва",
            full_address="г Москва",
        ))
        
        # Mock semantic search to return exact match
        mock_match = Mock()
        mock_match.similarity = 0.97
        mock_match.address = NormalizedAddress(region="Москва", full_address="г Москва")
        mock_match.vector_id = uuid.uuid4()
        
        service.semantic_search = AsyncMock(return_value=[mock_match])
        
        result = await service.smart_upsert(
            AddressInput(raw_address="Москва")
        )
        
        assert result.success is True
        assert result.match_result.match_quality == MatchQuality.EXACT
        assert "matched" in result.message.lower()

    @pytest.mark.asyncio
    async def test_smart_upsert_no_match_creates_new(self, mock_db, mock_qdrant):
        """Test upsert with no match - should create new."""
        service = FIASVectorService(mock_db, mock_qdrant)
        service.normalizer.normalize = AsyncMock(return_value=NormalizedAddress(
            region="Новый Город",
            full_address="г Новый Город",
        ))
        
        # Mock semantic search to return no matches
        service.semantic_search = AsyncMock(return_value=[])
        
        # Mock create method
        mock_record = Mock()
        mock_record.id = 1
        mock_record.vector_id = uuid.uuid4()
        service._create_address_record = AsyncMock(return_value=mock_record)
        
        result = await service.smart_upsert(
            AddressInput(raw_address="Новый Город"),
            auto_create=True
        )
        
        assert result.success is True
        assert result.match_result.match_quality == MatchQuality.NEW


class TestSemanticSearch:
    """Test semantic search functionality."""

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters(self, mock_db, mock_qdrant, mock_embedding_service):
        """Test search with region and city filters."""
        with patch("services.fias_vector.get_embedding_service", return_value=mock_embedding_service):
            service = FIASVectorService(mock_db, mock_qdrant)
            service.normalizer.normalize = AsyncMock(return_value=NormalizedAddress(
                region="Москва",
                city="Москва",
                full_address="г Москва",
            ))
            
            mock_qdrant.search = AsyncMock(return_value=[])
            
            await service.semantic_search(
                NormalizedAddress(region="Москва", full_address="г Москва"),
                region_filter="Москва",
                city_filter="Москва",
                top_k=5,
            )
            
            # Verify search was called with correct filter
            call_args = mock_qdrant.search.call_args
            assert call_args[1]["search_filter"].region == "Москва"
            assert call_args[1]["search_filter"].city == "Москва"
