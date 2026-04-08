"""FastAPI dependencies."""
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import get_db
from services.fias_vector import FIASVectorService
from services.qdrant_client import QdrantService


async def get_fias_service(
    db: AsyncSession = Depends(get_db),
) -> AsyncGenerator[FIASVectorService, None]:
    """Dependency for FIAS Vector Service."""
    service = FIASVectorService(db)
    try:
        await service.initialize()
        yield service
    finally:
        await service.close()
