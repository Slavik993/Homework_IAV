"""Example usage of FIAS Vector Service."""
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from db.connection import AsyncSessionLocal
from services.fias_vector import FIASVectorService
from models.schemas import AddressInput


async def example_add_address():
    """Example: Add an address with smart deduplication."""
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        try:
            # Example 1: Add a new address
            result = await service.smart_upsert(
                AddressInput(
                    raw_address="г Москва, ул Ленина, д 10, кв 5",
                    region_hint="Москва"
                ),
                auto_create=True
            )
            
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Match Quality: {result.match_result.match_quality}")
            
            if result.record:
                print(f"Created Record ID: {result.record.id}")
                print(f"Full Address: {result.record.full_address}")
            elif result.match_result.best_match:
                print(f"Matched to existing: {result.match_result.best_match.vector_id}")
                print(f"Similarity: {result.match_result.best_match.similarity:.3f}")
        
        finally:
            await service.close()


async def example_normalize():
    """Example: Normalize an address without storing."""
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        try:
            normalized = await service.normalize_address(
                "г Санкт-Петербург, пр-т Невский, д 100"
            )
            
            print(f"Region: {normalized.region}")
            print(f"City: {normalized.city}")
            print(f"Street: {normalized.street}")
            print(f"House: {normalized.house}")
            print(f"Full: {normalized.full_address}")
        
        finally:
            await service.close()


async def example_search():
    """Example: Search addresses by semantic similarity."""
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        try:
            matches = await service.search_addresses(
                query="ленина 10 москва",
                region_filter="Москва",
                top_k=5
            )
            
            for i, match in enumerate(matches, 1):
                print(f"{i}. {match.address.full_address} (sim: {match.similarity:.3f})")
        
        finally:
            await service.close()


async def example_match_only():
    """Example: Find matches without creating records."""
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        try:
            normalized = await service.normalize_address(
                "Москва, ул Ленина, дом 10"
            )
            
            result = await service.find_best_match(normalized)
            
            print(f"Input: {result.input_address}")
            print(f"Quality: {result.match_quality}")
            print(f"Action: {result.suggested_action}")
            
            if result.best_match:
                print(f"Best Match: {result.best_match.address.full_address}")
                print(f"Similarity: {result.best_match.similarity:.3f}")
            
            if result.alternatives:
                print("\nAlternatives:")
                for alt in result.alternatives:
                    print(f"  - {alt.address.full_address} ({alt.similarity:.3f})")
        
        finally:
            await service.close()


async def main():
    """Run all examples."""
    print("=" * 50)
    print("Example: Normalize Address")
    print("=" * 50)
    await example_normalize()
    
    print("\n" + "=" * 50)
    print("Example: Add Address")
    print("=" * 50)
    await example_add_address()
    
    print("\n" + "=" * 50)
    print("Example: Search Addresses")
    print("=" * 50)
    await example_search()
    
    print("\n" + "=" * 50)
    print("Example: Match Only")
    print("=" * 50)
    await example_match_only()


if __name__ == "__main__":
    asyncio.run(main())
