"""Demo script for FIAS Vector Service."""
import asyncio
import uuid
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from config.logger import configure_logging, get_logger
from db.connection import AsyncSessionLocal, init_db
from services.fias_vector import FIASVectorService
from models.schemas import AddressInput, NormalizedAddress

logger = get_logger(__name__)


async def demo_normalize():
    """Demo: Address normalization."""
    print("\n" + "=" * 60)
    print("DEMO 1: Address Normalization")
    print("=" * 60)
    
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        test_addresses = [
            "г Москва, ул Ленина, д 10, кв 5",
            "Санкт-Петербург, Невский проспект, 100",
            "Московская обл, г Домодедово, ул Каширское шоссе, 3а",
        ]
        
        for addr in test_addresses:
            print(f"\nInput: {addr}")
            normalized = await service.normalize_address(addr)
            print(f"  Region: {normalized.region}")
            print(f"  City: {normalized.city or 'N/A'}")
            print(f"  Street: {normalized.street or 'N/A'}")
            print(f"  House: {normalized.house or 'N/A'}")
            print(f"  Full: {normalized.full_address}")
        
        await service.close()


async def demo_smart_upsert():
    """Demo: Smart upsert with deduplication."""
    print("\n" + "=" * 60)
    print("DEMO 2: Smart Upsert (Deduplication)")
    print("=" * 60)
    
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        # First address - should create new
        addr1 = AddressInput(raw_address="г Москва, ул Ленина, д 10")
        print(f"\n1. Adding: {addr1.raw_address}")
        result1 = await service.smart_upsert(addr1)
        print(f"   Result: {result1.message}")
        print(f"   Quality: {result1.match_result.match_quality}")
        if result1.record:
            print(f"   Created ID: {result1.record.id}")
        
        # Similar address - should match existing
        addr2 = AddressInput(raw_address="Москва, Ленина улица, дом 10")
        print(f"\n2. Adding similar: {addr2.raw_address}")
        result2 = await service.smart_upsert(addr2)
        print(f"   Result: {result2.message}")
        print(f"   Quality: {result2.match_result.match_quality}")
        if result2.match_result.best_match:
            print(f"   Matched to: {result2.match_result.best_match.vector_id}")
            print(f"   Similarity: {result2.match_result.best_match.similarity:.3f}")
        
        # Different address - should create new
        addr3 = AddressInput(raw_address="г Санкт-Петербург, Невский пр-т, 50")
        print(f"\n3. Adding different: {addr3.raw_address}")
        result3 = await service.smart_upsert(addr3)
        print(f"   Result: {result3.message}")
        print(f"   Quality: {result3.match_result.match_quality}")
        
        await service.close()


async def demo_search():
    """Demo: Semantic search."""
    print("\n" + "=" * 60)
    print("DEMO 3: Semantic Search")
    print("=" * 60)
    
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        # First add some test data
        test_addresses = [
            "г Москва, ул Ленина, д 10",
            "г Москва, ул Ленина, д 15",
            "г Москва, ул Тверская, д 5",
            "г Санкт-Петербург, Невский пр-т, 100",
        ]
        
        print("\nAdding test addresses...")
        for addr in test_addresses:
            await service.smart_upsert(AddressInput(raw_address=addr))
        
        # Search
        queries = [
            "ленина 10 москва",
            "ленина дом 15",
            "тверская улица москва",
            "невский проспект питер",
        ]
        
        for query in queries:
            print(f"\nSearch: '{query}'")
            matches = await service.search_addresses(query, top_k=3)
            for i, match in enumerate(matches, 1):
                print(f"  {i}. {match.address.full_address} (score: {match.similarity:.3f})")
        
        await service.close()


async def demo_match_quality():
    """Demo: Match quality thresholds."""
    print("\n" + "=" * 60)
    print("DEMO 4: Match Quality Thresholds")
    print("=" * 60)
    
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        # Add a reference address
        ref_addr = AddressInput(raw_address="г Москва, ул Ленина, д 10, кв 5")
        await service.smart_upsert(ref_addr)
        
        # Test various similarities
        test_cases = [
            ("г Москва, ул Ленина, д 10, кв 5", "Exact match"),
            ("Москва, Ленина улица, дом 10, квартира 5", "Slight variation"),
            ("г Москва, ул Ленина, д 10", "Without apartment"),
            ("г Москва, ул Ленина, д 12", "Different house"),
            ("г Москва, ул Тверская, д 10", "Different street"),
        ]
        
        print(f"\nReference: {ref_addr.raw_address}\n")
        
        for test_addr, description in test_cases:
            normalized = await service.normalize_address(test_addr)
            result = await service.find_best_match(normalized)
            
            print(f"Test: {test_addr}")
            print(f"  Description: {description}")
            print(f"  Quality: {result.match_result.match_quality}")
            print(f"  Action: {result.suggested_action}")
            if result.best_match:
                print(f"  Similarity: {result.best_match.similarity:.3f}")
            print()
        
        await service.close()


async def demo_embedding():
    """Demo: Embedding generation."""
    print("\n" + "=" * 60)
    print("DEMO 5: Embedding Generation")
    print("=" * 60)
    
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        addresses = [
            "г Москва, ул Ленина, д 10",
            "Москва, Ленина улица, дом 10",
            "г Санкт-Петербург, Невский пр-т, 100",
        ]
        
        print("\nGenerating embeddings...")
        embeddings = []
        for addr in addresses:
            normalized = await service.normalize_address(addr)
            text = normalized.to_embedding_text()
            vector = service.get_embedding(text)
            embeddings.append((addr, vector))
            print(f"\nAddress: {addr}")
            print(f"Embedding text: {text}")
            print(f"Vector size: {len(vector)}")
            print(f"Vector (first 5): {vector[:5]}")
        
        # Compare similarities
        print("\n\nCosine similarities:")
        import numpy as np
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                v1 = np.array(embeddings[i][1])
                v2 = np.array(embeddings[j][1])
                similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                print(f"  '{embeddings[i][0]}' <-> '{embeddings[j][0]}': {similarity:.4f}")
        
        await service.close()


async def main():
    """Run all demos."""
    configure_logging("INFO")
    
    print("\n" + "=" * 60)
    print("FIAS VECTOR SERVICE DEMO")
    print("=" * 60)
    print(f"Time: {datetime.now()}")
    
    try:
        # Initialize DB
        await init_db()
        
        # Run demos
        await demo_normalize()
        await demo_smart_upsert()
        await demo_search()
        await demo_match_quality()
        await demo_embedding()
        
    except Exception as e:
        logger.error("demo_failed", error=str(e))
        raise
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
