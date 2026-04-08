"""Address normalization service using FIAS data."""
import re
import uuid
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from config.logger import get_logger
from models.fias import FiasAddressObject, FiasHouse
from models.schemas import NormalizedAddress, AddressComponent, AddressLevel

logger = get_logger(__name__)


class AddressNormalizer:
    """Service for normalizing addresses using FIAS database."""

    # AO levels mapping
    LEVEL_REGION = 1
    LEVEL_DISTRICT = 3
    LEVEL_CITY = 4
    LEVEL_SETTLEMENT = 6
    LEVEL_STREET = 7
    LEVEL_HOUSE = 8

    def __init__(self, db: AsyncSession):
        self.db = db

    async def normalize(self, raw_address: str) -> NormalizedAddress:
        """
        Normalize raw address string using FIAS database.
        
        This is a simplified implementation. In production, you would:
        1. Use fias library or DaData API for parsing
        2. Apply fuzzy matching for each component
        3. Use ML-based NER for address parsing
        """
        # Parse raw address components
        parsed = self._parse_raw_address(raw_address)
        
        # Match components to FIAS
        region_data = await self._match_region(parsed.get("region", ""))
        district_data = await self._match_district(
            parsed.get("district", ""),
            region_fias_id=region_data.get("fias_id") if region_data else None
        )
        city_data = await self._match_city(
            parsed.get("city", ""),
            region_fias_id=region_data.get("fias_id") if region_data else None,
            district_fias_id=district_data.get("fias_id") if district_data else None
        )
        settlement_data = await self._match_settlement(
            parsed.get("settlement", ""),
            city_fias_id=city_data.get("fias_id") if city_data else None
        )
        street_data = await self._match_street(
            parsed.get("street", ""),
            parent_fias_id=(
                settlement_data.get("fias_id") or 
                city_data.get("fias_id") or 
                district_data.get("fias_id") or
                region_data.get("fias_id")
            )
        )
        house_data = await self._match_house(
            parsed.get("house", ""),
            street_fias_id=street_data.get("fias_id") if street_data else None
        )

        # Build normalized address
        full_parts = []
        if region_data:
            full_parts.append(region_data.get("full_name", ""))
        if district_data:
            full_parts.append(district_data.get("full_name", ""))
        if city_data:
            full_parts.append(city_data.get("full_name", ""))
        if settlement_data:
            full_parts.append(settlement_data.get("full_name", ""))
        if street_data:
            full_parts.append(street_data.get("full_name", ""))
        if house_data:
            full_parts.append(f"д {house_data.get('number', '')}")
        if parsed.get("apartment"):
            full_parts.append(f"кв {parsed['apartment']}")

        return NormalizedAddress(
            region=region_data.get("name", parsed.get("region", "")) if region_data else parsed.get("region", ""),
            region_fias_id=uuid.UUID(region_data["fias_id"]) if region_data and region_data.get("fias_id") else None,
            district=district_data.get("name") if district_data else parsed.get("district"),
            district_fias_id=uuid.UUID(district_data["fias_id"]) if district_data and district_data.get("fias_id") else None,
            city=city_data.get("name") if city_data else parsed.get("city"),
            city_fias_id=uuid.UUID(city_data["fias_id"]) if city_data and city_data.get("fias_id") else None,
            settlement=settlement_data.get("name") if settlement_data else parsed.get("settlement"),
            settlement_fias_id=uuid.UUID(settlement_data["fias_id"]) if settlement_data and settlement_data.get("fias_id") else None,
            street=street_data.get("name") if street_data else parsed.get("street"),
            street_fias_id=uuid.UUID(street_data["fias_id"]) if street_data and street_data.get("fias_id") else None,
            house=house_data.get("number") if house_data else parsed.get("house"),
            house_fias_id=uuid.UUID(house_data["fias_id"]) if house_data and house_data.get("fias_id") else None,
            apartment=parsed.get("apartment"),
            full_address=", ".join(full_parts) if full_parts else raw_address,
            postal_code=parsed.get("postal_code"),
        )

    def _parse_raw_address(self, raw_address: str) -> Dict[str, Any]:
        """
        Parse raw address string into components.
        
        This is a regex-based parser. For production, consider:
        - Using DaData API
        - Using natasha/yargy libraries
        - Training custom NER model
        """
        result = {
            "region": "",
            "district": "",
            "city": "",
            "settlement": "",
            "street": "",
            "house": "",
            "building": "",
            "apartment": "",
            "postal_code": "",
        }

        # Postal code
        postal_match = re.search(r"(\d{6})", raw_address)
        if postal_match:
            result["postal_code"] = postal_match.group(1)

        # Apartment
        apt_patterns = [
            r"(?:кв|квартира|flat|apt)[:\.\s]*(\d+)",
            r"(?:кв|квартира|flat|apt)[:\.\s]*(\d+[а-я]?)",
        ]
        for pattern in apt_patterns:
            match = re.search(pattern, raw_address, re.IGNORECASE)
            if match:
                result["apartment"] = match.group(1)
                break

        # House with building
        house_patterns = [
            r"(?:д|дом|д\.|house)[:\.\s]*(\d+[\s/\-]?[а-яА-Яа-я]?)",
            r"(?:д|дом|д\.|house)[:\.\s]*(\d+)[\s/]*(?:корп|к|стр|с|б)?[:\.\s]*(\d+[а-я]?)",
        ]
        for pattern in house_patterns:
            match = re.search(pattern, raw_address, re.IGNORECASE)
            if match:
                result["house"] = match.group(1).strip()
                if len(match.groups()) > 1 and match.group(2):
                    result["building"] = match.group(2).strip()
                break

        # Street
        street_patterns = [
            r"(?:ул|улица|ул\.|пр-т|пр|проспект|пер|переулок|б-р|бульвар|ш|шоссе|наб|набережная)[:\.\s]*([^,;]+)",
            r"(?:,|\s|^)([^,;]*?(?:улица|проспект|переулок|бульвар|шоссе|набережная))",
        ]
        for pattern in street_patterns:
            match = re.search(pattern, raw_address, re.IGNORECASE)
            if match:
                street = match.group(1).strip()
                if len(street) > 3:
                    result["street"] = street
                    break

        # City
        city_patterns = [
            r"(?:г|гор|город|г\.)[:\.\s]*([^,;]+)",
            r"(?:,|\s|^)([^,;]*?(?:город|пос[её]лок|пгт|село|деревня))",
        ]
        for pattern in city_patterns:
            match = re.search(pattern, raw_address, re.IGNORECASE)
            if match:
                city = match.group(1).strip()
                if len(city) > 2:
                    result["city"] = city
                    break

        # Region (first word might be region)
        words = raw_address.split(",")
        if words and not result.get("city"):
            first_word = words[0].strip()
            if any(x in first_word.lower() for x in ["обл", "край", "респ", "ао", "г "]):
                result["region"] = first_word
            elif "москва" in raw_address.lower():
                result["region"] = "Москва"
                result["city"] = "Москва"
            elif "петербург" in raw_address.lower() or "спб" in raw_address.lower():
                result["region"] = "Санкт-Петербург"
                result["city"] = "Санкт-Петербург"

        return result

    async def _match_region(self, region_name: str) -> Optional[Dict[str, Any]]:
        """Match region to FIAS."""
        if not region_name:
            return None

        query = select(FiasAddressObject).where(
            and_(
                FiasAddressObject.ao_level == self.LEVEL_REGION,
                or_(
                    FiasAddressObject.formal_name.ilike(f"%{region_name}%"),
                    FiasAddressObject.off_name.ilike(f"%{region_name}%"),
                )
            )
        ).limit(1)

        result = await self.db.execute(query)
        obj = result.scalar_one_or_none()
        
        if obj:
            return {
                "fias_id": str(obj.ao_guid),
                "name": obj.formal_name,
                "full_name": f"{obj.short_name or ''} {obj.formal_name}".strip(),
            }
        return None

    async def _match_district(
        self, 
        district_name: str, 
        region_fias_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Match district to FIAS."""
        if not district_name:
            return None

        conditions = [
            FiasAddressObject.ao_level == self.LEVEL_DISTRICT,
            or_(
                FiasAddressObject.formal_name.ilike(f"%{district_name}%"),
                FiasAddressObject.off_name.ilike(f"%{district_name}%"),
            )
        ]
        
        if region_fias_id:
            parent_guid = uuid.UUID(region_fias_id)
            conditions.append(FiasAddressObject.parent_guid == parent_guid)

        query = select(FiasAddressObject).where(and_(*conditions)).limit(1)
        result = await self.db.execute(query)
        obj = result.scalar_one_or_none()
        
        if obj:
            return {
                "fias_id": str(obj.ao_guid),
                "name": obj.formal_name,
                "full_name": f"{obj.short_name or ''} {obj.formal_name}".strip(),
            }
        return None

    async def _match_city(
        self, 
        city_name: str,
        region_fias_id: Optional[str] = None,
        district_fias_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Match city to FIAS."""
        if not city_name:
            return None

        conditions = [
            FiasAddressObject.ao_level == self.LEVEL_CITY,
            or_(
                FiasAddressObject.formal_name.ilike(f"%{city_name}%"),
                FiasAddressObject.off_name.ilike(f"%{city_name}%"),
            )
        ]
        
        parent_id = district_fias_id or region_fias_id
        if parent_id:
            conditions.append(FiasAddressObject.parent_guid == uuid.UUID(parent_id))

        query = select(FiasAddressObject).where(and_(*conditions)).limit(1)
        result = await self.db.execute(query)
        obj = result.scalar_one_or_none()
        
        if obj:
            return {
                "fias_id": str(obj.ao_guid),
                "name": obj.formal_name,
                "full_name": f"{obj.short_name or ''} {obj.formal_name}".strip(),
            }
        return None

    async def _match_settlement(
        self,
        settlement_name: str,
        city_fias_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Match settlement to FIAS."""
        if not settlement_name:
            return None

        conditions = [
            FiasAddressObject.ao_level == self.LEVEL_SETTLEMENT,
            or_(
                FiasAddressObject.formal_name.ilike(f"%{settlement_name}%"),
                FiasAddressObject.off_name.ilike(f"%{settlement_name}%"),
            )
        ]
        
        if city_fias_id:
            conditions.append(FiasAddressObject.parent_guid == uuid.UUID(city_fias_id))

        query = select(FiasAddressObject).where(and_(*conditions)).limit(1)
        result = await self.db.execute(query)
        obj = result.scalar_one_or_none()
        
        if obj:
            return {
                "fias_id": str(obj.ao_guid),
                "name": obj.formal_name,
                "full_name": f"{obj.short_name or ''} {obj.formal_name}".strip(),
            }
        return None

    async def _match_street(
        self,
        street_name: str,
        parent_fias_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Match street to FIAS."""
        if not street_name:
            return None

        conditions = [
            FiasAddressObject.ao_level == self.LEVEL_STREET,
            or_(
                FiasAddressObject.formal_name.ilike(f"%{street_name}%"),
                FiasAddressObject.off_name.ilike(f"%{street_name}%"),
            )
        ]
        
        if parent_fias_id:
            conditions.append(FiasAddressObject.parent_guid == uuid.UUID(parent_fias_id))

        query = select(FiasAddressObject).where(and_(*conditions)).limit(1)
        result = await self.db.execute(query)
        obj = result.scalar_one_or_none()
        
        if obj:
            return {
                "fias_id": str(obj.ao_guid),
                "name": obj.formal_name,
                "full_name": f"{obj.short_name or ''} {obj.formal_name}".strip(),
            }
        return None

    async def _match_house(
        self,
        house_number: str,
        street_fias_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Match house to FIAS."""
        if not house_number or not street_fias_id:
            return None

        # Clean house number
        house_clean = re.sub(r"[^\d\w]", "", house_number).lower()
        
        query = select(FiasHouse).where(
            and_(
                FiasHouse.ao_guid == uuid.UUID(street_fias_id),
                or_(
                    FiasHouse.house_num.ilike(f"%{house_number}%"),
                    FiasHouse.house_num.ilike(f"%{house_clean}%"),
                )
            )
        ).limit(1)

        result = await self.db.execute(query)
        obj = result.scalar_one_or_none()
        
        if obj:
            full_num = obj.house_num or ""
            if obj.build_num:
                full_num += f" к{obj.build_num}"
            if obj.struc_num:
                full_num += f" с{obj.struc_num}"
                
            return {
                "fias_id": str(obj.house_guid),
                "number": full_num,
            }
        return None
