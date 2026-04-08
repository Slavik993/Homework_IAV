# FIAS Vector Address Service

Умная система записи адресов в основную базу данных через ФИАС с использованием векторной БД (Qdrant) и семантического поиска.

## Архитектура

```
┌──────────────┐      ┌─────────────────┐      ┌──────────────┐
│   FastAPI    │──────▶  FIASVector    │──────▶   Qdrant     │
│   Endpoint   │      │   Service      │      │ (Vector DB)  │
└──────────────┘      └────────┬────────┘      └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  PostgreSQL  │
                        │   + FIAS     │
                        └──────────────┘
```

## Компоненты

- **FIASVectorService** — основной сервис для нормализации, дедупликации и хранения адресов
- **Qdrant** — векторная БД для семантического поиска
- **PostgreSQL** — хранение нормализованных адресов и данных ФИАС
- **Sentence Transformers** — генерация эмбеддингов (`deepvk/USER-bge-m3`)

## Установка

```bash
# Клонирование репозитория
git clone <repo>
cd fias_vector_service

# Установка зависимостей
pip install -r requirements.txt

# Настройка окружения
cp .env.example .env
# Отредактируйте .env с вашими настройками
```

## Конфигурация (.env)

```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=fias_vector
POSTGRES_USER=fias_user
POSTGRES_PASSWORD=fias_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Embedding Model
EMBEDDING_MODEL=deepvk/USER-bge-m3
EMBEDDING_DEVICE=cpu

# Пороги схожести
SIMILARITY_HIGH=0.95
SIMILARITY_MEDIUM=0.82
SIMILARITY_LOW=0.75
```

## Запуск

```bash
# Запуск сервера
python main.py

# Или через uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### Добавление адреса (с дедупликацией)

```bash
POST /api/v1/addresses/add
```

```json
{
    "raw_address": "г Москва, ул Ленина, д 10, кв 5",
    "region_hint": "Москва"
}
```

**Логика smart upsert:**
- `similarity >= 0.95` → Использовать существующий адрес
- `0.82 <= similarity < 0.95` → Использовать существующий (высокая уверенность)
- `0.75 <= similarity < 0.82` → Ручная проверка рекомендуется
- `similarity < 0.75` → Создать новый адрес

### Нормализация адреса

```bash
POST /api/v1/addresses/normalize
```

```json
{
    "raw_address": "г Москва, ул Ленина, д 10"
}
```

### Поиск похожих адресов

```bash
POST /api/v1/addresses/match
```

Возвращает качество совпадения и рекомендуемое действие.

### Семантический поиск

```bash
POST /api/v1/addresses/search
```

```json
{
    "query": "ленина 10 москва",
    "region": "Москва",
    "top_k": 10
}
```

## Пример использования (Python)

```python
import asyncio
from fias_vector_service.services.fias_vector import FIASVectorService
from fias_vector_service.models.schemas import AddressInput
from db.connection import AsyncSessionLocal

async def main():
    async with AsyncSessionLocal() as db:
        service = FIASVectorService(db)
        await service.initialize()
        
        # Добавить адрес с автоматической дедупликацией
        result = await service.smart_upsert(
            AddressInput(raw_address="г Москва, ул Ленина, д 10, кв 5")
        )
        
        print(f"Success: {result.success}")
        print(f"Match Quality: {result.match_result.match_quality}")
        
        if result.record:
            print(f"Created: {result.record.full_address}")
        elif result.match_result.best_match:
            print(f"Matched: {result.match_result.best_match.address.full_address}")
        
        await service.close()

asyncio.run(main())
```

## Структура проекта

```
fias_vector_service/
├── api/
│   ├── __init__.py
│   ├── deps.py          # FastAPI dependencies
│   └── routes.py        # API endpoints
├── config/
│   ├── __init__.py
│   ├── settings.py      # Pydantic settings
│   └── logger.py        # Structlog configuration
├── db/
│   ├── __init__.py
│   └── connection.py    # SQLAlchemy async setup
├── models/
│   ├── __init__.py
│   ├── fias.py          # SQLAlchemy models
│   └── schemas.py       # Pydantic schemas
├── services/
│   ├── __init__.py
│   ├── qdrant_client.py # Qdrant integration
│   ├── embeddings.py    # Embedding service
│   ├── normalizer.py    # Address normalization
│   └── fias_vector.py   # Main service
├── examples/
│   └── usage.py         # Usage examples
├── tests/
│   └── test_services.py # Unit tests
├── main.py              # FastAPI app
├── requirements.txt
└── README.md
```

## Алгоритм Smart Upsert

```
1. Нормализация адреса через ФИАС
   └─ Парсинг компонентов (регион, город, улица, дом)
   └─ Поиск в БД ФИАС по каждому компоненту
   
2. Генерация эмбеддинга
   └─ Преобразование в текст: "Регион, Город, Улица, дом N"
   └─ USER-bge-m3 → 1024-мерный вектор
   
3. Векторный поиск (Qdrant)
   └─ HNSW search с cosine similarity
   └─ Фильтрация по региону/городу (metadata)
   
4. Определение действия
   └─ similarity >= 0.95: использовать существующий
   └─ 0.82-0.95: высокая уверенность
   └─ 0.75-0.82: ручная проверка
   └─ < 0.75: создать новый
   
5. Сохранение (при необходимости)
   └─ PostgreSQL: address_records
   └─ Qdrant: вектор + metadata
```

## Требования

- Python 3.11+
- PostgreSQL 14+ (с данными ФИАС)
- Qdrant (self-hosted или облако)
- 8GB+ RAM (для модели эмбеддингов)

## Лицензия

MIT
