# Scam Honeypot API

AI-powered honeypot system for scam detection and intelligence extraction.

## Features

- 🔍 **Real-time Scam Detection** - Pattern-based detection with <5ms latency
- 🎭 **Believable Personas** - 5 regional Indian personas with authentic language styles
- 🤖 **Autonomous Engagement** - LLM-powered multi-turn conversations
- 📊 **Intelligence Extraction** - UPI IDs, bank accounts, phone numbers, URLs
- 🔐 **API Key Authentication** - Secure access control
- 📡 **GUVI Callback** - Automatic result submission with retry logic

## Quick Start

### 1. Install Dependencies

```bash
cd scam-honeypot
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `API_KEY` - Your secret API key for authentication
- `NVIDIA_API_KEY` - Your NVIDIA API key from build.nvidia.com

### 3. Run the Server

```bash
# Development
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
curl -X POST http://localhost:8000/api/honeypot/interact \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_api_key" \
  -d '{
    "sessionId": "test-session-1",
    "message": {
      "sender": "scammer",
      "text": "Your bank account will be blocked. Share OTP now.",
      "timestamp": "2026-01-25T10:00:00Z"
    },
    "conversationHistory": [],
    "metadata": {
      "channel": "SMS",
      "language": "English",
      "locale": "IN"
    }
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/honeypot/interact` | POST | Main interaction endpoint |
| `/api/session/{id}` | GET | Get session info |
| `/api/session/{id}/terminate` | POST | End session & trigger callback |

## Project Structure

```
scam-honeypot/
├── main.py              # FastAPI app
├── config.py            # Settings
├── api/
│   ├── routes.py        # API endpoints
│   ├── schemas.py       # Pydantic models
│   └── auth.py          # Authentication
├── core/
│   ├── scam_detector.py # Scam detection
│   ├── agent_engine.py  # LLM agent
│   ├── session_manager.py
│   └── intelligence.py  # Entity extraction
├── personas/
│   └── templates.py     # Victim personas
├── utils/
│   ├── patterns.py      # Regex patterns
│   └── callback.py      # GUVI callback
└── tests/
```

## Running Tests

```bash
pytest tests/ -v
```

## Deployment

### Railway

```bash
railway login
railway init
railway up
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## License

MIT
