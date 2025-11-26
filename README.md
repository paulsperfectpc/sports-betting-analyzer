# AI Gambling Project - v2

Sports betting analysis app using **SportsData.io API** for NFL/NBA/NHL team and player data, with **Ollama LLM** (qwen3:4b) powered betting recommendations.

## Features

- ✅ FastAPI REST API
- ✅ SportsData.io integration for real-time sports data
- ✅ SQLite caching with incremental updates
- ✅ **Ollama LLM integration** for intelligent betting analysis
- ✅ Historical context awareness for improved predictions
- ✅ Docker-ready for Portainer deployment

## Project Structure

```
ai-gambling-proj/
├── app/
│   ├── main.py              # FastAPI app with endpoints
│   ├── sportsdata_client.py # SportsData.io API client
│   ├── cache.py             # SQLite caching layer
│   └── llm.py               # Ollama LLM integration (qwen3:4b)
├── tests/                   # Unit tests
├── Dockerfile               # Python 3.11-slim container
├── docker-compose.yml       # Service definition
├── requirements.txt         # Dependencies
├── .env.example            # Environment variable template
└── app-requirements-v2.md  # Detailed functional requirements
```

## Quick Start

### Local Development

1. **Set up environment**:
```bash
cp .env.example .env
# Edit .env and add your SPORTSDATA_API_KEY
```

2. **Run with Docker Compose**:
```bash
docker compose build
docker compose up
```

3. **Test the API**:
```bash
# Health check
curl http://localhost:11200/health

# Test Ollama connection
curl http://localhost:11200/test-ollama

# Search teams (with LLM analysis)
curl -X POST http://localhost:11200/search/teams \
  -H "Content-Type: application/json" \
  -d '{"teams": ["Patriots", "Cowboys"]}'

# Search player (with prop analysis)
curl -X POST http://localhost:11200/search/player \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Patrick Mahomes", "sport": "nfl"}'
```

### Deploy to Portainer

1. **Build and save the image**:
```bash
docker compose build
docker save ai-gambling-proj:latest | gzip > ai-gambling-proj.tar.gz
```

2. **Upload to Portainer** (http://10.254.254.220:9000):
   - Go to **Images** → **Import**
   - Upload `ai-gambling-proj.tar.gz`

3. **Create Stack**:
   - Go to **Stacks** → **Add Stack**
   - Name: `ai-gambling-proj`
   - Copy content from `docker-compose.yml`
   - Set environment variables:
     - `SPORTSDATA_API_KEY`: Your API key
     - `DATABASE_URL`: sqlite:///./data.db
     - `LLM_PROVIDER`: ollama
     - `LLM_MODEL`: qwen3:4b
     - `LLM_BASE_URL`: http://10.254.254.203:11434
   - Deploy

## Environment Variables

Required in `.env` file:

```bash
# SportsData.io API key (1000 pulls/month)
SPORTSDATA_API_KEY=your_key_here

# Database (SQLite default)
DATABASE_URL=sqlite:///./data.db

# LLM Configuration (Ollama)
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b
LLM_BASE_URL=http://10.254.254.203:11434

# Optional: enable stub data for testing
TESTING=0
```

## API Endpoints

### `GET /health`
Health check endpoint.

**Response**: `{"status": "ok"}`

---

### `GET /test-ollama`
Test Ollama connection and verify model availability.

**Response**:
```json
{
  "status": "connected",
  "base_url": "http://10.254.254.203:11434",
  "target_model": "qwen3:4b",
  "available_models": ["qwen3:4b", "..."],
  "model_ready": true
}
```

**Use this endpoint to verify your Ollama setup before running team/player analysis.**

---

### `POST /search/teams`
Search for team data and get **LLM-powered betting analysis**.

**Request Body**:
```json
{
  "teams": ["Patriots", "Cowboys"]
}
```

**Response**:
```json
{
  "results": {
    "Patriots": [...game data...],
    "Cowboys": [...game data...]
  },
  "analysis": {
    "recommendation": "spread",
    "pick": "Patriots +3.5",
    "confidence": 75,
    "reasoning": "Patriots showing strong defensive trends...",
    "risk_level": "medium",
    "key_factors": ["Recent form", "Head-to-head history", "Injury report"],
    "model": "qwen3:4b",
    "teams_analyzed": ["Patriots", "Cowboys"]
  }
}
```

**The LLM analyzes:**
- Historical game results (W/L records)
- Recent performance trends
- Head-to-head matchups
- Provides specific betting recommendations with confidence levels

---

### `POST /search/player`
Search for player data and get **LLM-powered prop betting analysis**.

**Request Body**:
```json
{
  "player_name": "Patrick Mahomes",
  "sport": "nfl"
}
```

**Response**:
```json
{
  "player": "Patrick Mahomes",
  "sport": "nfl",
  "data": {...player game stats...},
  "analysis": {
    "recommendation": "over 2.5 passing TDs",
    "line_value": "2.5",
    "confidence": 80,
    "hit_rate": "70% in last 10 games",
    "reasoning": "Mahomes averaging 2.8 TDs per game...",
    "risk_level": "low",
    "trends": ["Consistent TD production", "Favorable matchup"],
    "model": "qwen3:4b",
    "player": "Patrick Mahomes"
  }
}
```

**The LLM analyzes:**
- Last 10 games performance stats
- Averages and trends
- Hit rate on proposed prop lines
- Risk assessment and confidence levels

---

## LLM Integration Details

### How It Works

1. **Historical Context**: Uses cached game data to build comprehensive prompts
2. **Intelligent Analysis**: Ollama's qwen3:4b model analyzes patterns and trends
3. **Structured Output**: Returns JSON-formatted recommendations with reasoning
4. **Learning Over Time**: As more data is cached, analysis becomes more accurate

### Prompt Strategy

The LLM receives:
- Recent game results with W/L records
- Score details and performance metrics
- Current betting lines (when available)
- Historical trends from cached data

It provides:
- Specific betting recommendations
- Confidence scores (0-100)
- Detailed reasoning
- Risk assessment (low/medium/high)
- Key factors influencing the pick

### Testing the LLM Connection

**Before using team/player analysis, test your Ollama setup:**

```bash
curl http://localhost:11200/test-ollama
```

This verifies:
- ✅ Ollama server is reachable at 10.254.254.203:11434
- ✅ qwen3:4b model is loaded and available
- ✅ API endpoint is responding correctly

## SportsData.io API Endpoints Used

- NFL: `https://api.sportsdata.io/v3/nfl/scores/json/TeamGameStats/2024REG/NYJ/5`
- NBA: Similar structure for basketball
- NHL: Similar structure for hockey

API key passed via header: `Ocp-Apim-Subscription-Key`

## Notes

- Rate limit: **1000 API calls/month** (be efficient with caching!)
- **Cache saves all data** to minimize API usage and improve LLM analysis
- **LLM learns from history** - more cached data = better predictions
- **Test Ollama first** using `/test-ollama` endpoint
- Player prop analysis endpoint is ready, player data fetching needs SportsData.io player endpoints

## Troubleshooting

### Ollama Connection Issues

If `/test-ollama` returns an error:

1. **Check Ollama is running**:
   ```bash
   curl http://10.254.254.203:11434/api/tags
   ```

2. **Verify qwen3:4b is pulled**:
   ```bash
   # On the Ollama server
   ollama list
   # If not present:
   ollama pull qwen3:4b
   ```

3. **Check network connectivity** from your Docker container to Ollama server

### LLM Not Providing Good Analysis

- Ensure you have **cached data** for the teams/players you're analyzing
- The more historical context, the better the recommendations
- Check that betting lines are being passed to the LLM (optional but helpful)

## License

Internal use only.
