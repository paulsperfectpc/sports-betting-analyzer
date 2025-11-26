"""
Sports Betting Analyzer v2.0
Uses SportsData.io API for NFL, NBA, NHL data
Includes caching to minimize API calls and LLM analysis via Ollama
"""

from flask import Flask, render_template, request, jsonify
import requests
import sqlite3
import json
from datetime import datetime, timedelta
import os
import logging
import sys
from collections import deque

# =============================================================================
# IN-MEMORY LOG BUFFER FOR /logs PAGE
# =============================================================================
class LogBuffer(logging.Handler):
    def __init__(self, max_logs=500):
        super().__init__()
        self.logs = deque(maxlen=max_logs)
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': self.format(record)
        }
        self.logs.append(log_entry)
    
    def get_logs(self):
        return list(self.logs)
    
    def clear(self):
        self.logs.clear()

log_buffer = LogBuffer(max_logs=500)
log_buffer.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for verbose logging
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.addHandler(log_buffer)
logger.setLevel(logging.DEBUG)

app = Flask(__name__)

# =============================================================================
# ERROR HANDLERS - Always return JSON, never HTML
# =============================================================================
@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all uncaught exceptions and return JSON"""
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal error: {e}", exc_info=True)
    return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Flag to track if database has been initialized
_db_initialized = False

# =============================================================================
# CONFIGURATION
# =============================================================================
SPORTSDATA_API_KEY = os.getenv('SPORTSDATA_API_KEY', '1fdd78185de84dc1bd82ff59f254c087')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://10.254.254.203:11434')  # Updated Ollama host
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
DATABASE_PATH = os.getenv('DATABASE_PATH', '/app/data/sportsdata.db')

# SportsData.io API endpoints
SPORTSDATA_ENDPOINTS = {
    'NFL': {
        'scores': 'https://api.sportsdata.io/v3/nfl/scores/json',
        'stats': 'https://api.sportsdata.io/v3/nfl/stats/json',
        'odds': 'https://api.sportsdata.io/v3/nfl/odds/json'
    },
    'NBA': {
        'scores': 'https://api.sportsdata.io/v3/nba/scores/json',
        'stats': 'https://api.sportsdata.io/v3/nba/stats/json',
        'odds': 'https://api.sportsdata.io/v3/nba/odds/json'
    },
    'NHL': {
        'scores': 'https://api.sportsdata.io/v3/nhl/scores/json',
        'stats': 'https://api.sportsdata.io/v3/nhl/stats/json',
        'odds': 'https://api.sportsdata.io/v3/nhl/odds/json'
    }
}

# =============================================================================
# DATABASE SETUP
# =============================================================================
def init_database():
    """Initialize SQLite database for caching"""
    global _db_initialized
    if _db_initialized:
        return
    
    try:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
    
        # Teams cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                team_key TEXT NOT NULL,
                team_name TEXT NOT NULL,
                team_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sport, team_key)
            )
        ''')
        
        # Players cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                player_id TEXT NOT NULL,
                player_name TEXT NOT NULL,
                team_key TEXT,
                player_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sport, player_id)
            )
        ''')
        
        # Games cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_score INTEGER,
                away_score INTEGER,
                game_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sport, game_id)
            )
        ''')
        
        # Player game stats cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_game_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                player_id TEXT NOT NULL,
                game_id TEXT NOT NULL,
                game_date TEXT NOT NULL,
                stats_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sport, player_id, game_id)
            )
        ''')
        
        # Betting lines cache
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS betting_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sport TEXT NOT NULL,
                game_id TEXT NOT NULL,
                line_type TEXT NOT NULL,
                line_data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sport, game_id, line_type)
            )
        ''')
        
        # LLM conversation history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_type TEXT NOT NULL,
                query_params TEXT NOT NULL,
                llm_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        _db_initialized = True
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")


def get_db():
    """Get database connection"""
    # Ensure database is initialized before any connection
    init_database()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize database at module load time (for gunicorn)
init_database()


# =============================================================================
# SPORTSDATA.IO API FUNCTIONS
# =============================================================================
class SportsDataAPI:
    def __init__(self):
        self.api_key = SPORTSDATA_API_KEY
        self.request_count = 0
    
    def _make_request(self, url, params=None):
        """Make API request with key"""
        if params is None:
            params = {}
        params['key'] = self.api_key
        
        logger.debug(f"API Request to: {url}")
        
        try:
            response = requests.get(url, params=params, timeout=30)
            self.request_count += 1
            logger.info(f"API Request #{self.request_count}: {url} - Status: {response.status_code}")
            
            if not response.ok:
                logger.error(f"API Error: {response.status_code} - {response.text[:200]}")
                return None
            
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def get_current_season(self, sport):
        """Get current season identifier"""
        now = datetime.now()
        year = now.year
        
        if sport == 'NFL':
            # NFL season runs Sept-Feb, use year of season start
            return year if now.month >= 9 else year - 1
        elif sport == 'NBA':
            # NBA season runs Oct-June, use year of season start
            return year if now.month >= 10 else year - 1
        elif sport == 'NHL':
            # NHL season runs Oct-June, use year of season start
            return year if now.month >= 10 else year - 1
        return year
    
    def get_teams(self, sport):
        """Get all teams for a sport"""
        base_url = SPORTSDATA_ENDPOINTS[sport]['scores']
        # Correct endpoint is /Teams (capital T) for active teams
        url = f"{base_url}/Teams"
        return self._make_request(url)
    
    def get_team_schedule(self, sport, team_key, season=None):
        """Get team's schedule/games"""
        if season is None:
            season = self.get_current_season(sport)
        
        base_url = SPORTSDATA_ENDPOINTS[sport]['scores']
        
        if sport == 'NFL':
            url = f"{base_url}/Scores/{season}"
        elif sport in ['NBA', 'NHL']:
            url = f"{base_url}/Games/{season}"
        
        all_games = self._make_request(url)
        if not all_games:
            return []
        
        # Filter for team's games
        team_games = []
        for game in all_games:
            home = game.get('HomeTeam', '')
            away = game.get('AwayTeam', '')
            if team_key.upper() in [home.upper(), away.upper()]:
                team_games.append(game)
        
        return team_games
    
    def get_completed_games(self, sport, team_key, limit=10):
        """Get last N completed games for a team"""
        games = self.get_team_schedule(sport, team_key)
        
        completed = []
        for game in games:
            status = game.get('Status', '')
            if status in ['Final', 'F', 'F/OT']:
                completed.append(game)
        
        # Sort by date descending
        completed.sort(key=lambda x: x.get('DateTime', x.get('Day', '')), reverse=True)
        return completed[:limit]
    
    def get_upcoming_games(self, sport, team_key, limit=5):
        """Get upcoming games for a team"""
        games = self.get_team_schedule(sport, team_key)
        
        upcoming = []
        now = datetime.now()
        
        for game in games:
            status = game.get('Status', '')
            if status in ['Scheduled', 'InProgress']:
                game_date_str = game.get('DateTime', game.get('Day', ''))
                if game_date_str:
                    try:
                        game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                        if game_date.replace(tzinfo=None) >= now:
                            upcoming.append(game)
                    except:
                        upcoming.append(game)
        
        upcoming.sort(key=lambda x: x.get('DateTime', x.get('Day', '')))
        return upcoming[:limit]
    
    def get_game_odds(self, sport, game_id=None):
        """Get betting odds for games - returns current game odds from schedule data"""
        # The odds are typically included in the Games/Scores response
        # For dedicated odds, use TeamTrends or MatchupTrends
        base_url = SPORTSDATA_ENDPOINTS[sport]['scores']
        season = self.get_current_season(sport)
        
        if sport == 'NFL':
            url = f"{base_url}/Scores/{season}"
        else:
            url = f"{base_url}/Games/{season}"
        
        games = self._make_request(url)
        if not games:
            return None
            
        # Filter for upcoming games with odds
        upcoming_with_odds = []
        for game in games:
            status = game.get('Status', '')
            if status in ['Scheduled', 'InProgress']:
                # Games often include pregame odds
                if game.get('PointSpread') or game.get('OverUnder'):
                    upcoming_with_odds.append({
                        'GameId': game.get('GameID', game.get('ScoreID')),
                        'HomeTeam': game.get('HomeTeam'),
                        'AwayTeam': game.get('AwayTeam'),
                        'DateTime': game.get('DateTime', game.get('Day')),
                        'PointSpread': game.get('PointSpread'),
                        'OverUnder': game.get('OverUnder'),
                        'HomeTeamMoneyLine': game.get('HomeTeamMoneyLine'),
                        'AwayTeamMoneyLine': game.get('AwayTeamMoneyLine')
                    })
        
        if game_id:
            for game_odds in upcoming_with_odds:
                if str(game_odds.get('GameId')) == str(game_id):
                    return game_odds
        return upcoming_with_odds
    
    def get_team_trends(self, sport, team_key):
        """Get betting trends for a team"""
        base_url = SPORTSDATA_ENDPOINTS[sport]['odds']
        url = f"{base_url}/TeamTrends/{team_key.upper()}"
        return self._make_request(url)
    
    def get_matchup_trends(self, sport, team1, team2):
        """Get betting trends for a matchup between two teams"""
        base_url = SPORTSDATA_ENDPOINTS[sport]['odds']
        url = f"{base_url}/MatchupTrends/{team1.upper()}/{team2.upper()}"
        return self._make_request(url)
    
    def get_player_stats(self, sport, player_name):
        """Search for player and get their recent stats"""
        base_url = SPORTSDATA_ENDPOINTS[sport]['stats']
        season = self.get_current_season(sport)
        
        if sport == 'NFL':
            url = f"{base_url}/PlayerSeasonStats/{season}"
        elif sport == 'NBA':
            url = f"{base_url}/PlayerSeasonStats/{season}"
        elif sport == 'NHL':
            url = f"{base_url}/PlayerSeasonStats/{season}"
        
        all_players = self._make_request(url)
        if not all_players:
            return None
        
        # Search for player by name
        player_name_lower = player_name.lower()
        for player in all_players:
            full_name = player.get('Name', '').lower()
            if player_name_lower in full_name or full_name in player_name_lower:
                return player
        
        return None
    
    def get_player_game_logs(self, sport, player_id, limit=10):
        """Get player's game-by-game stats"""
        base_url = SPORTSDATA_ENDPOINTS[sport]['stats']
        season = self.get_current_season(sport)
        
        # All sports use the same format: PlayerGameStatsBySeason/{season}/{playerid}/{numberofgames}
        url = f"{base_url}/PlayerGameStatsBySeason/{season}/{player_id}/{limit}"
        
        game_logs = self._make_request(url)
        if game_logs:
            game_logs.sort(key=lambda x: x.get('DateTime', x.get('Day', '')), reverse=True)
            return game_logs[:limit]
        return []
    
    def get_player_props(self, sport, player_id):
        """Get betting props for a specific player"""
        base_url = SPORTSDATA_ENDPOINTS[sport]['odds']
        
        # Player props endpoint varies by sport
        if sport == 'NFL':
            url = f"{base_url}/PlayerPropsByPlayerID/{self.get_current_season(sport)}/1/{player_id}"
        else:
            url = f"{base_url}/PlayerPropsByDate/{datetime.now().strftime('%Y-%m-%d')}"
        
        props = self._make_request(url)
        if props and sport != 'NFL':
            # Filter for specific player
            return [p for p in props if str(p.get('PlayerID')) == str(player_id)]
        return props


# =============================================================================
# CACHING FUNCTIONS
# =============================================================================
def get_cached_team_games(sport, team_key, limit=10):
    """Get cached games for a team"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM games 
        WHERE sport = ? AND (home_team = ? OR away_team = ?)
        ORDER BY game_date DESC
        LIMIT ?
    ''', (sport, team_key.upper(), team_key.upper(), limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def cache_game(sport, game_data):
    """Cache a game's data"""
    conn = get_db()
    cursor = conn.cursor()
    
    game_id = game_data.get('GameID', game_data.get('ScoreID'))
    game_date = game_data.get('DateTime', game_data.get('Day', ''))
    
    cursor.execute('''
        INSERT OR REPLACE INTO games 
        (sport, game_id, game_date, home_team, away_team, home_score, away_score, game_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        sport,
        str(game_id),
        game_date,
        game_data.get('HomeTeam', ''),
        game_data.get('AwayTeam', ''),
        game_data.get('HomeTeamScore', game_data.get('HomeScore')),
        game_data.get('AwayTeamScore', game_data.get('AwayScore')),
        json.dumps(game_data)
    ))
    
    conn.commit()
    conn.close()


def get_cached_count(sport, team_key):
    """Get count of cached games for a team"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT COUNT(*) FROM games 
        WHERE sport = ? AND (home_team = ? OR away_team = ?)
    ''', (sport, team_key.upper(), team_key.upper()))
    
    count = cursor.fetchone()[0]
    conn.close()
    return count


# =============================================================================
# LLM INTEGRATION
# =============================================================================
def analyze_with_llm(data, query_type):
    """Send data to Ollama for analysis - with extended timeout for slower GPUs"""
    logger.info(f"Starting LLM analysis for query_type: {query_type}")
    
    try:
        # Limit data size to prevent huge prompts
        limited_data = {k: v for k, v in data.items() if k not in ['llm_analysis']}
        
        # Truncate game data to last 3 games to keep prompt small for faster inference
        if 'recent_games' in limited_data:
            limited_data['recent_games'] = limited_data['recent_games'][:3]
        if 'team1' in limited_data and 'recent_games' in limited_data.get('team1', {}):
            limited_data['team1']['recent_games'] = limited_data['team1']['recent_games'][:3]
        if 'team2' in limited_data and 'recent_games' in limited_data.get('team2', {}):
            limited_data['team2']['recent_games'] = limited_data['team2']['recent_games'][:3]
        
        if query_type == 'team':
            prompt = f"""Analyze this team's betting outlook briefly.

Data: {json.dumps(limited_data, indent=2)}

Provide: 1) Next game prediction 2) Best bet recommendation 3) Key factors. Be concise."""

        elif query_type == 'team_comparison':
            prompt = f"""Analyze this matchup briefly.

Data: {json.dumps(limited_data, indent=2)}

Provide: 1) Predicted winner 2) Spread pick 3) Over/Under pick. Be concise."""

        elif query_type == 'player':
            prompt = f"""Analyze this player's props briefly.

Data: {json.dumps(limited_data, indent=2)}

Provide: 1) Recent form 2) Best prop bet 3) Confidence level. Be concise."""

        else:
            prompt = f"Analyze briefly: {json.dumps(limited_data)}"
        
        # Detailed logging for LLM debugging
        ollama_url = f"{OLLAMA_HOST}/api/generate"
        logger.info(f"=" * 50)
        logger.info(f"LLM REQUEST STARTING")
        logger.info(f"Ollama URL: {ollama_url}")
        logger.info(f"Ollama Model: {OLLAMA_MODEL}")
        logger.info(f"Prompt length: {len(prompt)} chars")
        logger.info(f"=" * 50)
        
        # Test connectivity first
        try:
            logger.info(f"Testing Ollama connectivity...")
            test_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
            logger.info(f"Ollama connectivity test: {test_response.status_code}")
            if test_response.ok:
                models = test_response.json().get('models', [])
                model_names = [m.get('name', 'unknown') for m in models]
                logger.info(f"Available models: {model_names}")
        except Exception as conn_err:
            logger.error(f"Ollama connectivity test FAILED: {conn_err}")
        
        # Call Ollama with extended timeout for GTX 1070 (90 seconds)
        logger.info(f"Sending generate request to Ollama...")
        response = requests.post(
            ollama_url,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 300  # Reduced for faster response on slower GPU
                }
            },
            timeout=90  # Extended timeout for GTX 1070
        )
        
        logger.info(f"Ollama response received!")
        logger.info(f"Ollama response status: {response.status_code}")
        logger.info(f"Response headers: {dict(response.headers)}")
        
        if response.ok:
            result = response.json()
            analysis = result.get('response', 'No response generated')
            logger.info(f"LLM analysis complete, length: {len(analysis)} chars")
            logger.info(f"=" * 50)
            return {
                "success": True,
                "analysis": analysis
            }
        else:
            logger.error(f"Ollama error: {response.status_code} - {response.text[:500]}")
            logger.info(f"=" * 50)
            return {
                "success": False,
                "error": f"Ollama returned status {response.status_code}"
            }
            
    except requests.exceptions.Timeout as e:
        logger.error(f"=" * 50)
        logger.error(f"LLM TIMEOUT ERROR")
        logger.error(f"Ollama URL: {OLLAMA_HOST}/api/generate")
        logger.error(f"Timeout after 90 seconds")
        logger.error(f"Exception: {e}")
        logger.error(f"=" * 50)
        return {"success": False, "error": f"LLM request timed out connecting to {OLLAMA_HOST}"}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"=" * 50)
        logger.error(f"LLM CONNECTION ERROR")
        logger.error(f"Cannot connect to Ollama at: {OLLAMA_HOST}")
        logger.error(f"Exception: {e}")
        logger.error(f"=" * 50)
        return {"success": False, "error": f"Cannot connect to Ollama at {OLLAMA_HOST} - check if Ollama is running"}
    except Exception as e:
        logger.error(f"=" * 50)
        logger.error(f"LLM UNEXPECTED ERROR")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception: {e}", exc_info=True)
        logger.error(f"=" * 50)
        return {"success": False, "error": str(e)}


# =============================================================================
# API ROUTES
# =============================================================================
api = SportsDataAPI()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/api/search/team', methods=['POST'])
def search_team():
    """Search for a team and get their recent games + odds"""
    data = request.json
    sport = data.get('sport', '').upper()
    team_name = data.get('team', '').strip()
    
    if not sport or not team_name:
        return jsonify({"error": "Sport and team name required"}), 400
    
    if sport not in SPORTSDATA_ENDPOINTS:
        return jsonify({"error": f"Sport {sport} not supported"}), 400
    
    logger.info(f"Team search: {team_name} ({sport})")
    
    try:
        # First, verify the team exists by getting all teams
        all_teams = api.get_teams(sport)
        if not all_teams:
            return jsonify({"error": f"Could not fetch teams for {sport}. API may be unavailable."}), 500
        
        # Find matching team
        team_key = None
        for team in all_teams:
            team_abbrev = team.get('Key', team.get('TeamID', ''))
            team_full_name = team.get('Name', team.get('FullName', ''))
            team_city = team.get('City', '')
            
            if (team_name.upper() == str(team_abbrev).upper() or 
                team_name.lower() in team_full_name.lower() or
                team_name.lower() in team_city.lower()):
                team_key = team_abbrev
                break
        
        if not team_key:
            return jsonify({
                "error": f"Team '{team_name}' not found in {sport}",
                "available_teams": [t.get('Key', t.get('TeamID')) for t in all_teams[:32]]
            }), 404
        
        logger.info(f"Found team: {team_key}")
        
        # Check cache first
        cached_games = get_cached_team_games(sport, team_key, 10)
        logger.info(f"Cached games found: {len(cached_games)}")
        
        # Determine how many new games to fetch
        games_needed = 10 - len(cached_games)
        
        if games_needed > 0 or len(cached_games) == 0:
            # Fetch from API
            completed_games = api.get_completed_games(sport, team_key, 10)
            logger.info(f"Fetched {len(completed_games)} completed games from API")
            
            # Cache new games
            for game in completed_games:
                cache_game(sport, game)
            
            games_data = completed_games
        else:
            # Use cached data
            games_data = [json.loads(g['game_data']) for g in cached_games]
        
        # Ensure games are sorted by date (newest first)
        games_data.sort(key=lambda x: x.get('DateTime', x.get('Day', '')), reverse=True)
        
        # Get upcoming games
        upcoming = api.get_upcoming_games(sport, team_key, 3)
        logger.info(f"Fetched {len(upcoming)} upcoming games")
        
        # Get odds for upcoming games
        odds_data = []
        for game in upcoming[:1]:  # Just get odds for next game to save API calls
            game_id = game.get('GameID', game.get('ScoreID'))
            if game_id:
                odds = api.get_game_odds(sport, game_id)
                if odds:
                    odds_data.append(odds)
        
        # Try to get team trends (betting performance)
        team_trends = api.get_team_trends(sport, team_key)
        
        result = {
            "team": team_key,
            "sport": sport,
            "recent_games": games_data[:10],
            "upcoming_games": upcoming,
            "betting_lines": odds_data,
            "team_trends": team_trends if team_trends else None,
            "games_from_cache": len(cached_games),
            "api_calls_made": api.request_count
        }
        
        logger.info(f"Team data compiled, returning results (LLM analysis will be fetched separately)")
        
        # Return data immediately - LLM analysis fetched separately via /api/analyze
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Team search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/search/teams', methods=['POST'])
def search_teams():
    """Compare two teams"""
    data = request.json
    sport = data.get('sport', '').upper()
    team1 = data.get('team1', '').strip()
    team2 = data.get('team2', '').strip()
    
    if not sport or not team1 or not team2:
        return jsonify({"error": "Sport and both team names required"}), 400
    
    logger.info(f"Team comparison: {team1} vs {team2} ({sport})")
    
    try:
        # Get last 5 games for each team
        team1_games = api.get_completed_games(sport, team1, 5)
        team2_games = api.get_completed_games(sport, team2, 5)
        
        # Ensure sorted by date (newest first)
        team1_games.sort(key=lambda x: x.get('DateTime', x.get('Day', '')), reverse=True)
        team2_games.sort(key=lambda x: x.get('DateTime', x.get('Day', '')), reverse=True)
        
        # Cache games
        for game in team1_games + team2_games:
            cache_game(sport, game)
        
        # Check if teams are playing each other soon
        team1_upcoming = api.get_upcoming_games(sport, team1, 3)
        matchup_game = None
        
        for game in team1_upcoming:
            home = game.get('HomeTeam', '').upper()
            away = game.get('AwayTeam', '').upper()
            if team2.upper() in [home, away]:
                matchup_game = game
                break
        
        # Get odds for matchup if exists
        matchup_odds = None
        if matchup_game:
            game_id = matchup_game.get('GameID', matchup_game.get('ScoreID'))
            if game_id:
                matchup_odds = api.get_game_odds(sport, game_id)
        
        # Get matchup trends (head-to-head betting trends)
        matchup_trends = api.get_matchup_trends(sport, team1, team2)
        
        result = {
            "sport": sport,
            "team1": {
                "name": team1.upper(),
                "recent_games": team1_games
            },
            "team2": {
                "name": team2.upper(),
                "recent_games": team2_games
            },
            "matchup": matchup_game,
            "betting_lines": matchup_odds,
            "matchup_trends": matchup_trends if matchup_trends else None,
            "api_calls_made": api.request_count
        }
        
        logger.info(f"Team comparison data compiled, returning results")
        
        # Return data immediately - LLM analysis fetched separately via /api/analyze
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Team comparison error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/search/player', methods=['POST'])
def search_player():
    """Search for a player and get their stats"""
    data = request.json
    sport = data.get('sport', '').upper()
    player_name = data.get('player', '').strip()
    
    if not sport or not player_name:
        return jsonify({"error": "Sport and player name required"}), 400
    
    logger.info(f"Player search: {player_name} ({sport})")
    
    try:
        # Get player info
        player_stats = api.get_player_stats(sport, player_name)
        
        if not player_stats:
            return jsonify({"error": f"Player '{player_name}' not found"}), 404
        
        player_id = player_stats.get('PlayerID')
        
        # Get game logs
        game_logs = api.get_player_game_logs(sport, player_id, 10)
        
        # Get player props if available
        props = api.get_player_props(sport, player_id)
        
        result = {
            "player": player_stats.get('Name', player_name),
            "sport": sport,
            "team": player_stats.get('Team', 'Unknown'),
            "season_stats": player_stats,
            "game_logs": game_logs,
            "betting_props": props,
            "api_calls_made": api.request_count
        }
        
        logger.info(f"Player data compiled, returning results")
        
        # Return data immediately - LLM analysis fetched separately via /api/analyze
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Player search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/test/ollama', methods=['POST'])
def test_ollama():
    """Test Ollama connection with detailed logging"""
    data = request.json or {}
    prompt = data.get('prompt', 'Hello, respond with a short greeting.')
    
    logger.info(f"=" * 50)
    logger.info(f"OLLAMA TEST REQUEST")
    logger.info(f"Configured Host: {OLLAMA_HOST}")
    logger.info(f"Configured Model: {OLLAMA_MODEL}")
    logger.info(f"=" * 50)
    
    result = {
        "configured_host": OLLAMA_HOST,
        "configured_model": OLLAMA_MODEL,
        "connectivity_test": None,
        "available_models": [],
        "generate_test": None
    }
    
    # Step 1: Test basic connectivity
    try:
        logger.info(f"Step 1: Testing connectivity to {OLLAMA_HOST}...")
        conn_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        result["connectivity_test"] = {
            "success": conn_response.ok,
            "status_code": conn_response.status_code
        }
        logger.info(f"Connectivity test: {conn_response.status_code}")
        
        if conn_response.ok:
            models_data = conn_response.json()
            result["available_models"] = [m.get('name') for m in models_data.get('models', [])]
            logger.info(f"Available models: {result['available_models']}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection FAILED: {e}")
        result["connectivity_test"] = {"success": False, "error": f"Connection failed: {e}"}
        return jsonify({"success": False, **result}), 500
    except requests.exceptions.Timeout:
        logger.error(f"Connection TIMEOUT")
        result["connectivity_test"] = {"success": False, "error": "Connection timed out"}
        return jsonify({"success": False, **result}), 500
    except Exception as e:
        logger.error(f"Connection ERROR: {e}")
        result["connectivity_test"] = {"success": False, "error": str(e)}
        return jsonify({"success": False, **result}), 500
    
    # Step 2: Test generate endpoint
    try:
        logger.info(f"Step 2: Testing generate with model {OLLAMA_MODEL}...")
        gen_response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        logger.info(f"Generate response status: {gen_response.status_code}")
        
        if gen_response.ok:
            gen_result = gen_response.json()
            result["generate_test"] = {
                "success": True,
                "response": gen_result.get('response', ''),
                "model": gen_result.get('model', OLLAMA_MODEL)
            }
            logger.info(f"Generate SUCCESS")
            return jsonify({"success": True, **result})
        else:
            result["generate_test"] = {
                "success": False,
                "status_code": gen_response.status_code,
                "error": gen_response.text[:500]
            }
            logger.error(f"Generate FAILED: {gen_response.status_code}")
            return jsonify({"success": False, **result}), 500
            
    except requests.exceptions.Timeout:
        logger.error(f"Generate TIMEOUT")
        result["generate_test"] = {"success": False, "error": "Request timed out after 60s"}
        return jsonify({"success": False, **result}), 500
    except Exception as e:
        logger.error(f"Generate ERROR: {e}")
        result["generate_test"] = {"success": False, "error": str(e)}
        return jsonify({"success": False, **result}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Separate endpoint for LLM analysis - called after data is displayed"""
    data = request.json
    query_type = data.get('type', 'team')  # 'team', 'team_comparison', or 'player'
    analysis_data = data.get('data', {})
    
    if not analysis_data:
        return jsonify({"success": False, "error": "No data provided for analysis"}), 400
    
    logger.info(f"Starting async LLM analysis for type: {query_type}")
    
    try:
        result = analyze_with_llm(analysis_data, query_type)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_calls_made": api.request_count
    })


# =============================================================================
# LOGGING & DEBUG ROUTES
# =============================================================================
@app.route('/logs')
def logs_page():
    """Display application logs in browser"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>App Logs - Sports Analyzer</title>
    <style>
        body { background: #1a1a2e; color: #fff; font-family: monospace; padding: 20px; }
        h1 { color: #00d4ff; }
        .controls { margin-bottom: 20px; }
        button { padding: 10px 20px; margin-right: 10px; background: #7b2cbf; color: #fff; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #9b4ddf; }
        .log-entry { padding: 8px; border-bottom: 1px solid #333; }
        .DEBUG { color: #888; }
        .INFO { color: #00d4ff; }
        .WARNING { color: #ffa500; }
        .ERROR { color: #ff4757; }
        .CRITICAL { color: #ff0000; font-weight: bold; }
        #logs { max-height: 80vh; overflow-y: auto; background: #0d0d1a; padding: 15px; border-radius: 8px; }
        .config { background: #222; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .config h3 { margin-top: 0; color: #00d4ff; }
    </style>
</head>
<body>
    <h1>📋 Application Logs</h1>
    
    <div class="config">
        <h3>Current Configuration</h3>
        <div id="config"></div>
    </div>
    
    <div class="controls">
        <button onclick="fetchLogs()">🔄 Refresh</button>
        <button onclick="clearLogs()">🗑️ Clear Logs</button>
        <button onclick="testEndpoints()">🧪 Test All Endpoints</button>
        <label style="margin-left: 20px;">
            <input type="checkbox" id="auto-refresh"> Auto-refresh (5s)
        </label>
    </div>
    
    <div id="logs"></div>
    
    <script>
        let autoRefreshInterval;
        
        async function fetchConfig() {
            try {
                const res = await fetch('/api/debug/config');
                const data = await res.json();
                document.getElementById('config').innerHTML = `
                    <p><strong>SportsData API Key:</strong> ${data.sportsdata_api_key}</p>
                    <p><strong>Ollama Host:</strong> ${data.ollama_host}</p>
                    <p><strong>Ollama Model:</strong> ${data.ollama_model}</p>
                    <p><strong>Database Path:</strong> ${data.database_path}</p>
                    <p><strong>API Calls Made:</strong> ${data.api_calls_made}</p>
                `;
            } catch(e) {
                document.getElementById('config').innerHTML = '<p style="color: #ff4757;">Failed to load config</p>';
            }
        }
        
        async function fetchLogs() {
            try {
                const res = await fetch('/api/logs');
                const logs = await res.json();
                const container = document.getElementById('logs');
                container.innerHTML = logs.map(log => 
                    `<div class="log-entry ${log.level}"><strong>[${log.level}]</strong> ${log.timestamp} - ${log.message}</div>`
                ).reverse().join('');
            } catch(e) {
                document.getElementById('logs').innerHTML = '<p style="color: #ff4757;">Failed to fetch logs: ' + e.message + '</p>';
            }
        }
        
        async function clearLogs() {
            await fetch('/api/logs/clear', { method: 'POST' });
            fetchLogs();
        }
        
        async function testEndpoints() {
            const results = [];
            
            // Test health
            try {
                const res = await fetch('/api/health');
                results.push('✅ /api/health: ' + res.status);
            } catch(e) {
                results.push('❌ /api/health: ' + e.message);
            }
            
            // Test Ollama
            try {
                const res = await fetch('/api/test/ollama', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: 'test'})
                });
                const data = await res.json();
                results.push(data.success ? '✅ Ollama: Connected' : '❌ Ollama: ' + data.error);
            } catch(e) {
                results.push('❌ Ollama: ' + e.message);
            }
            
            // Test SportsData API
            try {
                const res = await fetch('/api/debug/test-sportsdata');
                const data = await res.json();
                results.push(data.success ? '✅ SportsData API: Connected' : '❌ SportsData API: ' + data.error);
            } catch(e) {
                results.push('❌ SportsData API: ' + e.message);
            }
            
            alert(results.join('\\n'));
            fetchLogs();
        }
        
        document.getElementById('auto-refresh').addEventListener('change', function() {
            if (this.checked) {
                autoRefreshInterval = setInterval(fetchLogs, 5000);
            } else {
                clearInterval(autoRefreshInterval);
            }
        });
        
        fetchConfig();
        fetchLogs();
    </script>
</body>
</html>
'''


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get application logs as JSON"""
    return jsonify(log_buffer.get_logs())


@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """Clear the log buffer"""
    log_buffer.clear()
    logger.info("Logs cleared by user")
    return jsonify({"success": True})


@app.route('/api/debug/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        "sportsdata_api_key": SPORTSDATA_API_KEY[:8] + "..." if SPORTSDATA_API_KEY else "NOT SET",
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
        "database_path": DATABASE_PATH,
        "api_calls_made": api.request_count
    })


@app.route('/api/debug/test-sportsdata', methods=['GET'])
def test_sportsdata():
    """Test SportsData.io API connection"""
    try:
        url = f"{SPORTSDATA_ENDPOINTS['NBA']['scores']}/teams"
        logger.info(f"Testing SportsData API: {url}")
        
        response = requests.get(url, params={'key': SPORTSDATA_API_KEY}, timeout=10)
        logger.info(f"SportsData API response: {response.status_code}")
        
        if response.ok:
            teams = response.json()
            return jsonify({
                "success": True,
                "message": f"Connected! Found {len(teams)} NBA teams",
                "sample": teams[0] if teams else None
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Status {response.status_code}: {response.text[:200]}"
            })
    except Exception as e:
        logger.error(f"SportsData API test failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        })


# =============================================================================
# STARTUP
# =============================================================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Sports Betting Analyzer v2.0")
    logger.info(f"SportsData.io API Key: {SPORTSDATA_API_KEY[:8]}...")
    logger.info(f"Ollama Host: {OLLAMA_HOST}")
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info("=" * 60)
    
    init_database()
    app.run(host='0.0.0.0', port=11200, debug=False)
