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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
SPORTSDATA_API_KEY = os.getenv('SPORTSDATA_API_KEY', '1fdd78185de84dc1bd82ff59f254c087')
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://10.254.254.220:11434')
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
    logger.info("Database initialized successfully")


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            self.request_count += 1
            logger.info(f"API Request #{self.request_count}: {url}")
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
        url = f"{base_url}/teams"
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
        """Get betting odds for games"""
        base_url = SPORTSDATA_ENDPOINTS[sport]['odds']
        season = self.get_current_season(sport)
        
        if sport == 'NFL':
            url = f"{base_url}/GameOddsByWeek/{season}/1"  # Current week
        else:
            url = f"{base_url}/GameOddsByDate/{datetime.now().strftime('%Y-%m-%d')}"
        
        odds = self._make_request(url)
        if odds and game_id:
            for game_odds in odds:
                if str(game_odds.get('GameId')) == str(game_id):
                    return game_odds
        return odds
    
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
        
        if sport == 'NFL':
            url = f"{base_url}/PlayerGameStatsBySeason/{season}/{player_id}/all"
        elif sport == 'NBA':
            url = f"{base_url}/PlayerGameStatsBySeason/{season}/{player_id}"
        elif sport == 'NHL':
            url = f"{base_url}/PlayerGameStatsBySeason/{season}/{player_id}"
        
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
    """Send data to Ollama for analysis"""
    try:
        if query_type == 'team':
            prompt = f"""You are a sports betting analyst. Analyze the following team data and betting lines.
            
Data: {json.dumps(data, indent=2)}

Based on this data, provide:
1. Your prediction for the team's next game (win/loss)
2. Recommended bets (spread, moneyline, over/under) with confidence levels
3. Key factors influencing your prediction
4. Any trends you notice from recent games

Be concise but thorough. Format your response clearly with sections."""

        elif query_type == 'team_comparison':
            prompt = f"""You are a sports betting analyst. Analyze this matchup between two teams.
            
Data: {json.dumps(data, indent=2)}

Based on this data, provide:
1. Predicted winner and score prediction
2. Spread recommendation (which team to take)
3. Over/Under recommendation
4. Moneyline value analysis
5. Key factors and trends supporting your picks

Be specific with numbers and confidence levels."""

        elif query_type == 'player':
            prompt = f"""You are a sports betting analyst specializing in player props.
            
Player Data: {json.dumps(data, indent=2)}

Based on this player's recent performance, provide:
1. Analysis of their recent form and trends
2. Recommended player prop bets with lines
3. Props to avoid
4. Confidence levels for each recommendation

Focus on statistical trends and matchup factors."""

        else:
            prompt = f"Analyze this sports data: {json.dumps(data)}"
        
        # Call Ollama
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            },
            timeout=120
        )
        
        if response.ok:
            result = response.json()
            return {
                "success": True,
                "analysis": result.get('response', 'No response generated')
            }
        else:
            return {
                "success": False,
                "error": f"Ollama returned status {response.status_code}"
            }
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "LLM request timed out"}
    except Exception as e:
        logger.error(f"LLM analysis error: {e}")
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
        # Check cache first
        cached_games = get_cached_team_games(sport, team_name, 10)
        
        # Determine how many new games to fetch
        games_needed = 10 - len(cached_games)
        
        if games_needed > 0 or len(cached_games) == 0:
            # Fetch from API
            completed_games = api.get_completed_games(sport, team_name, 10)
            
            # Cache new games
            for game in completed_games:
                cache_game(sport, game)
            
            games_data = completed_games
        else:
            # Use cached data
            games_data = [json.loads(g['game_data']) for g in cached_games]
        
        # Get upcoming games and odds
        upcoming = api.get_upcoming_games(sport, team_name, 3)
        
        # Get odds for upcoming games
        odds_data = []
        for game in upcoming[:1]:  # Just get odds for next game to save API calls
            game_id = game.get('GameID', game.get('ScoreID'))
            if game_id:
                odds = api.get_game_odds(sport, game_id)
                if odds:
                    odds_data.append(odds)
        
        result = {
            "team": team_name.upper(),
            "sport": sport,
            "recent_games": games_data[:10],
            "upcoming_games": upcoming,
            "betting_lines": odds_data,
            "games_from_cache": len(cached_games),
            "api_calls_made": api.request_count
        }
        
        # Get LLM analysis
        llm_result = analyze_with_llm(result, 'team')
        result['llm_analysis'] = llm_result
        
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
            "api_calls_made": api.request_count
        }
        
        # Get LLM analysis
        llm_result = analyze_with_llm(result, 'team_comparison')
        result['llm_analysis'] = llm_result
        
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
        
        # Get LLM analysis
        llm_result = analyze_with_llm(result, 'player')
        result['llm_analysis'] = llm_result
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Player search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/api/test/ollama', methods=['POST'])
def test_ollama():
    """Test Ollama connection"""
    data = request.json
    prompt = data.get('prompt', 'Hello, respond with a short greeting.')
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.ok:
            result = response.json()
            return jsonify({
                "success": True,
                "response": result.get('response', ''),
                "model": OLLAMA_MODEL
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Status {response.status_code}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "api_calls_made": api.request_count
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
