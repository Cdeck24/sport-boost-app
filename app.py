import streamlit as st
import requests
import datetime
import string
import pandas as pd
import concurrent.futures
import pulp
import io
import unicodedata

# --- ⬇️ CONFIGURATION LINKS ⬇️ ---
# 1. LIVE BOOSTS CSV (Your Google Sheet with saved multipliers)
SAVED_BOOSTS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1721203281&single=true&output=csv"

# 2. PROJECTION SOURCES (Updated Google Sheet Links)
SPORT_PROJECTION_URLS = {
    "nba": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=0&single=true&output=csv", 
    "nfl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1180552482&single=true&output=csv",
    "nhl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=401621588&single=true&output=csv"
}
# ---------------------------------------------------

# --- Page Configuration ---
st.set_page_config(page_title="Player Boost & Optimizer", layout="wide")

st.title("🏀 🏒 Player Boost & Lineup Optimizer")
st.markdown("""
This tool fetches live **Boost Multipliers** and allows you to merge them with 
**Fantasy Projections** to find the highest-scoring lineups using **Slot-Based Optimization**.
""")

# --- Helper Functions ---
def get_fantasy_day():
    """Returns the current date in US Eastern Time (approximate)."""
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    us_time = utc_now - datetime.timedelta(hours=5)
    return us_time.date()

def normalize_name(name):
    """Robust normalization for names with accent removal."""
    n = str(name).lower()
    # Normalize unicode characters (e.g. Doncic vs Dončić)
    try:
        n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode('utf-8')
    except:
        pass
    
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v', ' jr.', ' sr.']
    for suffix in suffixes:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
    return "".join(c for c in n if c.isalnum())

def normalize_position(pos):
    """Normalizes position strings."""
    if not pos or pd.isna(pos):
        return "UNKNOWN"
    p = str(pos).upper().strip()
    if "QUARTER" in p or p == "QB": return "QB"
    if "WIDE" in p or "RECEIVER" in p or p == "WR": return "WR"
    if "RUNNING" in p or "BACK" in p or p == "RB" or p == "HB": return "RB"
    if "TIGHT" in p or p == "TE": return "TE"
    return p

def find_col(columns, keywords):
    """Finds the first column that matches any keyword in the list (Case Insensitive)."""
    for col in columns:
        col_lower = str(col).lower()
        if any(str(k).lower() in col_lower for k in keywords):
            return col
    return None

def calculate_nba_custom_rating(row, mapping):
    """Calculates player rating based on the user-provided efficiency formula."""
    stats = {}
    for key, col_name in mapping.items():
        try:
            val = float(row.get(col_name, 0.0))
            if pd.isna(val): val = 0.0
            stats[key] = val
        except:
            stats[key] = 0.0

    rating = 0.0
    
    # --- 1. Scoring & Efficiency ---
    two_pm = stats['fgm'] - stats['3pm']
    missed_fg = stats['fga'] - stats['fgm']
    missed_ft = stats['fta'] - stats['ftm']

    rating += two_pm * 0.22
    rating += stats['3pm'] * 0.35
    rating += stats['ftm'] * 0.10
    
    rating -= missed_fg * 0.08
    rating -= missed_ft * 0.05

    # --- 2. Playmaking & Possession ---
    rating += stats['reb'] * 0.11
    rating += stats['ast'] * 0.15
    rating -= stats['to']  * 0.20

    # --- 3. Defense ---
    rating += stats['stl'] * 0.20
    rating += stats['blk'] * 0.18

    return round(rating, 2)

def fetch_data_for_sport(sport, target_date):
    """Fetches player data from API using the selected date."""
    letters = string.ascii_uppercase
    session = requests.Session()
    sport_data = []
    seen_players = set() 

    # Strict Date Strategy: Only check the specific date requested
    target_dates = [target_date]
    active_date_str = str(target_date)
    
    # Fetch Full Alphabet
    for letter in letters:
        query = letter
        url = (
            f"https://api.real.vg/players/sport/{sport}/search"
            f"?day={active_date_str}&includeNoOneOption=false"
            f"&query={query}&searchType=ratingLineup"
        )
        try:
            r = session.get(url, timeout=5)
            if r.status_code != 200: continue
            data = r.json()
            players = data.get("players", [])
            if not players: continue

            for player in players:
                raw_injury = player.get('injuryStatus')
                injury_status = str(raw_injury).strip().upper() if raw_injury else ""
                
                # Check for O/OUT
                if injury_status in ['O', 'OUT', 'IR', 'INJ']: 
                    continue

                position = player.get('position', 'Unknown')
                if sport.lower() == 'nhl' and position == 'G':
                    continue

                full_name = f"{player['firstName']} {player['lastName']}"
                
                if full_name in seen_players:
                    continue
                seen_players.add(full_name)

                # --- DEFAULT BOOST TO 0.0 ---
                boost_value = 0.0 
                details = player.get("details")
                
                # Try to extract boost if it exists
                if details and isinstance(details, list) and len(details) > 0 and "text" in details[0]:
                    text = details[0]["text"]
                    boost_str = text.replace("x", "").replace("+", "").strip()
                    try:
                        boost_value = float(boost_str) 
                    except ValueError:
                        pass 
                
                sport_data.append({
                    "Sport": sport.upper(),
                    "Player Name": full_name,
                    "Position": position,
                    "Boost": boost_value,
                    "Date": active_date_str,
                    "Injury": injury_status
                })
        except requests.RequestException:
            continue
            
    return sport_data

def load_projections_from_url(url):
    """Smart Fetcher: Tries to read URL as CSV first, then as HTML tables."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64;
