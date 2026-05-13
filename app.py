import streamlit as st
import requests
from requests.adapters import HTTPAdapter
import datetime
import string
import pandas as pd
import concurrent.futures
import pulp
import io
import unicodedata
import time
import re

# new library
from hashids import Hashids

# auth constants
REAL_API_BASE = 'https://web.realsports.io'
REAL_VERSION = '27'
REAL_REFERER = 'https://realsports.io/'
DEFAULT_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
DEFAULT_SEC_CH_UA = '"Chromium";v="125", "Not.A/Brand";v="24", "Google Chrome";v="125"'
DEVICE_NAME = 'Chrome on Windows'
# change this as well
DEVICE_UUID = '0e497d76-7bd5-4cf5-b63c-f194d1d4cbcf'
# you are going to find this yourself looks like fgOD12!sdg49!random-random-random , found in headers in inspect network
REAL_AUTH_TOKEN = 'xnr5VpW3!ApZk8L2E!4fe6e26f-949f-4936-ae3e-16384878932f'

# PRE-COMPILED REGEX FOR PERFORMANCE
BOOST_REGEX = re.compile(r"(\d+(\.\d+)?)")

# auth functions
def build_headers(token):
    return {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'DNT': '1',
        'Origin': 'https://realsports.io',
        'Referer': REAL_REFERER,
        'User-Agent': DEFAULT_USER_AGENT,
        'sec-ch-ua': DEFAULT_SEC_CH_UA,
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'real-auth-info': REAL_AUTH_TOKEN,
        'real-device-name': DEVICE_NAME,
        'real-device-type': 'desktop_web',
        'real-device-uuid': DEVICE_UUID,
        'real-request-token': token,
        'real-version': REAL_VERSION
    }

def generate_request_token():
    # Configuration
    salt = 'realwebapp'
    min_length = 16
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    seps = 'cfhistuCFHISTU'

    # shuffle
    def shuffle(alphabet_chars, salt_chars):
        if len(salt_chars) == 0:
            return alphabet_chars

        transformed = list(alphabet_chars)
        v = 0
        p = 0

        for i in range(len(transformed) - 1, 0, -1):
            v %= len(salt_chars)
            integer = ord(salt_chars[v])
            p += integer
            j = (integer + v + p) % i
            transformed[i], transformed[j] = transformed[j], transformed[i]
            v += 1

        return transformed

    #convert number to alphabet representation
    def to_alphabet(num, alphabet_chars):
        result = []
        alphabet_len = len(alphabet_chars)

        while True:
            result.insert(0, alphabet_chars[num % alphabet_len])
            num = num // alphabet_len
            if num == 0:
                break

        return result

    # Initialize alphabet and seps
    salt_chars = list(salt)
    alphabet_chars = list(alphabet)
    seps_chars = list(seps)

    # Get unique alphabet
    unique_alphabet = []
    seen = set()
    for char in alphabet_chars:
        if char not in seen:
            unique_alphabet.append(char)
            seen.add(char)

    # Remove seps from alphabet
    alphabet_list = [c for c in unique_alphabet if c not in seps_chars]

    # Filter seps
    filtered_seps = [c for c in seps_chars if c in unique_alphabet]
    seps_list = shuffle(filtered_seps, salt_chars)

    # Adjust seps and alphabet
    if len(seps_list) == 0 or len(alphabet_list) / len(seps_list) > 3.5:
        seps_length = max(2, (len(alphabet_list) + 3) // 4)

        if seps_length > len(seps_list):
            diff = seps_length - len(seps_list)
            seps_list.extend(alphabet_list[:diff])
            alphabet_list = alphabet_list[diff:]

    alphabet_list = shuffle(alphabet_list, salt_chars)

    # Setup guards
    guard_count = max(1, len(alphabet_list) // 12)

    if len(alphabet_list) < 3:
        guards = seps_list[:guard_count]
        seps_list = seps_list[guard_count:]
    else:
        guards = alphabet_list[:guard_count]
        alphabet_list = alphabet_list[guard_count:]

    # Encode timestamp
    timestamp_ms = int(time.time() * 1000)
    numbers = [timestamp_ms]

    alphabet_working = list(alphabet_list)

    # Calculate numbersIdInt
    numbers_id_int = 0
    for i, number in enumerate(numbers):
        numbers_id_int += number % (i + 100)

    # Lottery character
    ret = [alphabet_working[numbers_id_int % len(alphabet_working)]]
    lottery = list(ret)

    # Encode each number
    for i, number in enumerate(numbers):
        buffer = lottery + salt_chars + alphabet_working
        alphabet_working = shuffle(alphabet_working, buffer)
        last = to_alphabet(number, alphabet_working)
        ret.extend(last)

        if i + 1 < len(numbers):
            char_code = ord(last[0])
            extra_number = number % (char_code + i)
            ret.append(seps_list[extra_number % len(seps_list)])

    # Ensure minimum length
    if len(ret) < min_length:
        prefix_guard_index = (numbers_id_int + ord(ret[0])) % len(guards)
        ret.insert(0, guards[prefix_guard_index])

        if len(ret) < min_length:
            suffix_guard_index = (numbers_id_int + ord(ret[2])) % len(guards)
            ret.append(guards[suffix_guard_index])

    # Extend to minimum length with shuffling
    half_length = len(alphabet_working) // 2
    while len(ret) < min_length:
        alphabet_working = shuffle(alphabet_working, alphabet_working)
        ret = alphabet_working[half_length:] + ret + alphabet_working[:half_length]

        excess = len(ret) - min_length
        if excess > 0:
            half_of_excess = excess // 2
            ret = ret[half_of_excess:half_of_excess + min_length]

    return ''.join(ret)

token = generate_request_token()
HEADERS = build_headers(token)

# --- ⬇️ CONFIGURATION ⬇️ ---
# PROJECTION SOURCES (Google Sheet Links)
SPORT_PROJECTION_URLS = {
    "nba": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=0&single=true&output=csv", 
    "wnba": "", # Empty to force "Boosts Only" display for WNBA unless provided
    "nfl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1180552482&single=true&output=csv",
    "nhl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=401621588&single=true&output=csv",
    "ncaam": "", # Empty to force "Boosts Only" display for CBB
    "mlb": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=44331943&single=true&output=csv",
    "golf": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1539073771&single=true&output=csv"
}

NHL_LINES_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=15374641&single=true&output=csv"
MLB_PITCHERS_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1157919355&single=true&output=csv"

# ALIAS MAPPING: Link Nicknames from API to Legal Names in CSV 
# Keys and values must be strictly lowercase with NO spaces or punctuation
PLAYER_NAME_MAPPINGS = {
    "ggjackson": "gregoryjackson",
    "camthomas": "cameronthomas",
    "mohamedbamba": "mobamba",
    "nicclaxton": "nicolasclaxton",
    "pjtucker": "pjotucker",
    "timhardaway": "timhardawayjr",
    "kellyoubre": "kellyoubrejr",
    "michaelporter": "michaelporterjr",
    "mattboldy": "matthewboldy", 
    # Add any missing mappings you discover below!
}
# ---------------------------------------------------

# --- Page Configuration ---
st.set_page_config(page_title="Player Boost & Optimizer", layout="wide")

st.title("🏀 🏒 ⚾ ⛳ Player Boost & Lineup Optimizer")
st.markdown("""
This tool fetches live **Boost Multipliers** from the API and allows you to merge them with 
**Fantasy Projections** to find the highest-scoring lineups using **Slot-Based Optimization**.
""")

# --- GLOBAL STORAGE (Persists across sessions/refresh) ---
@st.cache_resource
class GlobalBoostStore:
    def __init__(self):
        self.data = pd.DataFrame(columns=['Sport', 'Player Name', 'Position', 'Boost', 'Date', 'Injury'])
    
    def update(self, new_df):
        self.data = new_df
        
    def get(self):
        return self.data

# Instantiate the global store
boost_store = GlobalBoostStore()

# --- Helper Functions ---
def get_fantasy_day():
    """Returns the current date in US Eastern Time (approximate)."""
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    us_time = utc_now - datetime.timedelta(hours=5)
    return us_time.date()

def normalize_name(name):
    """Robust normalization for names with accent removal and nickname mapping."""
    if not isinstance(name, str):
        name = str(name)
    n = name.lower().strip()
    
    # Handle "Last, First" automatically if a comma exists (for Golf sheets, etc)
    if ',' in n and n.count(',') == 1:
        parts = n.split(',')
        n = f"{parts[1].strip()} {parts[0].strip()}"
        
    try:
        n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode('utf-8')
    except Exception:
        pass
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v', ' jr.', ' sr.']
    for suffix in suffixes:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
            
    # Strip spaces and punctuation
    n = "".join(c for c in n if c.isalnum())
    
    # Apply mapping dictionary
    return PLAYER_NAME_MAPPINGS.get(n, n)

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
    """Finds the first column that matches any keyword exactly, then falls back to substring."""
    # Exact match first to avoid pulling 'earnedRuns' when looking for 'runs'
    for col in columns:
        col_lower = str(col).lower().strip()
        if any(str(k).lower().strip() == col_lower for k in keywords):
            return col
            
    # Substring match fallback
    for col in columns:
        col_lower = str(col).lower()
        if any(str(k).lower() in col_lower for k in keywords):
            return col
            
    return None

def parse_odds_to_prob(odds_str):
    """Safely parses positive american, negative american, fractional, and text odds to implied probability."""
    if pd.isna(odds_str): 
        return 0.0
    s = str(odds_str).strip().lower()
    if not s: return 0.0
    if s in ['even', 'ev']: return 0.5
    
    # Check for fractional (e.g., "17-1" or "17/1")
    if '-' in s and not s.startswith('-'):
        parts = s.split('-')
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                return den / (num + den)
            except: return 0.0
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                return den / (num + den)
            except: return 0.0
            
    # American Odds processing
    s = s.replace('+', '')
    try:
        val = float(s)
        if val < 0:
            return abs(val) / (abs(val) + 100.0)
        else:
            return 100.0 / (val + 100.0)
    except:
        return 0.0

def standardize_boost_columns(df):
    """Ensures the DataFrame has the standard column names used by the app."""
    col_map = {}
    
    name_col = find_col(df.columns, ["player name", "player", "name"])
    if name_col: col_map[name_col] = "Player Name"
    
    boost_col = find_col(df.columns, ["boost value", "boost", "multiplier"])
    if boost_col: col_map[boost_col] = "Boost"
    
    sport_col = find_col(df.columns, ["sport", "league"])
    if sport_col: col_map[sport_col] = "Sport"
    
    pos_col = find_col(df.columns, ["position", "pos"])
    if pos_col: col_map[pos_col] = "Position"
    
    date_col = find_col(df.columns, ["date", "day"])
    if date_col: col_map[date_col] = "Date"
    
    inj_col = find_col(df.columns, ["injury", "status"])
    if inj_col: col_map[inj_col] = "Injury"

    df = df.rename(columns=col_map)
    
    required_cols = ["Player Name", "Boost", "Sport", "Position", "Date", "Injury"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = None
            
    return df

def calculate_nba_custom_rating(row, mapping):
    """Calculates player rating based on the user-provided efficiency formula."""
    stats = {}
    for key, col_name in mapping.items():
        if col_name:
            try:
                val = float(row.get(col_name, 0.0))
                if pd.isna(val): val = 0.0
                stats[key] = val
            except:
                stats[key] = 0.0
        else:
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

def calculate_cbb_custom_rating(row, mapping):
    """Calculates CBB player rating based on specific efficiency weights."""
    stats = {}
    for key, col_name in mapping.items():
        if col_name:
            try:
                val = float(row.get(col_name, 0.0))
                if pd.isna(val): val = 0.0
                stats[key] = val
            except:
                stats[key] = 0.0
        else:
            stats[key] = 0.0

    # --- Scaling Logic ---
    scaling_factor = 1.0
    col_proj = mapping.get('proj_min')
    col_avg = mapping.get('avg_min')
    
    if col_proj and col_avg:
        try:
            p_min = float(row.get(col_proj, 0))
            a_min = float(row.get(col_avg, 0))
            if a_min > 0:
                scaling_factor = p_min / a_min
        except:
            pass # Keep 1.0 on error
            
    # Apply Scaling Factor to all counting stats
    if scaling_factor != 1.0:
        for k in stats:
            stats[k] = stats[k] * scaling_factor

    rating = 0.0

    # Derive Makes and Misses
    two_pm = stats['fgm'] - stats['3pm']
    missed_fg = stats['fga'] - stats['fgm']
    missed_ft = stats['fta'] - stats['ftm']

    # Scoring Weights
    rating += two_pm * 0.57
    rating += stats['3pm'] * 0.77
    rating += stats['ftm'] * 0.15

    # Efficiency Penalties
    rating -= missed_fg * 0.10
    rating -= missed_ft * 0.05

    # Stats Weights
    rating += stats['reb'] * 0.14
    rating += stats['ast'] * 0.18
    rating += stats['stl'] * 0.25
    rating += stats['blk'] * 0.29

    # Turnover Penalty
    rating -= stats['to'] * 0.24

    return round(rating, 2)

def calculate_nhl_custom_rating(row, mapping):
    """
    Calculates NHL Player Rating using standard box score stats.
    """
    stats = {}
    for key, col_name in mapping.items():
        if col_name:
            try:
                val = float(row.get(col_name, 0.0))
                if pd.isna(val): val = 0.0
                stats[key] = val
            except:
                stats[key] = 0.0
        else:
            stats[key] = 0.0

    rating = 0.0
    
    # Blended average for Points (accounting for standard and game-winning assists)
    rating += stats.get('points', 0) * 1.25
    
    # Goal Premium to reach ~2.40 total value (1.25 point + 0.99 premium + 0.16 shot)
    rating += stats.get('goals', 0) * 0.99
    
    # Standard values
    rating += stats.get('shots', 0) * 0.16
    rating += stats.get('blockedShots', 0) * 0.32

    return round(rating, 2)

def calculate_mlb_custom_rating(row, mapping):
    """
    Calculates MLB Rating automatically detecting Batters vs Pitchers.
    Calculates both scores independently and sums them for two-way players.
    """
    stats = {}
    for key, col_name in mapping.items():
        if col_name:
            try:
                val = float(row.get(col_name, 0.0))
                if pd.isna(val): val = 0.0
                stats[key] = val
            except:
                stats[key] = 0.0
        else:
            stats[key] = 0.0

    hitting_score = 0.0
    pitching_score = 0.0
    
    # --- BATTER LOGIC ---
    pa = stats.get('plateAppearances', 0)
    # Batter check: Must have PAs or Hitting stats
    if pa > 0 or stats.get('hits', 0) > 0:
        outs = pa - stats.get('hits', 0) - stats.get('walks', 0)
        total_bases = (
            stats.get('singles', 0) * 1 +
            stats.get('doubles', 0) * 2 +
            stats.get('triples', 0) * 3 +
            stats.get('homeRuns', 0) * 4
        )
        
        # PENALTIES (Adjusted to absorb LOB penalty)
        hitting_score += outs * -0.13
        hitting_score += stats.get('strikeouts', 0) * -0.02
        hitting_score += stats.get('caughtStealing', 0) * -0.47
        
        # OFFENSE
        hitting_score += stats.get('hits', 0) * 0.28
        hitting_score += total_bases * 0.25
        hitting_score += stats.get('runs', 0) * 0.70
        hitting_score += stats.get('runsBattedIn', 0) * 0.65
        
        # BASERUNNING & DISCIPLINE
        hitting_score += stats.get('stolenBases', 0) * 0.47
        hitting_score += stats.get('walks', 0) * 0.15

    # --- PITCHER LOGIC ---
    ip = stats.get('inningsPitched', 0.0)
    gs = stats.get('gamesStarted', 0.0)
    
    # Pitcher check: Must have IP or Pitching stats AND be a starter
    if (ip > 0 or stats.get('saves', 0) > 0 or stats.get('wins', 0) > 0) and gs >= 1:
        # Corrected Outs Logic: Treats IP as a mathematical average of innings.
        # 4.8 IP results in 14.4 outs (4.8 * 3.0), awarding partial credit for mathematical averages.
        p_outs = ip * 3.0
        
        pitching_score += p_outs * 0.38
        pitching_score += stats.get('strikeouts_pitching', 0) * 0.07 
        
        # Penalties use Allowed columns to avoid conflict with Batter columns
        pitching_score += stats.get('walksAllowed', 0) * -0.30
        pitching_score += stats.get('earnedRuns', 0) * -0.30
        pitching_score += stats.get('losses', 0) * -0.30

        pitching_score += stats.get('hitsAllowed', 0) * -0.38
        pitching_score += stats.get('homeRunsAllowed', 0) * -0.82 

        pitching_score += stats.get('wins', 0) * 0.30
        pitching_score += stats.get('saves', 0) * 0.05

    return round(hitting_score + pitching_score, 2)


def calculate_golf_custom_rating(row, mapping):
    """Calculates an expected value score based on implied probabilities of finishing positions."""
    prob_win = parse_odds_to_prob(row.get(mapping.get('to_win')))
    prob_t5 = parse_odds_to_prob(row.get(mapping.get('top_5')))
    prob_t10 = parse_odds_to_prob(row.get(mapping.get('top_10')))
    prob_t20 = parse_odds_to_prob(row.get(mapping.get('top_20')))
    prob_t40 = parse_odds_to_prob(row.get(mapping.get('top_40')))
    prob_cut = parse_odds_to_prob(row.get(mapping.get('make_cut')))

    # Weighted points based on probability to create a solid projection proxy score
    score = (
        (prob_win * 50) +
        (prob_t5 * 30) +
        (prob_t10 * 20) +
        (prob_t20 * 15) +
        (prob_t40 * 10) +
        (prob_cut * 20) 
    )
    return round(score, 2)


def fetch_letter(session, sport, date_str, query_str):
    """Helper to fetch a single query for a specific date using standard URL encoding."""
    url = f"https://web.realsports.io/players/sport/{sport}/search"
    # Passed securely to prevent spaces and special characters from breaking the URL
    params = {
        "day": date_str,
        "includeNoOneOption": "false",
        "query": query_str,
        "searchType": "ratingLineup"
    }
    
    for attempt in range(4):
        try:
            token = generate_request_token()
            headers = build_headers(token)
            
            r = session.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json().get("players", [])
            elif r.status_code == 429: # Rate limited
                time.sleep(1.5 * (attempt + 1)) # Exponential backoff
            else:
                break
        except:
            time.sleep(1)
            
    return []

def fetch_data_for_sport(sport, target_date):
    """Fetches player data from API using a smart 2-phase letter expansion."""
    session = requests.Session()
    # OPTIMIZATION: Mount a custom adapter to allow more concurrent connections without blocking
    adapter = HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=2)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    
    sport_data = []
    seen_players = set()
    active_date_str = str(target_date)

    single_letters = list(string.ascii_uppercase)
    special_chars = ['Š', 'Ć', 'Č', 'Ž', 'Đ', 'Ö', 'Ä', 'Ü', 'Å', 'Ø']
    base_queries = single_letters + special_chars
    
    def process_players(players):
        for player in players:
            full_name = f"{player.get('firstName', '')} {player.get('lastName', '')}".strip()
            if full_name in seen_players:
                continue
            seen_players.add(full_name)
            
            raw_injury = player.get('injuryStatus')
            injury_status = str(raw_injury).strip().upper() if raw_injury else ""

            position = player.get('position', 'Unknown')
            if sport.lower() == 'nhl' and position == 'G':
                continue
            
            boost_value = 0.0 
            
            # Primary fetch from direct 'multiplierBonus' object provided by API
            if "multiplierBonus" in player and player["multiplierBonus"] is not None:
                try:
                    boost_value = float(player["multiplierBonus"])
                except ValueError:
                    pass

            # Fallback to text parsing if multiplierBonus is missing or 0
            if boost_value == 0.0:
                details = player.get("details")
                if details and isinstance(details, list) and len(details) > 0 and "text" in details[0]:
                    text = str(details[0]["text"])
                    # Use globally compiled regex for performance
                    match = BOOST_REGEX.search(text)
                    if match:
                        boost_value = float(match.group(1))
            
            sport_data.append({
                "Sport": sport.upper(),
                "Player Name": full_name,
                "Position": position,
                "Boost": boost_value,
                "Date": active_date_str,
                "Injury": injury_status
            })

    # PHASE 1: Base Letters 
    # OPTIMIZATION: Bumped workers to 20 since connection pooling is now enabled
    letters_to_expand = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_req = {executor.submit(fetch_letter, session, sport, active_date_str, q): q for q in base_queries}
        
        for future in concurrent.futures.as_completed(future_to_req):
            q = future_to_req[future]
            try:
                players = future.result()
                if not players: continue
                process_players(players)
                
                # Smart Expansion trigger: if a single letter returns 10+ players, expand it.
                if len(players) >= 10 and len(q) == 1 and q in string.ascii_uppercase:
                    letters_to_expand.append(q)
            except:
                continue
                
    # PHASE 2: Expanded Queries (Only run on letters that hit the cap)
    if letters_to_expand:
        expanded_queries = [a + b for a in letters_to_expand for b in string.ascii_uppercase]
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_to_req_exp = {executor.submit(fetch_letter, session, sport, active_date_str, q): q for q in expanded_queries}
            
            for future in concurrent.futures.as_completed(future_to_req_exp):
                try:
                    players = future.result()
                    if players:
                        process_players(players)
                except:
                    continue
                
    return sport_data, active_date_str

def load_projections_from_url(url):
    """Smart Fetcher: Tries to read URL as CSV first, then as HTML tables."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        content = response.content
        try:
            return pd.read_csv(io.BytesIO(content)), "CSV"
        except:
            pass
        try:
            tables = pd.read_html(io.BytesIO(content))
            if tables:
                largest_table = max(tables, key=len)
                if len(largest_table) > 5:
                    return largest_table, "HTML"
        except:
            pass
        return None, "Could not identify CSV or HTML Table data."
    except Exception as e:
        return None, str(e)

def run_optimization(df, num_lineups=1, locked_slots=None, sport="", mlb_rule="Flexible (Highest Projected)"):
    """Runs Assignment Problem solver with players locked into specific slots."""
    if locked_slots is None:
        locked_slots = {}
        
    SLOT_ADDERS = [2.0, 1.8, 1.6, 1.4, 1.2]
    NUM_SLOTS = len(SLOT_ADDERS)
    
    if len(df) < NUM_SLOTS:
        return None

    # Sort by Optimization Score
    df = df.sort_values('Optimization Score', ascending=False)
    df = df.drop_duplicates(subset=['Player Name'], keep='first').reset_index(drop=True)
    
    if len(df) < NUM_SLOTS:
        return None

    prob = pulp.LpProblem("SlotOptimizer", pulp.LpMaximize)
    player_indices = list(df.index)
    slot_indices = list(range(NUM_SLOTS))
    
    x = pulp.LpVariable.dicts("x", (player_indices, slot_indices), cat="Binary")
    y = pulp.LpVariable.dicts("y", player_indices, cat="Binary")
    
    obj_terms = []
    for i in player_indices:
        for j in slot_indices:
            raw_boost = df.loc[i, 'Boost']
            adj_proj = df.loc[i, 'Adjusted Projection'] 
            slot_add = SLOT_ADDERS[j]
            points = (raw_boost + slot_add) * adj_proj
            obj_terms.append(points * x[i][j])
            
    prob += pulp.lpSum(obj_terms)
    
    # Each slot gets exactly 1 player
    for j in slot_indices:
        prob += pulp.lpSum([x[i][j] for i in player_indices]) == 1
        
    # Link player selection to slot selections
    for i in player_indices:
        prob += pulp.lpSum([x[i][j] for j in slot_indices]) == y[i]
        
    # Exactly NUM_SLOTS players selected total
    prob += pulp.lpSum([y[i] for i in player_indices]) == NUM_SLOTS

    # MLB Constraint: Dynamic based on user selection
    if sport == 'mlb' and 'Is_Pitcher' in df.columns and mlb_rule != "Flexible (Highest Projected)":
        pitcher_idx = [i for i in player_indices if df.loc[i, 'Is_Pitcher']]
        batter_idx = [i for i in player_indices if not df.loc[i, 'Is_Pitcher']]
        
        p_req, b_req = None, None
        if mlb_rule == "3 Pitchers / 2 Batters":
            p_req, b_req = 3, 2
        elif mlb_rule == "2 Pitchers / 3 Batters":
            p_req, b_req = 2, 3
        elif mlb_rule == "4 Pitchers / 1 Batter":
            p_req, b_req = 4, 1
        elif mlb_rule == "1 Pitcher / 4 Batters":
            p_req, b_req = 1, 4
            
        if p_req is not None and b_req is not None:
            prob += pulp.lpSum([y[i] for i in pitcher_idx]) == p_req
            prob += pulp.lpSum([y[i] for i in batter_idx]) == b_req

    # LOCK PLAYERS TO SPECIFIC SLOTS CONSTRAINT
    if locked_slots:
        for s_idx, p_name in locked_slots.items():
            # Find index of the specific player in our dataframe
            p_matches = df[df['Player Name'] == p_name].index.tolist()
            if p_matches:
                p_idx = p_matches[0]
                # Force this specific player into this specific slot
                prob += x[p_idx][s_idx] == 1

    generated_lineups = []
    for n in range(num_lineups):
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] == "Optimal":
            lineup_data = []
            selected_player_indices = []
            for j in slot_indices:
                for i in player_indices:
                    if x[i][j].varValue == 1.0:
                        selected_player_indices.append(i)
                        p_name = df.loc[i, "Player Name"]
                        p_pos = df.loc[i, "Position"]
                        p_proj_orig = df.loc[i, "Projection"]
                        p_boost = df.loc[i, "Boost"]
                        p_injury = df.loc[i, "Injury"]
                        slot_add = SLOT_ADDERS[j]
                        eff_boost = p_boost + slot_add
                        final_pts = eff_boost * p_proj_orig 
                        lineup_data.append({
                            "Slot": j + 1,
                            "Slot Bonus": f"+{slot_add}x",
                            "Player Name": p_name,
                            "Boost": p_boost,
                            "Eff. Boost": f"{eff_boost:.2f}x",
                            "Injury": p_injury,
                            "Projection": p_proj_orig,
                            "Optimization Score": final_pts
                        })
            lineup_df = pd.DataFrame(lineup_data).sort_values(by="Slot")
            generated_lineups.append(lineup_df)
            
            # Ensure different lineups on subsequent loops
            prob += pulp.lpSum([y[i] for i in selected_player_indices]) <= NUM_SLOTS - 1
        else:
            break
    return generated_lineups

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Boost Data")
    selected_sport = st.selectbox("Select League", ["nba", "wnba", "nhl", "nfl", "ncaam", "mlb", "golf"], index=0)
    
    # --- AUTO-CLEAR STALE PROJECTIONS ---
    if 'current_sport' not in st.session_state:
        st.session_state.current_sport = selected_sport
    
    if st.session_state.current_sport != selected_sport:
        st.session_state.proj_df = None
        st.session_state.current_sport = selected_sport
        st.rerun()

    fetch_btn = st.button("Fetch Live Boosts")

    st.header("2. Projections Source")
    input_options = ["Upload CSV", "Paste Text"]
    if any(SPORT_PROJECTION_URLS.values()):
        input_options.insert(0, "Use Global/Public Projections")
    
    input_method = st.radio("Source", input_options)
    
    nhl_proj_source = "Custom Formula (Original CSV)"
    if selected_sport == "nhl" and input_method == "Use Global/Public Projections":
        st.write("---")
        nhl_proj_source = st.radio("NHL Projection Strategy", ["Custom Formula (Original CSV)", "Fantasy Points (Lines CSV)"])
    
    uploaded_file = None
    pasted_text = None
    current_proj_url = None
    
    if input_method == "Use Global/Public Projections":
        sport_key = selected_sport.lower()
        url = SPORT_PROJECTION_URLS.get(sport_key)
        if url:
            st.success(f"✅ URL Configured for {sport_key.upper()}")
            st.caption(f"Source: {url[:40]}...")
            current_proj_url = url
        elif sport_key in ["ncaam", "golf", "wnba"]:
             st.info(f"ℹ️ No auto-projections for {sport_key.upper()}. Fetching boosts only unless you upload CSV or paste text.")
        else:
            st.warning(f"⚠️ No URL configured for {sport_key.upper()}.")
    elif input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    elif input_method == "Paste Text":
        pasted_text = st.text_area("Paste Data Here", height=150)

    st.header("3. Optimization Settings")
    wr_rb_bonus = 1.0
    qb_penalty = 1.0
    num_lineups = st.slider("Number of Lineups", 1, 10, 3)
    
    default_min_proj = 1.5 if selected_sport == 'nhl' else 0.0
    min_projection = st.slider("Min Base Projection", 0.0, 25.0, default_min_proj, step=0.1, help="Exclude players with a base projection lower than this value (applies to Batters in MLB).")
    
    min_pitcher_proj = 0.0
    mlb_roster_rule = "Flexible (Highest Projected)"
    if selected_sport == 'mlb':
        min_pitcher_proj = st.slider("Min Pitcher Base Projection", 0.0, 50.0, 10.0, step=0.5, help="Exclude pitchers with a base projection lower than this value.")
        mlb_roster_rule = st.selectbox(
            "MLB Lineup Constraint",
            ["Flexible (Highest Projected)", "3 Pitchers / 2 Batters", "2 Pitchers / 3 Batters", "4 Pitchers / 1 Batter", "1 Pitcher / 4 Batters"],
            index=0,
            help="Force the optimizer to pick a specific ratio of Pitchers to Batters, or let it blindly choose the highest raw scores."
        )
    
    min_proj_min = 0
    if selected_sport == 'ncaam':
        st.subheader("CBB Filters")
        min_proj_min = st.slider("Min Projected Minutes", 0, 40, 5, help="Filter out players with very low projected minutes.")


# --- TOP LEVEL TABS ---
app_tab, formula_tester_tab = st.tabs(["🚀 Main App", "🧮 Formula Tester"])


# --- FORMULA TESTER UI ---
with formula_tester_tab:
    st.header("🧮 Rating Formula Tester")
    st.write("Enter raw stats manually to verify how the custom efficiency formulas calculate player ratings.")
    
    test_sport = st.selectbox("Select Sport Formula", ["NBA", "WNBA", "NHL", "MLB Batter", "MLB Pitcher", "CBB"])
    
    if test_sport in ["NBA", "WNBA"]:
        st.subheader(f"{test_sport} Raw Stats")
        c1, c2, c3, c4 = st.columns(4)
        fgm = c1.number_input("FGM", 0.0, step=1.0, key="t_fgm")
        fga = c2.number_input("FGA", 0.0, step=1.0, key="t_fga")
        tpm = c3.number_input("3PM", 0.0, step=1.0, key="t_tpm")
        ftm = c4.number_input("FTM", 0.0, step=1.0, key="t_ftm")
        fta = c1.number_input("FTA", 0.0, step=1.0, key="t_fta")
        reb = c2.number_input("REB", 0.0, step=1.0, key="t_reb")
        ast = c3.number_input("AST", 0.0, step=1.0, key="t_ast")
        stl = c4.number_input("STL", 0.0, step=1.0, key="t_stl")
        blk = c1.number_input("BLK", 0.0, step=1.0, key="t_blk")
        tov = c2.number_input("TOV", 0.0, step=1.0, key="t_tov")
        
        row = {'fgm': fgm, 'fga': fga, '3pm': tpm, 'ftm': ftm, 'fta': fta, 'reb': reb, 'ast': ast, 'stl': stl, 'blk': blk, 'to': tov}
        mapping = {k: k for k in row.keys()}
        st.metric(f"Calculated {test_sport} Rating", f"{calculate_nba_custom_rating(row, mapping):.2f}")
        
    elif test_sport == "NHL":
        st.subheader("NHL Raw Stats")
        c1, c2, c3, c4 = st.columns(4)
        pts = c1.number_input("Points", 0.0, step=1.0, key="n_pts")
        gls = c2.number_input("Goals", 0.0, step=1.0, key="n_gls")
        sht = c3.number_input("Shots", 0.0, step=1.0, key="n_sht")
        blk = c4.number_input("Blocked Shots", 0.0, step=1.0, key="n_blk")
        
        row = {'points': pts, 'goals': gls, 'shots': sht, 'blockedShots': blk}
        mapping = {k: k for k in row.keys()}
        st.metric("Calculated NHL Rating", f"{calculate_nhl_custom_rating(row, mapping):.2f}")
        
    elif test_sport == "MLB Batter":
        st.subheader("MLB Batter Raw Stats")
        c1, c2, c3, c4 = st.columns(4)
        pa = c1.number_input("Plate Appearances (PA)", 0.0, step=1.0, key="b_pa")
        hits = c2.number_input("Hits", 0.0, step=1.0, key="b_hits")
        bb = c3.number_input("Walks (BB)", 0.0, step=1.0, key="b_bb")
        so = c4.number_input("Strikeouts (SO)", 0.0, step=1.0, key="b_so")
        
        s1b = c1.number_input("Singles (1B)", 0.0, step=1.0, key="b_1b")
        s2b = c2.number_input("Doubles (2B)", 0.0, step=1.0, key="b_2b")
        s3b = c3.number_input("Triples (3B)", 0.0, step=1.0, key="b_3b")
        hr = c4.number_input("Home Runs (HR)", 0.0, step=1.0, key="b_hr")
        
        runs = c1.number_input("Runs", 0.0, step=1.0, key="b_runs")
        rbi = c2.number_input("RBI", 0.0, step=1.0, key="b_rbi")
        sb = c3.number_input("Stolen Bases (SB)", 0.0, step=1.0, key="b_sb")
        cs = c4.number_input("Caught Stealing (CS)", 0.0, step=1.0, key="b_cs")
        
        row = {
            'plateAppearances': pa, 'hits': hits, 'walks': bb, 'strikeouts': so,
            'singles': s1b, 'doubles': s2b, 'triples': s3b, 'homeRuns': hr,
            'runs': runs, 'runsBattedIn': rbi, 'stolenBases': sb, 'caughtStealing': cs
        }
        mapping = {k: k for k in row.keys()}
        st.metric("Calculated MLB Batter Rating", f"{calculate_mlb_custom_rating(row, mapping):.2f}")
        
    elif test_sport == "MLB Pitcher":
        st.subheader("MLB Pitcher Raw Stats")
        c1, c2, c3, c4 = st.columns(4)
        ip = c1.number_input("Innings Pitched (IP)", 0.0, step=0.1, key="p_ip")
        so = c2.number_input("Strikeouts", 0.0, step=1.0, key="p_so")
        bb = c3.number_input("Walks Allowed", 0.0, step=1.0, key="p_bb")
        er = c4.number_input("Earned Runs (ER)", 0.0, step=1.0, key="p_er")
        
        hits = c1.number_input("Hits Allowed", 0.0, step=1.0, key="p_hits")
        hr = c2.number_input("HR Allowed", 0.0, step=1.0, key="p_hr")
        wins = c3.number_input("Wins", 0.0, step=1.0, key="p_wins")
        losses = c4.number_input("Losses", 0.0, step=1.0, key="p_loss")
        saves = c1.number_input("Saves", 0.0, step=1.0, key="p_sv")
        
        # MAPPING stats for test
        row = {
            'inningsPitched': ip, 'strikeouts_pitching': so, 'walksAllowed': bb, 
            'earnedRuns': er, 'hitsAllowed': hits, 'homeRunsAllowed': hr, 
            'wins': wins, 'losses': losses, 'saves': saves,
            'plateAppearances': 0, 'hits': 0, # Ensure hitting stats are 0 for pure pitcher test
            'gamesStarted': 1 # Force to 1 so the tester computes the score
        }
        mapping = {k: k for k in row.keys()}
        st.metric("Calculated MLB Pitcher Rating", f"{calculate_mlb_custom_rating(row, mapping):.2f}")
        st.info("Note: 'hitsAllowed' is used for Pitcher math, 'hits' is used for Batter math.")
        
    elif test_sport == "CBB":
        st.subheader("CBB Raw Stats")
        c1, c2, c3, c4 = st.columns(4)
        proj_min = c1.number_input("Projected Minutes", 0.0, step=1.0, key="c_pmin")
        avg_min = c2.number_input("Season Avg Minutes", 0.0, step=1.0, key="c_amin")
        st.write("---")
        c1, c2, c3, c4 = st.columns(4)
        fgm = c1.number_input("FGM", 0.0, step=1.0, key="c_fgm")
        fga = c2.number_input("FGA", 0.0, step=1.0, key="c_fga")
        tpm = c3.number_input("3PM", 0.0, step=1.0, key="c_tpm")
        ftm = c4.number_input("FTM", 0.0, step=1.0, key="c_ftm")
        fta = c1.number_input("FTA", 0.0, step=1.0, key="c_fta")
        reb = c2.number_input("REB", 0.0, step=1.0, key="c_reb")
        ast = c3.number_input("AST", 0.0, step=1.0, key="c_ast")
        stl = c4.number_input("STL", 0.0, step=1.0, key="c_stl")
        blk = c1.number_input("BLK", 0.0, step=1.0, key="c_blk")
        tov = c2.number_input("TOV", 0.0, step=1.0, key="c_tov")
        
        row = {
            'proj_min': proj_min, 'avg_min': avg_min,
            'fgm': fgm, 'fga': fga, '3pm': tpm, 'ftm': ftm, 'fta': fta, 
            'reb': reb, 'ast': ast, 'stl': stl, 'blk': blk, 'to': tov
        }
        mapping = {k: k for k in row.keys()}
        st.metric("Calculated CBB Rating", f"{calculate_cbb_custom_rating(row, mapping):.2f}")


# --- MAIN APP LOGIC ---
with app_tab:
    # 1. Fetch Live Logic (Merges into Global Store)
    if fetch_btn:
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text(f"Fetching {selected_sport.upper()}...")
            
            fetch_date = get_fantasy_day()
            data, fetch_date_str = fetch_data_for_sport(selected_sport, fetch_date)
            all_results.extend(data)
        except Exception as e:
            st.error(f"Error fetching data: {e}")
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        if all_results:
            raw_df = pd.DataFrame(all_results)
            raw_df = raw_df.sort_values('Date', ascending=False)
            raw_df = raw_df.drop_duplicates(subset=['Player Name'], keep='first')
            
            api_data_map = {row['Player Name']: row.to_dict() for _, row in raw_df.iterrows()}
            
            current_df = boost_store.get()
            if current_df.empty:
                 current_df = pd.DataFrame(columns=['Sport', 'Player Name', 'Position', 'Boost', 'Date', 'Injury'])
            current_df = standardize_boost_columns(current_df)
            
            updated_rows = []
            processed_names = set()
            
            for _, row in current_df.iterrows():
                if str(row.get('Sport', '')).upper() == selected_sport.upper():
                    name = row['Player Name']
                    processed_names.add(name)
                    new_row = row.to_dict()
                    new_row['Date'] = str(fetch_date_str) 
                    
                    if name in api_data_map:
                        api_row = api_data_map[name]
                        new_row['Injury'] = api_row.get('Injury', '')
                        api_boost = api_row.get('Boost', 0.0)
                        old_boost = row.get('Boost', 0.0)
                        if api_boost > 0.0 and api_boost != old_boost:
                             new_row['Boost'] = api_boost
                    
                    updated_rows.append(new_row)
                else:
                    updated_rows.append(row.to_dict())
                    
            for name, row in api_data_map.items():
                if name not in processed_names:
                    updated_rows.append(row)
                    
            boost_store.update(pd.DataFrame(updated_rows))
            st.success(f"Fetched and Merged Data for {selected_sport.upper()}!")
        else:
            st.warning(f"No active data found in API for {selected_sport.upper()}. Using any stored data.")


    # 2. Projection Logic 
    if 'proj_df' not in st.session_state:
        st.session_state.proj_df = None

    df_proj = st.session_state.proj_df
    df_proj_copy = None

    if input_method == "Use Global/Public Projections" and current_proj_url:
        if st.session_state.proj_df is None:
             df_proj_copy, _ = load_projections_from_url(current_proj_url)
             if df_proj_copy is not None:
                 # Merge MLB Pitchers if MLB is selected
                 if selected_sport == 'mlb':
                     try:
                         pitchers_df, _ = load_projections_from_url(MLB_PITCHERS_URL)
                         if pitchers_df is not None and not pitchers_df.empty:
                             # Rename conflicting pitcher columns to ensure they don't merge with hitting stats
                             p_rename_map = {
                                 "hits": "hitsAllowed",
                                 "homeruns": "homeRunsAllowed",
                                 "walks": "walksAllowed",
                                 "strikeouts": "strikeouts_pitching",
                                 "runs": "runsAllowed"
                             }
                             actual_rename = {}
                             for col in pitchers_df.columns:
                                 cleaned_col = col.strip().lower()
                                 if cleaned_col in p_rename_map:
                                     actual_rename[col] = p_rename_map[cleaned_col]
                             pitchers_df = pitchers_df.rename(columns=actual_rename)

                             df_proj_copy = pd.concat([df_proj_copy, pitchers_df], ignore_index=True)
                             
                             # Aggregate two-way players into a single row using their name
                             player_col = find_col(df_proj_copy.columns, ["player", "name", "who"])
                             if player_col:
                                 df_proj_copy['grp_name'] = df_proj_copy[player_col].astype(str).str.lower().str.strip()
                                 agg_dict = {}
                                 for c in df_proj_copy.columns:
                                     if c == 'grp_name': continue
                                     try:
                                         df_proj_copy[c] = df_proj_copy[c].astype(float)
                                         agg_dict[c] = 'sum'
                                     except (ValueError, TypeError):
                                         agg_dict[c] = 'first'
                                         
                                 df_proj_copy = df_proj_copy.groupby('grp_name', as_index=False).agg(agg_dict)
                                 df_proj_copy = df_proj_copy.drop(columns=['grp_name'])
                     except Exception as e:
                         pass
                         
                 st.session_state.proj_df = df_proj_copy
                 st.rerun()
        else:
             df_proj_copy = st.session_state.proj_df
    elif uploaded_file:
        try: df_proj_copy = pd.read_csv(uploaded_file)
        except: pass
    elif pasted_text:
        try:
            df_proj_copy = pd.read_csv(io.StringIO(pasted_text), sep="\t")
            if len(df_proj_copy.columns) < 2: df_proj_copy = pd.read_csv(io.StringIO(pasted_text), sep=",")
        except: pass

    # 3. Merging & Optimization
    df_boosts = boost_store.get()

    proceed = False
    if not df_boosts.empty:
        proceed = True

    if proceed:
        df_boosts = standardize_boost_columns(df_boosts)
        sport_boosts = df_boosts[df_boosts['Sport'].str.upper() == selected_sport.upper()].copy()

        # --- CASE A: HAVE PROJECTIONS ---
        if df_proj_copy is not None and not df_proj_copy.empty:
            df_proj = df_proj_copy.copy()
            
            if isinstance(df_proj.columns, pd.MultiIndex):
                df_proj.columns = [' '.join(col).strip() for col in df_proj.columns.values]
            df_proj.columns = [str(c).strip() for c in df_proj.columns]
            
            first_name_col = find_col(df_proj.columns, ["first name", "firstname", "first"])
            last_name_col = find_col(df_proj.columns, ["last name", "lastname", "last"])
            name_col = None
            if first_name_col and last_name_col:
                df_proj['Calculated_Full_Name'] = df_proj[first_name_col].astype(str) + " " + df_proj[last_name_col].astype(str)
                name_col = 'Calculated_Full_Name'
            else:
                name_col = find_col(df_proj.columns, ["player", "name", "who"])

            # --- NHL SECONDARY LINES CSV MERGE ---
            if selected_sport == 'nhl' and input_method == "Use Global/Public Projections" and name_col:
                try:
                    lines_df, _ = load_projections_from_url(NHL_LINES_URL)
                    if lines_df is not None and not lines_df.empty:
                        l_first_name = find_col(lines_df.columns, ["first name", "firstname", "first"])
                        l_last_name = find_col(lines_df.columns, ["last name", "lastname", "last"])
                        l_name_col = None
                        
                        if l_first_name and l_last_name:
                            lines_df['Calc_Name'] = lines_df[l_first_name].astype(str) + " " + lines_df[l_last_name].astype(str)
                            l_name_col = 'Calc_Name'
                        else:
                            l_name_col = find_col(lines_df.columns, ["player", "name", "who"])
                        
                        if l_name_col:
                            rl_col = find_col(lines_df.columns, ["reg_line", "line"])
                            pp_col = find_col(lines_df.columns, ["pp_line", "power"])
                            fpts_col = find_col(lines_df.columns, ["fpts", "fantasy points", "fantasy", "proj fpts", "projection"])
                            
                            lines_df['join_key'] = lines_df[l_name_col].apply(normalize_name)
                            cols_to_keep = ['join_key']
                            
                            if rl_col: 
                                lines_df = lines_df.rename(columns={rl_col: 'reg_line'})
                                cols_to_keep.append('reg_line')
                            if pp_col: 
                                lines_df = lines_df.rename(columns={pp_col: 'pp_line'})
                                cols_to_keep.append('pp_line')
                            if fpts_col:
                                lines_df = lines_df.rename(columns={fpts_col: 'lines_csv_fpts'})
                                cols_to_keep.append('lines_csv_fpts')
                                
                            lines_df = lines_df[cols_to_keep].drop_duplicates(subset=['join_key'])
                            
                            df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                            df_proj = pd.merge(df_proj, lines_df, on='join_key', how='left')
                            df_proj = df_proj.drop(columns=['join_key'])
                            st.success("✅ Secondary NHL Line Data Successfully Merged")
                except Exception as e:
                    pass


            points_col = None 
            
            if selected_sport in ["nba", "wnba"]:
                nba_cols_map = {
                    "fgm": find_col(df_proj.columns, ["fieldGoalsMade", "fgm"]),
                    "fga": find_col(df_proj.columns, ["fieldGoalsAttempted", "fga"]),
                    "3pm": find_col(df_proj.columns, ["threePointsMade", "3pm"]),
                    "ftm": find_col(df_proj.columns, ["freeThrowsMade", "ftm"]),
                    "fta": find_col(df_proj.columns, ["freeThrowsAttempted", "fta"]),
                    "reb": find_col(df_proj.columns, ["rebounds", "reb", "tot reb"]),
                    "ast": find_col(df_proj.columns, ["assists", "ast"]),
                    "stl": find_col(df_proj.columns, ["steals", "stl"]),
                    "blk": find_col(df_proj.columns, ["blocks", "blk"]),
                    "to":  find_col(df_proj.columns, ["turnovers", "to", "tov"])
                }
                if all(v is not None for v in nba_cols_map.values()):
                    df_proj['Calculated_Rating'] = df_proj.apply(lambda row: calculate_nba_custom_rating(row, nba_cols_map), axis=1)
                    points_col = 'Calculated_Rating'
                    st.success(f"✅ {selected_sport.upper()} Custom Efficiency Rating Applied")
                    
            if selected_sport == "mlb":
                mlb_cols_map = {
                    "plateAppearances": find_col(df_proj.columns, ["plateappearances", "pa"]),
                    "hits": find_col(df_proj.columns, ["hits"]),
                    "walks": find_col(df_proj.columns, ["walks", "bb"]),
                    "singles": find_col(df_proj.columns, ["singles", "1b"]),
                    "doubles": find_col(df_proj.columns, ["doubles", "2b"]),
                    "triples": find_col(df_proj.columns, ["triples", "3b"]),
                    "homeRuns": find_col(df_proj.columns, ["homeruns", "hr"]),
                    "strikeouts": find_col(df_proj.columns, ["strikeouts", "so"]),
                    "caughtStealing": find_col(df_proj.columns, ["caughtstealing", "cs"]),
                    "runs": find_col(df_proj.columns, ["runs"]),
                    "runsBattedIn": find_col(df_proj.columns, ["runsbattedin", "rbi"]),
                    "stolenBases": find_col(df_proj.columns, ["stolenbases", "sb"]),
                    "inningsPitched": find_col(df_proj.columns, ["inningspitched", "ip"]),
                    "earnedRuns": find_col(df_proj.columns, ["earnedruns", "er"]),
                    "losses": find_col(df_proj.columns, ["losses"]),
                    "wins": find_col(df_proj.columns, ["wins"]),
                    "saves": find_col(df_proj.columns, ["saves", "sv"]),
                    "hitsAllowed": find_col(df_proj.columns, ["hitsallowed"]),
                    "homeRunsAllowed": find_col(df_proj.columns, ["homerunsallowed"]),
                    "walksAllowed": find_col(df_proj.columns, ["walksallowed"]),
                    "strikeouts_pitching": find_col(df_proj.columns, ["strikeouts_pitching"]),
                    "gamesStarted": find_col(df_proj.columns, ["gamesstarted", "gs"])
                }
                
                if mlb_cols_map["plateAppearances"] or mlb_cols_map["inningsPitched"]:
                    df_proj['Calculated_Rating'] = df_proj.apply(lambda row: calculate_mlb_custom_rating(row, mlb_cols_map), axis=1)
                    points_col = 'Calculated_Rating'
                    st.success("✅ MLB Custom Efficiency Rating Applied (Batter & Pitcher Supported)")
                    
                # Extract to DataFrame so we can use it to filter out non-starters later
                if mlb_cols_map["gamesStarted"]:
                    df_proj['gamesStarted'] = pd.to_numeric(df_proj[mlb_cols_map["gamesStarted"]], errors='coerce').fillna(0)
                else:
                    df_proj['gamesStarted'] = 0
                    
                if mlb_cols_map["inningsPitched"]:
                    df_proj['inningsPitched'] = pd.to_numeric(df_proj[mlb_cols_map["inningsPitched"]], errors='coerce').fillna(0)
                else:
                    df_proj['inningsPitched'] = 0
                    
            if selected_sport == "golf":
                golf_cols_map = {
                    "to_win": find_col(df_proj.columns, ["to win", "win"]),
                    "top_5": find_col(df_proj.columns, ["top 5", "t5"]),
                    "top_10": find_col(df_proj.columns, ["top 10", "t10"]),
                    "top_20": find_col(df_proj.columns, ["top 20", "t20"]),
                    "top_40": find_col(df_proj.columns, ["top 40", "t40"]),
                    "make_cut": find_col(df_proj.columns, ["make cut", "cut"])
                }
                if any(v is not None for v in golf_cols_map.values()):
                    df_proj['Calculated_Rating'] = df_proj.apply(lambda row: calculate_golf_custom_rating(row, golf_cols_map), axis=1)
                    points_col = 'Calculated_Rating'
                    st.success("✅ Golf Odds Implied Probability Rating Applied")

            
            if selected_sport == "nhl":
                if nhl_proj_source == "Fantasy Points (Lines CSV)" and 'lines_csv_fpts' in df_proj.columns:
                    points_col = 'lines_csv_fpts'
                    st.success("✅ NHL Fantasy Points (from Lines CSV) Applied")
                else:
                    if nhl_proj_source == "Fantasy Points (Lines CSV)":
                        st.warning("⚠️ Could not find Fantasy Points in Lines CSV. Falling back to formula.")
                        
                    nhl_cols_map = {
                        "points": find_col(df_proj.columns, ["points", "pts"]),
                        "goals": find_col(df_proj.columns, ["goals"]),
                        "shots": find_col(df_proj.columns, ["shots", "sog"]),
                        "blockedShots": find_col(df_proj.columns, ["blocks", "blk", "blocked"])
                    }
                    if all(v is not None for v in nhl_cols_map.values()):
                        df_proj['Calculated_Rating'] = df_proj.apply(lambda row: calculate_nhl_custom_rating(row, nhl_cols_map), axis=1)
                        points_col = 'Calculated_Rating'
                        st.success("✅ NHL Custom Efficiency Rating Applied")
                    elif not points_col:
                        points_col = find_col(df_proj.columns, ["ppg_projection"])

            if not points_col:
                points_col = find_col(df_proj.columns, ["ppg", "fantasy", "proj", "fpts", "pts", "avg", "fp"])

            pos_col = find_col(df_proj.columns, ["pos", "position"])
            slate_col = find_col(df_proj.columns, ["slate", "contest", "label"])
            game_col = find_col(df_proj.columns, ["game", "matchup", "match", "gameinfo"])
            team_col = find_col(df_proj.columns, ["team", "tm", "squad"])
            opp_col = find_col(df_proj.columns, ["opp", "opponent", "vs"])
            
            if not game_col and not (team_col and opp_col):
                for col in df_proj.columns:
                    sample = df_proj[col].dropna().astype(str).head(5)
                    if any(" v " in x.lower() or " vs " in x.lower() or "@" in x for x in sample):
                        game_col = col
                        break

            if name_col and points_col:
                if selected_sport == 'nhl':
                    # Convert line data to numeric specifically
                    rl_col = find_col(df_proj.columns, ["reg_line"])
                    pp_col = find_col(df_proj.columns, ["pp_line"])
                    if rl_col:
                        df_proj[rl_col] = pd.to_numeric(df_proj[rl_col], errors='coerce')
                    if pp_col:
                        df_proj[pp_col] = pd.to_numeric(df_proj[pp_col], errors='coerce')

                sport_boosts['join_key'] = sport_boosts['Player Name'].apply(normalize_name)
                df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                
                # Use outer join so EVERY player from API or CSV is available in the app (like Tester)
                # For Golf, use right join to only show players listed in the uploaded sheet
                if selected_sport == 'golf':
                    merged_df = pd.merge(sport_boosts, df_proj, on='join_key', how='right')
                else:
                    merged_df = pd.merge(sport_boosts, df_proj, on='join_key', how='outer')
                
                merged_df['Boost'] = merged_df['Boost'].fillna(0.0)
                if name_col in merged_df.columns:
                    merged_df['Player Name'] = merged_df['Player Name'].fillna(merged_df[name_col])
                
                if pos_col and pos_col in merged_df.columns:
                    merged_df['Position'] = merged_df['Position'].fillna(merged_df[pos_col])
                if 'Position' not in merged_df.columns:
                    merged_df['Position'] = 'UNKNOWN'
                    
                merged_df['Position'] = merged_df['Position'].fillna('UNKNOWN').apply(normalize_position)
                
                # Convert Position to numeric for Golf so it sorts properly (1, 2, 3 instead of 1, 10, 2)
                if selected_sport == 'golf':
                    merged_df['Position'] = pd.to_numeric(merged_df['Position'], errors='coerce').astype('Int64')
                
                merged_df['Injury'] = merged_df['Injury'].fillna('')
                merged_df['Sport'] = merged_df['Sport'].fillna(selected_sport.upper())

                if not merged_df.empty:
                    if points_col in merged_df.columns:
                        merged_df = merged_df.rename(columns={points_col: 'Projection'})
                    else:
                        merged_df['Projection'] = 0.0
                        
                    if slate_col and slate_col in merged_df.columns:
                        merged_df['Slate'] = merged_df[slate_col].fillna("ALL")
                    else:
                        merged_df['Slate'] = "ALL"
                        
                    if team_col and opp_col and team_col in merged_df.columns and opp_col in merged_df.columns:
                        def get_game(x):
                            t = x.get(team_col)
                            o = x.get(opp_col)
                            if pd.notna(t) and pd.notna(o):
                                return " vs ".join(sorted([str(t), str(o)]))
                            return "Unknown"
                        merged_df['Game'] = merged_df.apply(get_game, axis=1)
                    elif game_col and game_col in merged_df.columns:
                        merged_df['Game'] = merged_df[game_col].fillna("Unknown")
                    else:
                        merged_df['Game'] = "ALL"

                    merged_df['Projection'] = pd.to_numeric(merged_df['Projection'], errors='coerce').fillna(0)

                    def get_bias_multiplier(row):
                        if row['Position'] in ['WR', 'RB']: return wr_rb_bonus
                        if row['Position'] == 'QB': return qb_penalty
                        return 1.0

                    merged_df['Bias'] = merged_df.apply(get_bias_multiplier, axis=1)
                    merged_df['Adjusted Projection'] = merged_df['Projection'] * merged_df['Bias']
                    merged_df['Optimization Score'] = (merged_df['Boost'] + 2.0) * merged_df['Adjusted Projection']
                    merged_df['Est. Score'] = merged_df['Boost'] * merged_df['Projection']
                    
                    # NEW: Calculate expected points for each specific slot
                    merged_df['Slot 1 (2.0x)'] = (merged_df['Boost'] + 2.0) * merged_df['Projection']
                    merged_df['Slot 2 (1.8x)'] = (merged_df['Boost'] + 1.8) * merged_df['Projection']
                    merged_df['Slot 3 (1.6x)'] = (merged_df['Boost'] + 1.6) * merged_df['Projection']
                    merged_df['Slot 4 (1.4x)'] = (merged_df['Boost'] + 1.4) * merged_df['Projection']
                    merged_df['Slot 5 (1.2x)'] = (merged_df['Boost'] + 1.2) * merged_df['Projection']

                    if selected_sport == 'mlb':
                        if 'inningsPitched' in merged_df.columns:
                            merged_df['Is_Pitcher'] = pd.to_numeric(merged_df['inningsPitched'], errors='coerce').fillna(0) > 0
                        else:
                            merged_df['Is_Pitcher'] = False

                    # NEW: Restricting columns visually in tabs per user request (added slot values)
                    display_cols = ['Player Name', 'Boost', 'Injury', 'Projection', 'Optimization Score', 'Slot 1 (2.0x)', 'Slot 2 (1.8x)', 'Slot 3 (1.6x)', 'Slot 4 (1.4x)', 'Slot 5 (1.2x)']
                    
                    # Add NHL line data into the visual display if it exists
                    if selected_sport == 'nhl':
                        rl_col = find_col(merged_df.columns, ["reg_line"])
                        pp_col = find_col(merged_df.columns, ["pp_line"])
                        if pp_col: display_cols.insert(3, pp_col)
                        if rl_col: display_cols.insert(3, rl_col)
                        
                    # Add Golf odds data into the visual display if it exists
                    if selected_sport == 'golf':
                        for c_key in ["to_win", "top_5", "top_10", "top_20", "top_40", "make_cut"]:
                            c_name = golf_cols_map.get(c_key)
                            if c_name and c_name in merged_df.columns:
                                display_cols.insert(4, c_name)

                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Browser", "💎 Best Value", "🚀 Lineup Optimizer", "🧩 Lineup Assistant", "🧪 Lineup Tester"])
                    
                    # Shared formatting for the number columns
                    format_cfg = {
                        "Optimization Score": st.column_config.NumberColumn(format="%.2f"),
                        "Slot 1 (2.0x)": st.column_config.NumberColumn(format="%.2f"),
                        "Slot 2 (1.8x)": st.column_config.NumberColumn(format="%.2f"),
                        "Slot 3 (1.6x)": st.column_config.NumberColumn(format="%.2f"),
                        "Slot 4 (1.4x)": st.column_config.NumberColumn(format="%.2f"),
                        "Slot 5 (1.2x)": st.column_config.NumberColumn(format="%.2f")
                    }
                    
                    with tab1:
                        available_cols = [c for c in display_cols if c in merged_df.columns]
                        st.dataframe(merged_df[available_cols].sort_values('Optimization Score', ascending=False), use_container_width=True, hide_index=True, column_config=format_cfg)

                    with tab2:
                        best_value_df = merged_df.copy()
                        
                        # Exclude goalies from the Best Value list for NHL
                        if selected_sport == 'nhl':
                            best_value_df = best_value_df[~best_value_df['Position'].isin(['G', 'GOALIE'])]
                        
                        # Exclude non-starting pitchers from the Best Value list for MLB
                        if selected_sport == 'mlb' and 'gamesStarted' in best_value_df.columns and 'inningsPitched' in best_value_df.columns:
                            best_value_df = best_value_df[~((best_value_df['inningsPitched'] > 0) & (best_value_df['gamesStarted'] < 1))]
                            
                        # Apply adjustable minimum projection filter
                        if selected_sport == 'mlb' and 'Is_Pitcher' in best_value_df.columns:
                            proj_mask = (
                                (best_value_df['Is_Pitcher'] & (best_value_df['Projection'] >= min_pitcher_proj)) |
                                (~best_value_df['Is_Pitcher'] & (best_value_df['Projection'] >= min_projection))
                            )
                            best_value_df = best_value_df[proj_mask]
                        else:
                            best_value_df = best_value_df[best_value_df['Projection'] >= min_projection]
                            
                        st.dataframe(
                            best_value_df[available_cols].sort_values('Optimization Score', ascending=False).head(50), 
                            use_container_width=True,
                            hide_index=True,
                            column_config=format_cfg
                        )

                    with tab3:
                        col1, col2 = st.columns(2)
                        with col1:
                            unique_slates = sorted(list(set(merged_df['Slate'].astype(str).unique().tolist()) - {"ALL"}))
                            slate_options = ["ALL"] + unique_slates
                            selected_slates = st.multiselect("Filter by Slate:", slate_options, default=["ALL"])

                        with col2:
                            unique_games = sorted(list(set(merged_df['Game'].astype(str).unique().tolist()) - {"ALL"}))
                            game_options = ["ALL"] + unique_games
                            selected_games = st.multiselect("Filter by Game:", game_options, default=["ALL"])
                        
                        all_player_names = sorted(merged_df['Player Name'].dropna().unique().tolist())
                        excluded_players = st.multiselect(
                            "Filter by Player (Exclude):",
                            all_player_names,
                            default=[],
                            placeholder="Search for players to exclude..."
                        )
                        
                        filtered_df = merged_df.copy()
                        
                        if excluded_players:
                             filtered_df = filtered_df[~filtered_df['Player Name'].isin(excluded_players)]

                        if "ALL" not in selected_slates:
                            filtered_df = filtered_df[filtered_df['Slate'].isin(selected_slates)]
                        if "ALL" not in selected_games:
                            filtered_df = filtered_df[filtered_df['Game'].isin(selected_games)]
                            
                        # CRITICAL FIX: Automatically drop strictly 'OUT' players AND players below the adjustable min projection
                        base_mask = (~filtered_df['Injury'].astype(str).str.strip().str.upper().isin(['O', 'OUT', 'IR', 'INJ'])) & (filtered_df['Projection'] > 0)
                        
                        if selected_sport == 'mlb' and 'Is_Pitcher' in filtered_df.columns:
                            proj_mask = (
                                (filtered_df['Is_Pitcher'] & (filtered_df['Projection'] >= min_pitcher_proj)) |
                                (~filtered_df['Is_Pitcher'] & (filtered_df['Projection'] >= min_projection))
                            )
                            opt_df = filtered_df[base_mask & proj_mask].copy()
                        else:
                            opt_df = filtered_df[base_mask & (filtered_df['Projection'] >= min_projection)].copy()
                        
                        # APPLY MLB NON-STARTER FILTER TO OPTIMIZER
                        if selected_sport == 'mlb' and 'gamesStarted' in opt_df.columns and 'inningsPitched' in opt_df.columns:
                            opt_df = opt_df[~((opt_df['inningsPitched'] > 0) & (opt_df['gamesStarted'] < 1))]
                        
                        if selected_sport == 'nhl':
                            st.caption(f"Pool Size: {len(opt_df)} Players (excludes injured & proj < {min_projection})")
                        elif selected_sport == 'mlb':
                            st.caption(f"Pool Size: {len(opt_df)} Players (excludes injured, batters < {min_projection}, pitchers < {min_pitcher_proj}, & non-starting pitchers)")
                        else:
                            st.caption(f"Pool Size: {len(opt_df)} Players (excludes injured & proj < {min_projection})")

                        if st.button("Generate Optimal Lineups"):
                            lineups = run_optimization(opt_df, num_lineups, sport=selected_sport, mlb_rule=mlb_roster_rule)
                            if lineups:
                                for idx, lineup in enumerate(lineups):
                                    total_score = lineup['Optimization Score'].sum()
                                    q_players = lineup[lineup['Injury'].astype(str).str.startswith('Q', na=False)]['Player Name'].tolist()
                                    warn_icon = "⚠️ " if q_players else ""
                                    
                                    with st.expander(f"{warn_icon}Lineup #{idx+1} | Total Score: {total_score:.2f}", expanded=(idx==0)):
                                        if q_players:
                                            st.warning(f"**Questionable Status:** {', '.join(q_players)}")
                                        
                                        # Set column order for output Lineups
                                        lineup_disp_cols = ['Slot', 'Player Name', 'Boost', 'Eff. Boost', 'Injury', 'Projection', 'Optimization Score']
                                        st.dataframe(
                                            lineup[lineup_disp_cols], 
                                            column_config={
                                                "Optimization Score": st.column_config.NumberColumn(format="%.2f"),
                                                "Projection": st.column_config.NumberColumn(format="%.2f"),
                                            },
                                            use_container_width=True,
                                            hide_index=True
                                        )
                            else:
                                st.error("Could not generate lineup.")

                    # --- NEW: TAB 4 (Lineup Assistant) ---
                    with tab4:
                        st.write("Manually lock specific players into specific slots and let the optimizer fill the rest.")
                        
                        # Prepare list with a default "Empty" option
                        all_assistant_names = ["-- Unassigned --"] + sorted(merged_df['Player Name'].dropna().astype(str).unique().tolist())
                        
                        locked_slots = {}
                        
                        # Create 5 columns for the 5 slots
                        colA, colB, colC, colD, colE = st.columns(5)
                        
                        with colA:
                            s1 = st.selectbox("Slot 1 (2.0x)", all_assistant_names, key="lock_s1")
                            if s1 != "-- Unassigned --": locked_slots[0] = s1
                        with colB:
                            s2 = st.selectbox("Slot 2 (1.8x)", all_assistant_names, key="lock_s2")
                            if s2 != "-- Unassigned --": locked_slots[1] = s2
                        with colC:
                            s3 = st.selectbox("Slot 3 (1.6x)", all_assistant_names, key="lock_s3")
                            if s3 != "-- Unassigned --": locked_slots[2] = s3
                        with colD:
                            s4 = st.selectbox("Slot 4 (1.4x)", all_assistant_names, key="lock_s4")
                            if s4 != "-- Unassigned --": locked_slots[3] = s4
                        with colE:
                            s5 = st.selectbox("Slot 5 (1.2x)", all_assistant_names, key="lock_s5")
                            if s5 != "-- Unassigned --": locked_slots[4] = s5

                        # Validation
                        selected_locked_names = list(locked_slots.values())
                        has_duplicates = len(selected_locked_names) != len(set(selected_locked_names))
                        
                        st.write("---")
                        assistant_excluded = st.multiselect(
                            "❌ Exclude Players from remaining slots:",
                            [p for p in sorted(merged_df['Player Name'].dropna().unique().tolist()) if p not in selected_locked_names],
                            default=[],
                            placeholder="Search players to ignore..."
                        )
                            
                        st.caption(f"Slots remaining to fill automatically: **{5 - len(locked_slots)}**")
                        b_num_lineups = st.slider("Number of Assisted Lineups to Generate", 1, 10, 3, key="builder_slider")
                        
                        if has_duplicates:
                            st.error("⚠️ You cannot lock the same player into multiple slots at once.")
                        elif st.button("Build Assistant Lineups"):
                            if len(locked_slots) > 0:
                                builder_df = merged_df.copy()
                                
                                # Filter out strictly OUT players and those below min_projection for the assistant pool (unless locked manually)
                                base_mask = (~builder_df['Injury'].astype(str).str.strip().str.upper().isin(['O', 'OUT', 'IR', 'INJ'])) & (builder_df['Projection'] > 0)
                                
                                if selected_sport == 'mlb' and 'Is_Pitcher' in builder_df.columns:
                                    proj_mask = (
                                        (builder_df['Is_Pitcher'] & (builder_df['Projection'] >= min_pitcher_proj)) |
                                        (~builder_df['Is_Pitcher'] & (builder_df['Projection'] >= min_projection))
                                    )
                                    valid_pool = builder_df[base_mask & proj_mask]
                                else:
                                    valid_pool = builder_df[base_mask & (builder_df['Projection'] >= min_projection)]
                                    
                                builder_df = builder_df[builder_df.index.isin(valid_pool.index) | builder_df['Player Name'].isin(selected_locked_names)]

                                # APPLY MLB NON-STARTER FILTER TO ASSISTANT (allow locked players to bypass filter)
                                if selected_sport == 'mlb' and 'gamesStarted' in builder_df.columns and 'inningsPitched' in builder_df.columns:
                                    builder_df = builder_df[
                                        (~((builder_df['inningsPitched'] > 0) & (builder_df['gamesStarted'] < 1))) |
                                        builder_df['Player Name'].isin(selected_locked_names)
                                    ]

                                if assistant_excluded:
                                    builder_df = builder_df[~builder_df['Player Name'].isin(assistant_excluded)]
                                    
                                built_lineups = run_optimization(builder_df, b_num_lineups, locked_slots=locked_slots, sport=selected_sport, mlb_rule=mlb_roster_rule)
                                if built_lineups:
                                    for idx, lineup in enumerate(built_lineups):
                                        total_score = lineup['Optimization Score'].sum()
                                        q_players = lineup[lineup['Injury'].astype(str).str.startswith('Q', na=False)]['Player Name'].tolist()
                                        warn_icon = "⚠️ " if q_players else ""
                                        
                                        with st.expander(f"{warn_icon}Assistant Lineup #{idx+1} | Total Score: {total_score:.2f}", expanded=(idx==0)):
                                            if q_players:
                                                st.warning(f"**Questionable Status:** {', '.join(q_players)}")
                                                
                                            lineup_disp_cols = ['Slot', 'Player Name', 'Boost', 'Eff. Boost', 'Injury', 'Projection', 'Optimization Score']
                                            st.dataframe(
                                                lineup[lineup_disp_cols], 
                                                column_config={
                                                    "Optimization Score": st.column_config.NumberColumn(format="%.2f"),
                                                    "Projection": st.column_config.NumberColumn(format="%.2f"),
                                                },
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                else:
                                    st.error("Could not generate a valid lineup with these locks. Check your constraints.")
                            else:
                                st.info("Please lock at least one player to use the assistant.")
                                
                    # --- NEW: TAB 5 (Lineup Tester) ---
                    with tab5:
                        st.write("Test a specific custom lineup to see its projected score.")
                        
                        test_names = ["-- Select Player --"] + sorted(merged_df['Player Name'].dropna().astype(str).unique().tolist())
                        
                        t_colA, t_colB, t_colC, t_colD, t_colE = st.columns(5)
                        
                        with t_colA:
                            t_s1 = st.selectbox("Slot 1 (2.0x)", test_names, key="test_s1")
                        with t_colB:
                            t_s2 = st.selectbox("Slot 2 (1.8x)", test_names, key="test_s2")
                        with colC:
                            t_s3 = st.selectbox("Slot 3 (1.6x)", test_names, key="test_s3")
                        with colD:
                            t_s4 = st.selectbox("Slot 4 (1.4x)", test_names, key="test_s4")
                        with colE:
                            t_s5 = st.selectbox("Slot 5 (1.2x)", test_names, key="test_s5")
                            
                        tester_selections = [t_s1, t_s2, t_s3, t_s4, t_s5]
                        tester_valid = [name for name in tester_selections if name != "-- Select Player --"]
                        
                        if len(tester_valid) != len(set(tester_valid)):
                            st.error("⚠️ You cannot select the same player in multiple slots.")
                        elif len(tester_valid) > 0:
                            tester_data = []
                            slot_adders = [2.0, 1.8, 1.6, 1.4, 1.2]
                            
                            for i, p_name in enumerate(tester_selections):
                                if p_name != "-- Select Player --":
                                    p_row = merged_df[merged_df['Player Name'] == p_name].iloc[0]
                                    p_proj = p_row['Projection']
                                    p_boost = p_row['Boost']
                                    p_inj = p_row['Injury']
                                    
                                    slot_add = slot_adders[i]
                                    eff_boost = p_boost + slot_add
                                    pts = eff_boost * p_proj
                                    
                                    tester_data.append({
                                        "Slot": i + 1,
                                        "Slot Bonus": f"+{slot_add}x",
                                        "Player Name": p_name,
                                        "Boost": p_boost,
                                        "Eff. Boost": f"{eff_boost:.2f}x",
                                        "Injury": p_inj,
                                        "Projection": p_proj,
                                        "Score": pts
                                    })
                                    
                            if tester_data:
                                tester_df = pd.DataFrame(tester_data)
                                total_test_score = tester_df['Score'].sum()
                                
                                st.subheader(f"Total Projected Score: {total_test_score:.2f}")
                                
                                q_players_test = tester_df[tester_df['Injury'].astype(str).str.startswith('Q', na=False)]['Player Name'].tolist()
                                if q_players_test:
                                    st.warning(f"**Questionable Status:** {', '.join(q_players_test)}")
                                
                                st.dataframe(
                                    tester_df,
                                    column_config={
                                        "Score": st.column_config.NumberColumn(format="%.2f"),
                                        "Projection": st.column_config.NumberColumn(format="%.2f"),
                                    },
                                    use_container_width=True,
                                    hide_index=True
                                )
            
            else:
                # --- CASE B: BOOSTS ONLY (Fallback) ---
                if not sport_boosts.empty:
                    st.subheader(f"Raw Boosts for {selected_sport.upper()}")
                    
                    display_boosts = sport_boosts.copy()
                    
                    if allowed_participants:
                        display_boosts['norm_for_filter'] = display_boosts['Player Name'].astype(str).apply(normalize_name)
                        display_boosts = display_boosts[display_boosts['norm_for_filter'].isin(allowed_participants)].drop(columns=['norm_for_filter'])
                        st.success(f"🎯 Participant Filter Active: {len(display_boosts)} players matched from sheet.")
                    else:
                        st.write("Showing the raw API boost data.")
                        
                    if selected_sport == 'golf':
                        display_boosts['Position'] = pd.to_numeric(display_boosts['Position'], errors='coerce').astype('Int64')
                        
                    cols_to_show = ['Player Name', 'Boost', 'Position', 'Injury', 'Date']
                    if selected_sport == 'golf':
                        cols_to_show = ['Player Name', 'Boost', 'Position']
                        
                    st.dataframe(
                        display_boosts[cols_to_show], 
                        use_container_width=True
                    )
                else:
                    st.info("Waiting for data fetch...")

        else:
            # --- CASE B: BOOSTS ONLY (No Projections provided at all) ---
            if not sport_boosts.empty:
                st.subheader(f"Raw Boosts for {selected_sport.upper()} (No Projections Found)")
                
                st.write("Since no projections CSV is available, showing just the raw API boost data.")
                
                display_boosts = sport_boosts.copy()
                
                # Fix Golf numerical sorting
                if selected_sport == 'golf':
                    display_boosts['Position'] = pd.to_numeric(display_boosts['Position'], errors='coerce').astype('Int64')
                    
                cols_to_show = ['Player Name', 'Boost', 'Position', 'Injury', 'Date']
                if selected_sport == 'golf':
                    cols_to_show = ['Player Name', 'Boost', 'Position']
                    
                st.dataframe(
                    display_boosts[cols_to_show], 
                    use_container_width=True
                )
            else:
                st.info("Waiting for data fetch...")
    else:
        st.write("Waiting for data fetch...")
