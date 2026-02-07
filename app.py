import streamlit as st
import requests
import datetime
import string
import pandas as pd
import concurrent.futures
import pulp
import io
import unicodedata
import time

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
    "nfl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1180552482&single=true&output=csv",
    "nhl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=401621588&single=true&output=csv",
    "ncaam": "" # Empty to force "Boosts Only" display for CBB
}
# ---------------------------------------------------

# --- Page Configuration ---
st.set_page_config(page_title="Player Boost & Optimizer", layout="wide")

st.title("🏀 🏒 Player Boost & Lineup Optimizer")
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
    """Robust normalization for names with accent removal."""
    if not isinstance(name, str):
        name = str(name)
    n = name.lower().strip()
    try:
        n = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode('utf-8')
    except Exception:
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

def calculate_cbb_custom_rating(row, mapping):
    """Calculates CBB player rating based on specific efficiency weights."""
    stats = {}
    for key, col_name in mapping.items():
        try:
            val = float(row.get(col_name, 0.0))
            if pd.isna(val): val = 0.0
            stats[key] = val
        except:
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

def fetch_letter(session, sport, date_str, letter):
    """Helper to fetch a single letter for a specific date."""
    url = (
        f"https://web.realsports.io/players/sport/{sport}/search"
        f"?day={date_str}&includeNoOneOption=false"
        f"&query={letter}&searchType=ratingLineup"
    )
    try:
        r = session.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("players", [])
    except:
        st.write("Passed")
        pass
    return []

def fetch_data_for_sport(sport, target_date):
    """Fetches player data from API using strictly current fantasy day."""
    session = requests.Session()
    session.headers.update(HEADERS)
    sport_data = []
    
    # Strict Date Strategy: Only check the specific date requested
    active_date_str = str(target_date)

    # Parallel Fetch Alphabet + Accented Characters
    letters = list(string.ascii_uppercase) + ['Š', 'Ć', 'Č', 'Ž', 'Đ', 'Ö', 'Ä', 'Ü', 'Å', 'Ø']
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_req = {executor.submit(fetch_letter, session, sport, active_date_str, letter): letter for letter in letters}
        
        for future in concurrent.futures.as_completed(future_to_req):
            try:
                players = future.result()
                if not players: continue

                for player in players:
                    raw_injury = player.get('injuryStatus')
                    injury_status = str(raw_injury).strip().upper() if raw_injury else ""
                    
                    if injury_status in ['O', 'OUT', 'IR', 'INJ']: 
                        continue

                    position = player.get('position', 'Unknown')
                    if sport.lower() == 'nhl' and position == 'G':
                        continue

                    full_name = f"{player['firstName']} {player['lastName']}"
                    
                    boost_value = 0.0 
                    details = player.get("details")
                    
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

def run_optimization(df, num_lineups=1):
    """Runs Assignment Problem solver."""
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
    
    for j in slot_indices:
        prob += pulp.lpSum([x[i][j] for i in player_indices]) == 1
    for i in player_indices:
        prob += pulp.lpSum([x[i][j] for j in slot_indices]) == y[i]
    prob += pulp.lpSum([y[i] for i in player_indices]) == NUM_SLOTS

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
                            "Position": p_pos,
                            "Player Name": p_name,
                            "Injury": p_injury,
                            "Projection": p_proj_orig,
                            "Base Boost": p_boost,
                            "Eff. Boost": f"{eff_boost:.2f}x",
                            "Points": final_pts
                        })
            lineup_df = pd.DataFrame(lineup_data).sort_values(by="Slot")
            generated_lineups.append(lineup_df)
            prob += pulp.lpSum([y[i] for i in selected_player_indices]) <= NUM_SLOTS - 1
        else:
            break
    return generated_lineups

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Boost Data")
    selected_sport = st.selectbox("Select League", ["nba", "nhl", "nfl", "ncaam"], index=0)
    
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
        elif sport_key == "ncaam":
             st.info("ℹ️ No auto-projections for NCAAM. Fetching boosts only.")
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
    
    # NEW: CBB Minute Filter
    min_proj_min = 0
    if selected_sport == 'ncaam':
        st.subheader("CBB Filters")
        min_proj_min = st.slider("Min Projected Minutes", 0, 40, 5, help="Filter out players with very low projected minutes.")


# --- Main Logic ---

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
        # Convert list to DF immediately to handle duplicates
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

# 3. Projection Logic
if 'proj_df' not in st.session_state:
    st.session_state.proj_df = None

df_proj = st.session_state.proj_df
df_proj_copy = None

if input_method == "Use Global/Public Projections" and current_proj_url:
    if st.session_state.proj_df is None:
         df_proj_copy, _ = load_projections_from_url(current_proj_url)
         if df_proj_copy is not None:
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

# 4. Merging & Optimization
df_boosts = boost_store.get()

proceed = False
if not df_boosts.empty:
    proceed = True

if proceed:
    # Filter boost data for the selected sport immediately
    df_boosts = standardize_boost_columns(df_boosts)
    sport_boosts = df_boosts[df_boosts['Sport'].str.upper() == selected_sport.upper()].copy()

    # --- CASE A: HAVE PROJECTIONS ---
    if df_proj_copy is not None and not df_proj_copy.empty:
        df_proj = df_proj_copy.copy()
        
        # Standardize Projection Cols
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

        points_col = None 
        
        # --- SPECIAL NBA RATING LOGIC ---
        if selected_sport == "nba":
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
                st.success("✅ NBA Custom Efficiency Rating Applied")
        
        # --- SPECIAL CBB RATING LOGIC ---
        if selected_sport == "ncaam":
            # Add minute columns to map for scaling
            proj_min_col = find_col(df_proj.columns, ["proj min", "projected minutes", "proj_min", "p_min", "projected"])
            other_cols = [c for c in df_proj.columns if c != proj_min_col]
            avg_min_col = find_col(other_cols, ["avg min", "minutes", "min", "mpg"])

            cbb_cols_map = {
                "fgm": find_col(df_proj.columns, ["fieldGoalsMade", "fgm"]),
                "fga": find_col(df_proj.columns, ["fieldGoalsAttempted", "fga"]),
                "3pm": find_col(df_proj.columns, ["threePointsMade", "3pm", "3pt"]),
                "ftm": find_col(df_proj.columns, ["freeThrowsMade", "ftm"]),
                "fta": find_col(df_proj.columns, ["freeThrowsAttempted", "fta"]),
                "reb": find_col(df_proj.columns, ["rebounds", "reb", "tot reb"]),
                "ast": find_col(df_proj.columns, ["assists", "ast"]),
                "stl": find_col(df_proj.columns, ["steals", "stl"]),
                "blk": find_col(df_proj.columns, ["blocks", "blk"]),
                "to":  find_col(df_proj.columns, ["turnovers", "to", "tov"]),
                "proj_min": proj_min_col,
                "avg_min": avg_min_col
            }
            
            stat_keys_check = ['fgm', 'fga', '3pm', 'ftm', 'fta', 'reb', 'ast', 'stl', 'blk', 'to']
            missing_keys = [k for k in stat_keys_check if cbb_cols_map[k] is None]
            
            if not missing_keys:
                df_proj['Calculated_Rating'] = df_proj.apply(lambda row: calculate_cbb_custom_rating(row, cbb_cols_map), axis=1)
                points_col = 'Calculated_Rating'
                st.success("✅ CBB Custom Efficiency Rating Applied (Scaled by Minutes)")
                
                # --- NEW: Filter by Projected Minutes if column found ---
                if proj_min_col and min_proj_min > 0:
                    df_proj[proj_min_col] = pd.to_numeric(df_proj[proj_min_col], errors='coerce').fillna(0)
                    initial_cbb_count = len(df_proj)
                    df_proj = df_proj[df_proj[proj_min_col] >= min_proj_min]
                    st.info(f"🏀 Filtered out {initial_cbb_count - len(df_proj)} players with < {min_proj_min} min.")
            else:
                 pass # Fallback to fpts or skip custom rating

        if selected_sport == "nhl" and not points_col:
            points_col = find_col(df_proj.columns, ["ppg_projection"])

        if not points_col:
            points_col = find_col(df_proj.columns, ["ppg", "fantasy", "proj", "fpts", "pts", "avg", "fp"])

        pos_col = find_col(df_proj.columns, ["pos", "position"])
        slate_col = find_col(df_proj.columns, ["slate", "contest", "label"])
        game_col = find_col(df_proj.columns, ["game", "matchup", "match"])
        team_col = find_col(df_proj.columns, ["team", "tm", "squad"])
        opp_col = find_col(df_proj.columns, ["opp", "opponent", "vs"])
        
        injury_csv_col = find_col(df_proj.columns, ["injury", "status"])
        if injury_csv_col:
            df_proj = df_proj[~df_proj[injury_csv_col].astype(str).str.strip().str.upper().isin(['O', 'OUT', 'IR', 'INJ'])]

        if not game_col and not (team_col and opp_col):
            for col in df_proj.columns:
                sample = df_proj[col].dropna().astype(str).head(5)
                if any(" v " in x.lower() or " vs " in x.lower() or "@" in x for x in sample):
                    game_col = col
                    break

        if name_col and points_col:
            if selected_sport == 'nhl':
                rl_col = find_col(df_proj.columns, ["reg_line"])
                pp_col = find_col(df_proj.columns, ["pp_line"])
                if rl_col and pp_col:
                    df_proj[rl_col] = pd.to_numeric(df_proj[rl_col], errors='coerce')
                    df_proj[pp_col] = pd.to_numeric(df_proj[pp_col], errors='coerce')
                    df_proj = df_proj[(df_proj[rl_col] == 1) & (df_proj[pp_col] == 1)]

            sport_boosts['join_key'] = sport_boosts['Player Name'].apply(normalize_name)
            df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
            
            merged_df = pd.merge(sport_boosts, df_proj, on='join_key', how='right')
            
            merged_df['Boost'] = merged_df['Boost'].fillna(0.0)
            merged_df['Player Name'] = merged_df['Player Name'].fillna(merged_df[name_col])
            if pos_col:
                merged_df['Position'] = merged_df['Position'].fillna(merged_df[pos_col])
            merged_df['Injury'] = merged_df['Injury'].fillna('')
            merged_df['Sport'] = merged_df['Sport'].fillna(selected_sport.upper())

            merged_df = merged_df[~merged_df['Injury'].astype(str).str.strip().str.upper().isin(['O', 'OUT', 'IR', 'INJ'])]

            if not merged_df.empty:
                merged_df = merged_df.rename(columns={points_col: 'Projection'})
                if pos_col:
                    merged_df['Position'] = merged_df[pos_col].fillna(merged_df['Position'])
                merged_df['Position'] = merged_df['Position'].apply(normalize_position)
                
                if slate_col:
                    merged_df['Slate'] = merged_df[slate_col].fillna("ALL")
                else:
                    merged_df['Slate'] = "ALL"
                    
                if team_col and opp_col:
                    merged_df['Game'] = merged_df.apply(lambda x: " vs ".join(sorted([str(x[team_col]), str(x[opp_col])])), axis=1)
                elif game_col:
                    merged_df['Game'] = merged_df[game_col].fillna("Unknown")
                else:
                    merged_df['Game'] = "ALL"

                merged_df['Projection'] = pd.to_numeric(merged_df['Projection'], errors='coerce').fillna(0)
                merged_df = merged_df[merged_df['Projection'] > 0]

                def get_bias_multiplier(row):
                    if row['Position'] in ['WR', 'RB']: return wr_rb_bonus
                    if row['Position'] == 'QB': return qb_penalty
                    return 1.0

                merged_df['Bias'] = merged_df.apply(get_bias_multiplier, axis=1)
                merged_df['Adjusted Projection'] = merged_df['Projection'] * merged_df['Bias']
                merged_df['Optimization Score'] = (merged_df['Boost'] + 2.0) * merged_df['Adjusted Projection']
                merged_df['Est. Score'] = merged_df['Boost'] * merged_df['Projection']

                tab1, tab2, tab3 = st.tabs(["📊 Data Browser", "💎 Best Value", "🚀 Lineup Optimizer"])
                
                with tab1:
                    cols = ['Sport', 'Slate', 'Game', 'Position', 'Player Name', 'Injury', 'Boost', 'Projection', 'Optimization Score']
                    if selected_sport == 'ncaam' and 'proj_min_col' in locals() and proj_min_col:
                         cols.append(proj_min_col)
                    cols = [c for c in cols if c in merged_df.columns]
                    st.dataframe(merged_df[cols].sort_values('Optimization Score', ascending=False), use_container_width=True)

                with tab2:
                    value_cols = ['Position', 'Player Name', 'Injury', 'Boost', 'Projection', 'Optimization Score']
                    st.dataframe(
                        merged_df[value_cols].sort_values('Optimization Score', ascending=False).head(50), 
                        use_container_width=True,
                        column_config={"Optimization Score": st.column_config.NumberColumn(format="%.2f")}
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
                    
                    # --- NEW: EXCLUDE PLAYERS MULTISELECT ---
                    all_player_names = sorted(merged_df['Player Name'].dropna().unique().tolist())
                    excluded_players = st.multiselect(
                        "Filter by Player (Exclude):",
                        all_player_names,
                        default=[],
                        placeholder="Search for players to exclude..."
                    )
                    
                    filtered_df = merged_df.copy()
                    
                    # Apply Exclusions
                    if excluded_players:
                         filtered_df = filtered_df[~filtered_df['Player Name'].isin(excluded_players)]

                    if "ALL" not in selected_slates:
                        filtered_df = filtered_df[filtered_df['Slate'].isin(selected_slates)]
                    if "ALL" not in selected_games:
                        filtered_df = filtered_df[filtered_df['Game'].isin(selected_games)]
                        
                    st.caption(f"Pool Size: {len(filtered_df)} Players")

                    if st.button("Generate Optimal Lineups"):
                        lineups = run_optimization(filtered_df, num_lineups)
                        if lineups:
                            for idx, lineup in enumerate(lineups):
                                total_score = lineup['Points'].sum()
                                q_players = lineup[lineup['Injury'].astype(str).str.startswith('Q', na=False)]['Player Name'].tolist()
                                warn_icon = "⚠️ " if q_players else ""
                                
                                with st.expander(f"{warn_icon}Lineup #{idx+1} | Total Score: {total_score:.2f}", expanded=(idx==0)):
                                    if q_players:
                                        st.warning(f"**Questionable Status:** {', '.join(q_players)}")
                                    st.dataframe(
                                        lineup.drop(columns=['Injury']), 
                                        column_config={
                                            "Points": st.column_config.NumberColumn(format="%.2f"),
                                            "Projection": st.column_config.NumberColumn(format="%.2f"),
                                        },
                                        use_container_width=True,
                                        hide_index=True
                                    )
                        else:
                            st.error("Could not generate lineup.")
    else:
        # --- CASE B: BOOSTS ONLY (No Projections) ---
        if not sport_boosts.empty:
            st.subheader(f"Raw Boosts for {selected_sport.upper()} (No Projections Found)")
            st.write("Since no projections CSV is available, showing just the raw API boost data.")
            st.dataframe(
                sport_boosts[['Player Name', 'Boost', 'Position', 'Injury', 'Date']], 
                use_container_width=True
            )
        else:
            st.info("Waiting for data fetch...")
else:
    st.write("Waiting for data fetch...")
