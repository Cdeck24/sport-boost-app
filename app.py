import streamlit as st
import requests
import datetime
import string
import pandas as pd
import concurrent.futures
import pulp
import io
import unicodedata

# --- ⬇️ CONFIGURATION ⬇️ ---
# PROJECTION SOURCES (Google Sheet Links)
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

def fetch_letter(session, sport, date_str, letter):
    """Helper to fetch a single letter for a specific date."""
    url = (
        f"https://api.real.vg/players/sport/{sport}/search"
        f"?day={date_str}&includeNoOneOption=false"
        f"&query={letter}&searchType=ratingLineup"
    )
    try:
        r = session.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("players", [])
    except:
        pass
    return []

def fetch_data_for_sport(sport, target_date):
    """Fetches player data from API using STRICTLY the selected date."""
    session = requests.Session()
    sport_data = []
    
    # Strict Date Strategy: Only check the specific date requested
    active_date_str = str(target_date)

    # Parallel Fetch Alphabet + Accented Characters
    letters = list(string.ascii_uppercase) + ['Š', 'Ć', 'Č', 'Ž', 'Đ', 'Ö', 'Ä', 'Ü', 'Å', 'Ø']
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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
            
    return sport_data

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
    selected_sport = st.selectbox("Select League", ["nba", "nhl", "nfl"], index=0)
    
    # Simple Fetch Button
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
            current_proj_url = url
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

# --- Main Logic ---

# 1. Fetch Live Logic (Merges into Global Store)
if fetch_btn:
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        status_text.text(f"Fetching {selected_sport.upper()}...")
        # Automatically use today's date (US time)
        fetch_date = get_fantasy_day()
        data = fetch_data_for_sport(selected_sport, fetch_date)
        all_results.extend(data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    if all_results:
        # Map of API results: Name -> Row
        api_data_map = {row['Player Name']: row for row in all_results}
        
        # Get current global state (creates empty DF if none exists)
        current_df = boost_store.get()
        if current_df.empty:
             current_df = pd.DataFrame(columns=['Sport', 'Player Name', 'Position', 'Boost', 'Date', 'Injury'])
        
        current_df = standardize_boost_columns(current_df)
        
        updated_rows = []
        processed_names = set()
        
        # 1. Update Existing Players in Store (Preserve old boosts if API = 0.0)
        for _, row in current_df.iterrows():
            if str(row.get('Sport', '')).upper() == selected_sport.upper():
                name = row['Player Name']
                processed_names.add(name)
                
                new_row = row.to_dict()
                new_row['Date'] = str(fetch_date) # Update date to today
                
                if name in api_data_map:
                    api_row = api_data_map[name]
                    new_row['Injury'] = api_row.get('Injury', '')
                    
                    # Update boost ONLY if API has valid new data > 0.0
                    api_boost = api_row.get('Boost', 0.0)
                    old_boost = row.get('Boost', 0.0)
                    if api_boost > 0.0 and api_boost != old_boost:
                         new_row['Boost'] = api_boost
                
                updated_rows.append(new_row)
            else:
                updated_rows.append(row.to_dict())
                
        # 2. Add New Players found in API
        for name, row in api_data_map.items():
            if name not in processed_names:
                updated_rows.append(row)
                
        # Update Global Store
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
    # Auto-load if missing
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
if not df_boosts.empty and df_proj_copy is not None:
    proceed = True

if proceed:
    df_proj = df_proj_copy.copy()
    
    # Standardize Boost Cols
    df_boosts = standardize_boost_columns(df_boosts)

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

        df_boosts['join_key'] = df_boosts['Player Name'].apply(normalize_name)
        df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
        
        merged_df = pd.merge(df_boosts, df_proj, on='join_key', how='right')
        
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
                cols = [c for c in cols if c in merged_df.columns]
                st.dataframe(merged_df[cols].sort_values('Optimization Score', ascending=False), use_container_width=True)

            with tab2:
                value_cols = ['Position', 'Player Name', 'Injury', 'Boost', 'Projection', 'Optimization Score']
                st.dataframe(
                    merged_df[value_cols].sort_values('Optimization Score', ascending=False).head(50), 
                    use_container_width=True
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
                
                filtered_df = merged_df.copy()
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
                                    use_container_width=True,
                                    hide_index=True
                                )
                    else:
                        st.error("Could not generate lineup.")
    else:
        st.info("Waiting for data...")
else:
    st.write("Waiting for data fetch...")
