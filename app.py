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
    
    # NEW: Date Picker for explicit control
    target_date = st.date_input("Game Date", get_fantasy_day())
    
    fetch_btn = st.button("Fetch Live Boosts (Update Injuries)")

    # Allow downloading boosts whenever data is present
    if 'boost_data' in st.session_state and not st.session_state.boost_data.empty:
        csv_buffer = st.session_state.boost_data.to_csv(index=False).encode('utf-8')
        # Use target_date in filename to prevent date mismatches
        st.download_button(
            label="💾 Save Current Boosts",
            data=csv_buffer,
            file_name=f"boosts_backup_{target_date}.csv",
            mime="text/csv"
        )

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
        else:
            st.warning(f"⚠️ No URL configured for {sport_key.upper()}.")
    elif input_method == "Upload CSV":
        st.info("Upload CSV with Name, Points (e.g. 'FPTS', 'Proj'), and Position.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            st.session_state.proj_df = pd.read_csv(uploaded_file)
            
    elif input_method == "Paste Text":
        st.info("Copy table from website and paste here.")
        pasted_text = st.text_area("Paste Data Here", height=150, placeholder="Player Name   FPTS   Position...")
        if pasted_text:
            try:
                st.session_state.proj_df = pd.read_csv(io.StringIO(pasted_text), sep="\t")
                if len(st.session_state.proj_df.columns) < 2:
                    st.session_state.proj_df = pd.read_csv(io.StringIO(pasted_text), sep=",")
            except:
                st.error("Could not parse pasted text.")

    st.header("3. Optimization Settings")
    wr_rb_bonus = 1.0
    qb_penalty = 1.0
    num_lineups = st.slider("Number of Lineups", 1, 10, 3)

# --- Main Logic ---

# 1. Initialize Boost Data (Auto-Load from Google Sheet)
if 'boost_data' not in st.session_state:
    try:
        # Load from the Google Sheet URL provided by user
        saved_df = pd.read_csv(SAVED_BOOSTS_URL)
        st.session_state.boost_data = saved_df
        
        # Check Date Consistency
        if 'Date' in saved_df.columns:
            # Try to grab the most common date
            loaded_dates = saved_df['Date'].astype(str).unique()
            if len(loaded_dates) > 0:
                most_common = loaded_dates[0]
                if str(most_common) != str(target_date):
                    st.toast(f"⚠️ Loaded boosts are from {most_common}. Selected Date: {target_date}", icon="📅")
    except Exception as e:
        st.session_state.boost_data = pd.DataFrame(columns=['Sport', 'Player Name', 'Position', 'Boost', 'Date', 'Injury'])
        st.error(f"Could not load Saved Boosts from Google Sheet: {e}")

if 'proj_df' not in st.session_state:
    st.session_state.proj_df = None

# 2. Fetch Live Logic (Merges into Saved Data)
if fetch_btn:
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    try:
        status_text.text(f"Fetching {selected_sport.upper()}...")
        # Now passing target_date to the fetch function
        data = fetch_data_for_sport(selected_sport, target_date)
        all_results.extend(data)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()
    
    if all_results:
        # Map of API results: Name -> Row
        api_data_map = {row['Player Name']: row for row in all_results}
        
        # Get current state (Saved CSV)
        current_df = st.session_state.boost_data.copy()
        
        # We need to construct a new list of rows
        updated_rows = []
        
        # Track which names we have processed
        processed_names = set()
        
        # 1. Update Existing Players in Saved CSV
        for _, row in current_df.iterrows():
            if str(row.get('Sport', '')).upper() == selected_sport.upper():
                name = row['Player Name']
                processed_names.add(name)
                
                # Start with saved data (Trusted Boost)
                new_row = row.to_dict()
                
                # If API has this player, update INJURY status (Trusted Status)
                if name in api_data_map:
                    api_row = api_data_map[name]
                    new_row['Injury'] = api_row.get('Injury', '')
                    # We do NOT update Boost from API here to avoid 0.0 override
                
                updated_rows.append(new_row)
            else:
                updated_rows.append(row.to_dict())
                
        # 2. Add New Players found in API but not in CSV
        for name, row in api_data_map.items():
            if name not in processed_names:
                updated_rows.append(row)
                
        st.session_state.boost_data = pd.DataFrame(updated_rows)
        st.success(f"Updated Injury Statuses for {selected_sport.upper()}!")
    else:
        st.warning(f"No active data found in API for {selected_sport.upper()}. Using Saved Boosts.")

# Check proceed condition
proceed = False
if not st.session_state.boost_data.empty:
    proceed = True
if st.session_state.proj_df is not None:
    proceed = True

if proceed:
    df_boosts = st.session_state.boost_data
    df_proj = st.session_state.proj_df
    
    df_proj_copy = None
    error_msg = None
    source_type = None
    
    if input_method == "Use Global/Public Projections" and current_proj_url:
        if st.session_state.proj_df is None:
             df_proj_copy, source_type = load_projections_from_url(current_proj_url)
             if df_proj_copy is not None:
                 st.session_state.proj_df = df_proj_copy
             else:
                 error_msg = source_type
        else:
             df_proj_copy = st.session_state.proj_df
             
    elif uploaded_file:
        try: df_proj_copy = pd.read_csv(uploaded_file)
        except Exception as e: error_msg = f"Error reading file: {e}"
    elif pasted_text:
        try:
            df_proj_copy = pd.read_csv(io.StringIO(pasted_text), sep="\t")
            if len(df_proj_copy.columns) < 2: df_proj_copy = pd.read_csv(io.StringIO(pasted_text), sep=",")
        except Exception as e: error_msg = f"Error reading text: {e}"
    elif st.session_state.proj_df is not None:
        df_proj_copy = st.session_state.proj_df

    if df_proj_copy is not None:
        try:
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
                
                missing_keys = [k for k, v in nba_cols_map.items() if v is None]
                if not missing_keys:
                    df_proj['Calculated_Rating'] = df_proj.apply(
                        lambda row: calculate_nba_custom_rating(row, nba_cols_map), axis=1
                    )
                    points_col = 'Calculated_Rating'
                    st.success("✅ Applied Custom NBA Efficiency Formula using raw stats.")
                else:
                    st.error(f"❌ NBA Custom Rating Failed. Missing stats for: {', '.join(missing_keys)}")
            
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
                        initial_count = len(df_proj)
                        df_proj = df_proj[(df_proj[rl_col] == 1) & (df_proj[pp_col] == 1)]
                        if len(df_proj) < initial_count:
                            st.info(f"🏒 **NHL Line Filter Active:** Kept {len(df_proj)} players (Line 1 & PP 1).")

                df_boosts['join_key'] = df_boosts['Player Name'].apply(normalize_name)
                df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                
                # Right Join: Use CSV as master list to ensure all projections are present
                merged_df = pd.merge(df_boosts, df_proj, on='join_key', how='right')
                
                # Fill NAs
                merged_df['Boost'] = merged_df['Boost'].fillna(0.0)
                merged_df['Player Name'] = merged_df['Player Name'].fillna(merged_df[name_col])
                if pos_col:
                    merged_df['Position'] = merged_df['Position'].fillna(merged_df[pos_col])
                merged_df['Injury'] = merged_df['Injury'].fillna('')
                merged_df['Sport'] = merged_df['Sport'].fillna(selected_sport.upper())

                merged_df = merged_df[~merged_df['Injury'].astype(str).str.strip().str.upper().isin(['O', 'OUT', 'IR', 'INJ'])]

                if merged_df.empty:
                    st.error("No players matched! Check names/dates.")
                else:
                    merged_df = merged_df.rename(columns={points_col: 'Projection'})
                    if pos_col:
                        merged_df['Position'] = merged_df[pos_col].fillna(merged_df['Position'])
                    merged_df['Position'] = merged_df['Position'].apply(normalize_position)
                    
                    if slate_col:
                        merged_df['Slate'] = merged_df[slate_col].fillna("ALL")
                    else:
                        merged_df['Slate'] = "ALL"
                        
                    if team_col and opp_col:
                        merged_df['Game'] = merged_df.apply(
                            lambda x: " vs ".join(sorted([str(x[team_col]), str(x[opp_col])])), axis=1
                        )
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
                        st.markdown("### Player Pool (Raw Data)")
                        cols = ['Sport', 'Slate', 'Game', 'Position', 'Player Name', 'Injury', 'Boost', 'Projection', 'Optimization Score']
                        cols = [c for c in cols if c in merged_df.columns]
                        # Sort by Optimization Score so 0-boost players appear correctly based on their projection value
                        st.dataframe(merged_df[cols].sort_values('Optimization Score', ascending=False), use_container_width=True)
                        csv_data = merged_df[cols].to_csv(index=False)
                        st.download_button("Download Data CSV", csv_data, "player_pool.csv", "text/csv")

                    with tab2:
                        st.markdown("### Top Value Plays")
                        value_cols = ['Position', 'Player Name', 'Injury', 'Boost', 'Projection', 'Optimization Score']
                        st.dataframe(
                            merged_df[value_cols].sort_values('Optimization Score', ascending=False).head(50), 
                            use_container_width=True,
                            column_config={"Optimization Score": st.column_config.NumberColumn(format="%.2f")}
                        )

                    with tab3:
                        st.subheader("Generate Lineups")
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
                                            column_config={
                                                "Points": st.column_config.NumberColumn(format="%.2f"),
                                                "Projection": st.column_config.NumberColumn(format="%.2f"),
                                            },
                                            use_container_width=True,
                                            hide_index=True
                                        )
                            else:
                                st.error("Could not generate lineup. Not enough players matched (need at least 5).")
            else:
                st.error(f"Could not find Name or Points columns. Found: {df_proj.columns.tolist()}")
        except Exception as e:
            st.error(f"Error processing data: {e}")
    elif error_msg:
         st.error(error_msg)
    else:
        st.info("Waiting for Projections (Upload, Paste, or Configure Global URL).")
else:
    st.write("Waiting for data fetch...")
