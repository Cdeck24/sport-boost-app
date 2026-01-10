import streamlit as st
import requests
import datetime
import string
import pandas as pd
import concurrent.futures
import pulp
import io

# --- ⬇️ PASTE YOUR GOOGLE SHEET CSV LINKS HERE ⬇️ ---
SPORT_PROJECTION_URLS = {
    "nba": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=0&single=true&output=csv", 
    "nfl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1180552482&single=true&output=csv",
    "nhl": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=401621588&single=true&output=csv",
    "mlb": "",
    "ncaam": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSnuLbwe_6u39hsVARUjkjA6iDbg8AFSkr2BBUoMqZBPBVFU-ilTjJ5lOvJ5Sxq-d28CohPCVKJYA01/pub?gid=1476893875&single=true&output=csv"
}
# ---------------------------------------------------

# --- Page Configuration ---
st.set_page_config(page_title="Player Boost & Optimizer", layout="wide")

st.title("🏀 🏒 Player Boost & Lineup Optimizer")
st.markdown("""
This tool fetches live **Boost Multipliers** from the API and allows you to merge them with 
**Fantasy Projections** to find the highest-scoring lineups using **Slot-Based Optimization**.
""")

# --- Helper Functions ---
def normalize_name(name):
    """Robust normalization for names (especially NFL)."""
    n = str(name).lower()
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v', ' jr.', ' sr.']
    for suffix in suffixes:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
    # Remove punctuation like periods in 'A.J.' -> 'aj'
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
    """Finds the first column that matches any keyword in the list."""
    for col in columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

def fetch_data_for_sport(sport):
    """
    Fetches player data. 
    SMART NFL LOGIC: Automatically searches the next 7 days for data if today returns nothing.
    """
    letters = string.ascii_uppercase
    session = requests.Session()
    sport_data = []

    # 1. Determine Date Strategy
    target_dates = [datetime.date.today()]
    
    # If NFL, we probe the next 7 days because games aren't daily
    if sport.lower() == 'nfl':
        target_dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(7)]

    # 2. Find the correct date (Probe)
    active_date_str = str(datetime.date.today())
    
    if len(target_dates) > 1:
        found_date = False
        for d in target_dates:
            d_str = str(d)
            # Probe with a common letter to see if data exists
            probe_url = (
                f"https://api.real.vg/players/sport/{sport}/search"
                f"?day={d_str}&includeNoOneOption=false"
                f"&query=S&searchType=ratingLineup"
            )
            try:
                r = session.get(probe_url, timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("players"):
                        active_date_str = d_str
                        found_date = True
                        break
            except:
                pass
        
        if not found_date:
            # Fallback to today if probing failed, but usually means no games
            active_date_str = str(datetime.date.today())

    # 3. Fetch Full Alphabet for the active date
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
                # -- INJURY CHECK --
                injury_status = player.get('injuryStatus', '')
                if injury_status == 'O':
                    continue # Skip OUT players entirely

                position = player.get('position', 'Unknown')

                # -- NHL GOALIE FILTER --
                if sport.lower() == 'nhl' and position == 'G':
                    continue

                full_name = f"{player['firstName']} {player['lastName']}"
                boost_value = 0.0 # Default to 0.0 (No Boost)
                
                details = player.get("details")
                
                # Check for explicit boost
                if details and isinstance(details, list) and len(details) > 0 and "text" in details[0]:
                    text = details[0]["text"]
                    boost_str = text.replace("x", "").replace("+", "").strip()
                    try:
                        boost_value = float(boost_str) 
                    except ValueError:
                        pass 
                
                # Always add the player, even if boost is just 0.0
                sport_data.append({
                    "Sport": sport.upper(),
                    "Player Name": full_name,
                    "Position": position,
                    "Boost": boost_value,
                    "Date": active_date_str,
                    "Injury": injury_status # Store status for display/warning later
                })
        except requests.RequestException:
            continue
            
    return sport_data

def run_optimization(df, num_lineups=1):
    """Runs Assignment Problem solver."""
    SLOT_ADDERS = [2.0, 1.8, 1.6, 1.4, 1.2]
    NUM_SLOTS = len(SLOT_ADDERS)
    
    # Safety Check: Do we have enough players?
    if len(df) < NUM_SLOTS:
        return None

    df = df.sort_values('Optimization Score', ascending=False)
    df = df.drop_duplicates(subset=['Player Name'], keep='first').reset_index(drop=True)
    
    # Double check after dedup
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
                            "Injury": p_injury, # Include in output
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
    st.header("1. Fetch Boosts")
    selected_sports = st.multiselect("Select Leagues", ["ncaam", "nba", "nhl", "mlb", "nfl"], default=["nba"])
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
        if len(selected_sports) == 1:
            sport_key = selected_sports[0].lower()
            url = SPORT_PROJECTION_URLS.get(sport_key)
            if url:
                st.success(f"✅ Connected to {sport_key.upper()} Google Sheet")
                current_proj_url = url
            else:
                st.warning(f"⚠️ No Google Sheet link configured for {sport_key.upper()}.")
        elif len(selected_sports) > 1:
            st.warning("⚠️ Please select only ONE sport to use Global Projections.")
        else:
            st.info("Select a sport to load projections.")
            
    elif input_method == "Upload CSV":
        st.info("Upload CSV with Name, Points (e.g. 'FPTS', 'Proj'), and Position.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    else:
        st.info("Copy table from website and paste here.")
        pasted_text = st.text_area("Paste Data Here", height=150, placeholder="Player Name   FPTS   Position...")

# --- Main Logic ---
if 'boost_data' not in st.session_state:
    st.session_state.boost_data = pd.DataFrame()

if fetch_btn:
    if not selected_sports:
        st.warning("Please select at least one sport.")
    else:
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_sports)) as executor:
            future_to_sport = {executor.submit(fetch_data_for_sport, sport): sport for sport in selected_sports}
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_sport):
                sport = future_to_sport[future]
                try:
                    data = future.result()
                    all_results.extend(data)
                    status_text.text(f"Fetched {sport.upper()}")
                except Exception:
                    pass
                completed_count += 1
                progress_bar.progress(completed_count / len(selected_sports))
        status_text.empty()
        progress_bar.empty()
        
        if all_results:
            st.session_state.boost_data = pd.DataFrame(all_results)
            # Show the date we actually found data for (useful for NFL)
            found_dates = sorted(list(set(r['Date'] for r in all_results if 'Date' in r)))
            date_msg = f" (Date: {found_dates[0]})" if len(found_dates) == 1 else ""
            st.success(f"Fetched {len(st.session_state.boost_data)} players{date_msg}.")
        else:
            st.warning("No boosts found. (Checked next 7 days for NFL).")

if not st.session_state.boost_data.empty:
    df_boosts = st.session_state.boost_data
    df_proj = None
    error_msg = None
    
    # 1. Load Data
    if input_method == "Use Global/Public Projections" and current_proj_url:
        try:
            df_proj = pd.read_csv(current_proj_url)
        except Exception as e:
            error_msg = f"Error reading Global URL: {e}"
    elif uploaded_file:
        try:
            df_proj = pd.read_csv(uploaded_file)
        except Exception as e:
            error_msg = f"Error reading file: {e}"
    elif pasted_text:
        try:
            df_proj = pd.read_csv(io.StringIO(pasted_text), sep="\t")
            if len(df_proj.columns) < 2:
                df_proj = pd.read_csv(io.StringIO(pasted_text), sep=",")
        except Exception as e:
            error_msg = f"Error reading text: {e}"

    # 2. Process Data
    if df_proj is not None:
        try:
            df_proj.columns = [c.strip() for c in df_proj.columns]
            
            # --- Special Logic for Split Names (NFL) ---
            first_name_col = find_col(df_proj.columns, ["first name", "firstname", "first"])
            last_name_col = find_col(df_proj.columns, ["last name", "lastname", "last"])
            
            name_col = None
            if first_name_col and last_name_col:
                df_proj['Calculated_Full_Name'] = df_proj[first_name_col].astype(str) + " " + df_proj[last_name_col].astype(str)
                name_col = 'Calculated_Full_Name'
            else:
                name_col = find_col(df_proj.columns, ["player", "name", "who"])

            # Updated Points Search
            points_col = find_col(df_proj.columns, ["ppg", "fantasy", "proj", "fpts", "pts", "avg", "fp"])
            pos_col = find_col(df_proj.columns, ["pos", "position"])

            if name_col and points_col:
                df_boosts['join_key'] = df_boosts['Player Name'].apply(normalize_name)
                df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                
                merged_df = pd.merge(df_boosts, df_proj, on='join_key', how='inner')
                
                if merged_df.empty:
                    st.error("No players matched! This usually means names didn't match or the date is wrong.")
                    # Debug Info
                    with st.expander("Debug: Show unmatched data"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("First 5 Boost Names:", df_boosts['Player Name'].head().tolist())
                        with col2:
                            st.write("First 5 Projection Names:", df_proj[name_col].head().tolist())
                else:
                    merged_df = merged_df.rename(columns={points_col: 'Projection'})
                    if pos_col:
                        merged_df['Position'] = merged_df[pos_col].fillna(merged_df['Position'])
                    merged_df['Position'] = merged_df['Position'].apply(normalize_position)
                    
                    merged_df['Projection'] = pd.to_numeric(merged_df['Projection'], errors='coerce').fillna(0)
                    merged_df = merged_df[merged_df['Projection'] > 0]

                    tab1, tab2 = st.tabs(["📊 Data Browser", "🚀 Lineup Optimizer"])
                    
                    with tab1:
                        st.markdown("### Player Pool")
                        # Add Injury to display if available
                        cols = ['Sport', 'Position', 'Player Name', 'Injury', 'Boost', 'Projection']
                        st.dataframe(merged_df[cols].sort_values('Boost', ascending=False), use_container_width=True)
                        csv_data = merged_df[cols].to_csv(index=False)
                        st.download_button("Download Data CSV", csv_data, "player_pool.csv", "text/csv")
                    
                    with tab2:
                        st.subheader("Optimizer Settings")
                        use_bias = st.checkbox("Apply NFL Position Prioritization", value=(True if "NFL" in selected_sports else False))
                        
                        if use_bias:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                wr_rb_bonus = st.slider("WR/RB Multiplier (Bonus)", 1.0, 1.5, 1.2, 0.05)
                            with col_b:
                                qb_penalty = st.slider("QB Multiplier (Penalty)", 0.5, 1.0, 0.8, 0.05)
                        else:
                            wr_rb_bonus = 1.0
                            qb_penalty = 1.0

                        def get_bias_multiplier(row):
                            if row['Position'] in ['WR', 'RB']: return wr_rb_bonus
                            if row['Position'] == 'QB': return qb_penalty
                            return 1.0

                        merged_df['Bias'] = merged_df.apply(get_bias_multiplier, axis=1)
                        merged_df['Adjusted Projection'] = merged_df['Projection'] * merged_df['Bias']
                        merged_df['Optimization Score'] = merged_df['Boost'] * merged_df['Adjusted Projection']

                        st.divider()
                        num_lineups = st.slider("Number of Lineups", 1, 10, 3)

                        if st.button("Generate Optimal Lineups"):
                            lineups = run_optimization(merged_df, num_lineups)
                            if lineups:
                                for idx, lineup in enumerate(lineups):
                                    total_score = lineup['Points'].sum()
                                    
                                    # Check for Questionable players
                                    q_players = lineup[lineup['Injury'] == 'Q']['Player Name'].tolist()
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
