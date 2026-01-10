import streamlit as st
import requests
import datetime
import string
import pandas as pd
import concurrent.futures
import pulp

# --- Page Configuration ---
st.set_page_config(page_title="Player Boost & Optimizer", layout="wide")

st.title("🏀 🏒 Player Boost & Lineup Optimizer")
st.markdown("""
This tool fetches live **Boost Multipliers** from the API and allows you to merge them with 
**Fantasy Projections** to find the highest-scoring lineups using **Slot-Based Optimization**.
""")

# --- Helper Functions ---
def normalize_name(name):
    """
    Robust normalization for names (especially NFL).
    Removes suffixes like Jr, III and strips punctuation to ensure 'Patrick Mahomes II' matches 'Patrick Mahomes'.
    """
    n = str(name).lower()
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v', ' jr.', ' sr.']
    for suffix in suffixes:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
    return "".join(c for c in n if c.isalnum())

def normalize_position(pos):
    """Normalizes position strings (e.g., 'Wide Receiver' -> 'WR')."""
    if not pos or pd.isna(pos):
        return "UNKNOWN"
    p = str(pos).upper().strip()
    if "QUARTER" in p or p == "QB": return "QB"
    if "WIDE" in p or "RECEIVER" in p or p == "WR": return "WR"
    if "RUNNING" in p or "BACK" in p or p == "RB" or p == "HB": return "RB"
    if "TIGHT" in p or p == "TE": return "TE"
    return p

def fetch_data_for_sport(sport):
    """Fetches player data for a specific sport (defaults to today)."""
    letters = string.ascii_uppercase
    current_date = str(datetime.date.today())
    sport_data = []
    session = requests.Session()
    
    for letter in letters:
        query = letter
        url = (
            f"https://api.real.vg/players/sport/{sport}/search"
            f"?day={current_date}&includeNoOneOption=false"
            f"&query={query}&searchType=ratingLineup"
        )
        
        try:
            r = session.get(url, timeout=5)
            if r.status_code != 200: continue

            data = r.json()
            players = data.get("players", [])

            if not players: continue

            for player in players:
                full_name = f"{player['firstName']} {player['lastName']}"
                boost_value = None 
                
                # Attempt to get position from API (if available)
                position = player.get('position', 'Unknown')

                details = player.get("details")
                if details and isinstance(details, list) and len(details) > 0 and "text" in details[0]:
                    text = details[0]["text"]
                    boost_str = text.replace("x", "").replace("+", "").strip()
                    try:
                        boost_value = float(boost_str) 
                    except ValueError:
                        pass 
                
                if boost_value is not None:
                    sport_data.append({
                        "Sport": sport.upper(),
                        "Player Name": full_name,
                        "Position": position,
                        "Boost": boost_value
                    })
                
        except requests.RequestException:
            continue

    return sport_data

def run_optimization(df, num_lineups=1):
    """
    Runs an Assignment Problem solver.
    Uses 'Optimization Score' to maximize value.
    Assigns 5 players to 5 specific slots.
    """
    # 1. Clean Data: Remove duplicates (keep highest Optimization Score)
    df = df.sort_values('Optimization Score', ascending=False)
    df = df.drop_duplicates(subset=['Player Name'], keep='first').reset_index(drop=True)
    
    # Constants
    SLOT_ADDERS = [2.0, 1.8, 1.6, 1.4, 1.2]
    NUM_SLOTS = len(SLOT_ADDERS)
    
    prob = pulp.LpProblem("SlotOptimizer", pulp.LpMaximize)
    
    # Indices
    player_indices = list(df.index)
    slot_indices = list(range(NUM_SLOTS))
    
    # --- Variables ---
    x = pulp.LpVariable.dicts("x", (player_indices, slot_indices), cat="Binary")
    y = pulp.LpVariable.dicts("y", player_indices, cat="Binary")
    
    # --- Objective ---
    # Maximize sum of weighted optimization scores
    obj_terms = []
    for i in player_indices:
        for j in slot_indices:
            # We use the Optimization Score (which might have position bias applied)
            # The boost logic applies to the RAW boost, but the bias applies to the base projection
            
            # Re-calculate effective points for the solver using the Adjusted Projection
            # Effective Points = (Boost + SlotBonus) * Adjusted_Projection
            
            # Retrieve parameters
            raw_boost = df.loc[i, 'Boost']
            adj_proj = df.loc[i, 'Adjusted Projection'] # This has the position bias
            slot_add = SLOT_ADDERS[j]
            
            points = (raw_boost + slot_add) * adj_proj
            obj_terms.append(points * x[i][j])
            
    prob += pulp.lpSum(obj_terms)
    
    # --- Constraints ---
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
                        slot_add = SLOT_ADDERS[j]
                        eff_boost = p_boost + slot_add
                        final_pts = eff_boost * p_proj_orig # Display TRUE points, not biased points
                        
                        lineup_data.append({
                            "Slot": j + 1,
                            "Slot Bonus": f"+{slot_add}x",
                            "Position": p_pos,
                            "Player Name": p_name,
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
    # Removed Date Picker
    fetch_btn = st.button("Fetch Live Boosts")

    st.header("2. Upload Projections")
    st.info("Upload CSV with `Player Name`, `Fantasy Points`, and optional `Position`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

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
            st.success(f"Fetched {len(st.session_state.boost_data)} players.")
        else:
            st.warning("No boosts found.")

if not st.session_state.boost_data.empty:
    df_boosts = st.session_state.boost_data
    
    if uploaded_file is not None:
        try:
            df_proj = pd.read_csv(uploaded_file)
            df_proj.columns = [c.strip() for c in df_proj.columns]
            
            name_col = next((c for c in df_proj.columns if "player" in c.lower()), None)
            points_col = next((c for c in df_proj.columns if "fantasy" in c.lower()), None)
            # Try to find position column in CSV
            pos_col = next((c for c in df_proj.columns if "pos" in c.lower()), None)

            if name_col and points_col:
                df_boosts['join_key'] = df_boosts['Player Name'].apply(normalize_name)
                df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                
                merged_df = pd.merge(df_boosts, df_proj, on='join_key', how='inner')
                
                if merged_df.empty:
                    st.error("No players matched!")
                else:
                    # Rename standard columns
                    merged_df = merged_df.rename(columns={points_col: 'Projection'})
                    
                    # Consolidate Position: Prefer CSV, fallback to API, fallback to Unknown
                    if pos_col:
                        merged_df['Position'] = merged_df[pos_col].fillna(merged_df['Position'])
                    
                    merged_df['Position'] = merged_df['Position'].apply(normalize_position)
                    
                    # Filter invalid projections
                    merged_df['Projection'] = pd.to_numeric(merged_df['Projection'], errors='coerce').fillna(0)
                    merged_df = merged_df[merged_df['Projection'] > 0]

                    tab1, tab2 = st.tabs(["📊 Data Browser", "🚀 Lineup Optimizer"])
                    
                    with tab1:
                        st.markdown("### Player Pool")
                        # Display raw stats
                        merged_df['Est. Score'] = merged_df['Boost'] * merged_df['Projection']
                        cols = ['Sport', 'Position', 'Player Name', 'Boost', 'Projection', 'Est. Score']
                        st.dataframe(merged_df[cols].sort_values('Est. Score', ascending=False), use_container_width=True)
                    
                    with tab2:
                        st.subheader("Optimizer Settings")
                        
                        # -- NFL BIAS SETTINGS --
                        use_bias = st.checkbox("Apply NFL Position Prioritization", value=(True if "NFL" in selected_sports else False))
                        
                        if use_bias:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                wr_rb_bonus = st.slider("WR/RB Multiplier (Bonus)", 1.0, 1.5, 1.2, 0.05, help="Increases WR/RB projection for optimization logic only.")
                            with col_b:
                                qb_penalty = st.slider("QB Multiplier (Penalty)", 0.5, 1.0, 0.8, 0.05, help="Reduces QB projection for optimization logic only.")
                        else:
                            wr_rb_bonus = 1.0
                            qb_penalty = 1.0

                        # Calculate "Adjusted Projection" for the Optimizer
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
                                    with st.expander(f"Lineup #{idx+1} | Total Score: {total_score:.2f}", expanded=(idx==0)):
                                        st.dataframe(
                                            lineup, 
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
                st.error("Could not find required columns in CSV.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload CSV to continue.")
else:
    st.write("Waiting for data fetch...")
