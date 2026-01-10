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
    # 1. Lowercase
    n = str(name).lower()
    
    # 2. Remove common suffixes (check for space + suffix to avoid partial matches)
    suffixes = [' jr', ' sr', ' ii', ' iii', ' iv', ' v', ' jr.', ' sr.']
    for suffix in suffixes:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
            break
            
    # 3. Keep only alphanumeric chars (removes spaces, dots, hyphens, apostrophes)
    # Examples: 
    # "A.J. Brown" -> "ajbrown"
    # "JuJu Smith-Schuster" -> "jujusmithschuster"
    return "".join(c for c in n if c.isalnum())

def fetch_data_for_sport(sport, target_date):
    """Fetches player data for a specific sport on a specific date."""
    letters = string.ascii_uppercase
    current_date = str(target_date)
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
                        "Boost": boost_value
                    })
                
        except requests.RequestException:
            continue

    return sport_data

def run_optimization(df, num_lineups=1):
    """
    Runs an Assignment Problem solver.
    Assigns 5 players to 5 specific slots to maximize total score.
    Slot Multipliers: +2.0, +1.8, +1.6, +1.4, +1.2
    """
    # 1. Clean Data: Remove duplicates (keep highest Base Score)
    # We need a temporary base score for sorting duplicates
    df['Temp_Score'] = df['Boost'] * df['Projection']
    df = df.sort_values('Temp_Score', ascending=False)
    df = df.drop_duplicates(subset=['Player Name'], keep='first').reset_index(drop=True)
    
    # Constants
    SLOT_ADDERS = [2.0, 1.8, 1.6, 1.4, 1.2] # The fixed multipliers for Slot 1 to 5
    NUM_SLOTS = len(SLOT_ADDERS)
    
    prob = pulp.LpProblem("SlotOptimizer", pulp.LpMaximize)
    
    # Indices
    player_indices = list(df.index)
    slot_indices = list(range(NUM_SLOTS))
    
    # --- Variables ---
    # x[i][j] = 1 if player i is in slot j
    x = pulp.LpVariable.dicts("x", (player_indices, slot_indices), cat="Binary")
    
    # y[i] = 1 if player i is selected (in ANY slot)
    y = pulp.LpVariable.dicts("y", player_indices, cat="Binary")
    
    # --- Objective ---
    # Maximize sum of ( (Boost + Slot_Adder) * Projection )
    obj_terms = []
    for i in player_indices:
        for j in slot_indices:
            # Calculate points for this specific player in this specific slot
            effective_boost = df.loc[i, 'Boost'] + SLOT_ADDERS[j]
            points = effective_boost * df.loc[i, 'Projection']
            obj_terms.append(points * x[i][j])
            
    prob += pulp.lpSum(obj_terms)
    
    # --- Constraints ---
    
    # 1. Each slot must have exactly 1 player
    for j in slot_indices:
        prob += pulp.lpSum([x[i][j] for i in player_indices]) == 1
        
    # 2. Link x and y: If player is in a slot, y must be 1. If not, y is 0.
    # Also ensures a player can only be in ONE slot max.
    for i in player_indices:
        prob += pulp.lpSum([x[i][j] for j in slot_indices]) == y[i]
        
    # 3. Total players selected must equal Number of Slots (5)
    prob += pulp.lpSum([y[i] for i in player_indices]) == NUM_SLOTS

    generated_lineups = []

    # --- Solve Loop ---
    for n in range(num_lineups):
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob.status] == "Optimal":
            lineup_data = []
            selected_player_indices = []
            
            # Extract result
            for j in slot_indices:
                for i in player_indices:
                    if x[i][j].varValue == 1.0:
                        selected_player_indices.append(i)
                        
                        p_name = df.loc[i, "Player Name"]
                        p_proj = df.loc[i, "Projection"]
                        p_boost = df.loc[i, "Boost"]
                        slot_add = SLOT_ADDERS[j]
                        eff_boost = p_boost + slot_add
                        final_pts = eff_boost * p_proj
                        
                        lineup_data.append({
                            "Slot": j + 1,
                            "Slot Bonus": f"+{slot_add}x",
                            "Player Name": p_name,
                            "Projection": p_proj,
                            "Base Boost": p_boost,
                            "Eff. Boost": f"{eff_boost:.2f}x",
                            "Points": final_pts
                        })
            
            # Create DF and Sort by Slot
            lineup_df = pd.DataFrame(lineup_data)
            lineup_df = lineup_df.sort_values(by="Slot")
            generated_lineups.append(lineup_df)
            
            # Constraint: Exclude this specific SET of players from appearing again
            # We enforce that the sum of y variables for these specific players must be <= 4
            prob += pulp.lpSum([y[i] for i in selected_player_indices]) <= NUM_SLOTS - 1
            
        else:
            break

    return generated_lineups

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Fetch Boosts")
    selected_sports = st.multiselect(
        "Select Leagues",
        ["ncaam", "nba", "nhl", "mlb", "nfl"], 
        default=["nba"]
    )
    
    # Added Date Picker
    target_date = st.date_input("Select Date", datetime.date.today())
    
    fetch_btn = st.button("Fetch Live Boosts")

    st.header("2. Upload Projections")
    st.info("Upload a CSV with columns: `Player Name` and `Fantasy Points`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# --- Main Logic ---

# Initialize session state
if 'boost_data' not in st.session_state:
    st.session_state.boost_data = pd.DataFrame()

# Step 1: Fetch Data
if fetch_btn:
    if not selected_sports:
        st.warning("Please select at least one sport.")
    else:
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_sports)) as executor:
            # Pass the target_date to the fetch function
            future_to_sport = {executor.submit(fetch_data_for_sport, sport, target_date): sport for sport in selected_sports}
            
            completed_count = 0
            total_sports = len(selected_sports)
            
            for future in concurrent.futures.as_completed(future_to_sport):
                sport = future_to_sport[future]
                try:
                    data = future.result()
                    all_results.extend(data)
                    status_text.text(f"Fetched {sport.upper()}")
                except Exception:
                    pass
                completed_count += 1
                progress_bar.progress(completed_count / total_sports)

        status_text.empty()
        progress_bar.empty()
        
        if all_results:
            st.session_state.boost_data = pd.DataFrame(all_results)
            st.success(f"Fetched {len(st.session_state.boost_data)} players for {target_date}.")
        else:
            st.warning(f"No boosts found for {target_date}.")

# Step 2: Merge & Display
if not st.session_state.boost_data.empty:
    df_boosts = st.session_state.boost_data
    
    # -- Handle Projections Merge --
    if uploaded_file is not None:
        try:
            df_proj = pd.read_csv(uploaded_file)
            
            # Normalize column names
            df_proj.columns = [c.strip() for c in df_proj.columns]
            
            # Identify critical columns
            name_col = next((c for c in df_proj.columns if "player" in c.lower()), None)
            points_col = next((c for c in df_proj.columns if "fantasy" in c.lower()), None)

            if name_col and points_col:
                # Normalize names for merging
                df_boosts['join_key'] = df_boosts['Player Name'].apply(normalize_name)
                df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                
                # Merge
                merged_df = pd.merge(df_boosts, df_proj, on='join_key', how='inner')
                
                if merged_df.empty:
                    st.error("No players matched! This is usually due to name spelling differences.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.warning("Boost Names (First 5)")
                        st.write(df_boosts['Player Name'].head().tolist())
                    with col2:
                        st.warning("Projection Names (First 5)")
                        st.write(df_proj[name_col].head().tolist())
                else:
                    # Standardize Projection Column Name for Optimizer
                    merged_df = merged_df.rename(columns={points_col: 'Projection'})
                    
                    # Filter: Remove players with 0, empty, or invalid projections
                    merged_df['Projection'] = pd.to_numeric(merged_df['Projection'], errors='coerce').fillna(0)
                    merged_df = merged_df[merged_df['Projection'] > 0]

                    # Calculate a "Base Score" just for sorting the Data Browser list
                    # (Note: Actual score depends on slot, this is just for reference)
                    merged_df['Base Score (No Slot)'] = merged_df['Boost'] * merged_df['Projection']
                    
                    final_df = merged_df.sort_values(by="Base Score (No Slot)", ascending=False)
                    
                    # --- TABS INTERFACE ---
                    tab1, tab2 = st.tabs(["📊 Data Browser", "🚀 Lineup Optimizer"])
                    
                    with tab1:
                        st.markdown("### Player Pool")
                        cols_to_show = ['Sport', 'Player Name', 'Boost', 'Projection', 'Base Score (No Slot)']
                        st.dataframe(final_df[cols_to_show], use_container_width=True)
                    
                    with tab2:
                        st.subheader("Optimizer Settings")
                        st.markdown("""
                        **Slot Rules:**
                        - **Slot 1:** +2.0x
                        - **Slot 2:** +1.8x
                        - **Slot 3:** +1.6x
                        - **Slot 4:** +1.4x
                        - **Slot 5:** +1.2x
                        """)
                        
                        num_lineups = st.slider("Number of Lineups to Generate", 1, 10, 3)

                        if st.button("Generate Optimal Lineups"):
                            # Use the entire pool for optimization
                            lineups = run_optimization(final_df, num_lineups)
                            
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
                                st.error("Could not generate a lineup. Ensure you have enough players.")

            else:
                st.error("Could not find 'Player' or 'Fantasy' columns in your CSV.")
                st.write("Columns found:", df_proj.columns.tolist())
                
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
    else:
        st.info("Upload a CSV to enable Projections & Optimization.")
        st.dataframe(df_boosts, use_container_width=True)

else:
    st.write("Waiting for data fetch...")
