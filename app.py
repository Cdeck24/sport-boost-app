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
**Fantasy Projections** to find the highest-scoring lineups.
""")

# --- Helper Functions ---
def normalize_name(name):
    """Simple helper to normalize names for better matching."""
    return str(name).lower().strip().replace(".", "").replace("'", "")

def fetch_data_for_sport(sport):
    """Fetches player data for a specific sport."""
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

def run_optimization(df, roster_size, num_lineups=1):
    """
    Runs a linear programming solver to find the optimal lineup(s).
    df: DataFrame containing 'Player Name' and 'Total Score'
    Returns a list of DataFrames (one for each lineup).
    """
    # 1. Clean Data: Remove duplicates (keep highest score because df is sorted)
    df = df.drop_duplicates(subset=['Player Name'], keep='first').reset_index(drop=True)
    
    prob = pulp.LpProblem("FantasyOptimizer", pulp.LpMaximize)
    
    player_indices = list(df.index)
    
    # Decision variable: 1 if player is selected, 0 otherwise
    player_vars = pulp.LpVariable.dicts("Player", player_indices, cat="Binary")
    
    # Objective: Maximize Total Score
    prob += pulp.lpSum([df.loc[i, "Total Score"] * player_vars[i] for i in player_indices])
    
    # Constraint: Roster Size
    prob += pulp.lpSum([player_vars[i] for i in player_indices]) == roster_size

    generated_lineups = []

    # Loop to generate multiple lineups
    for i in range(num_lineups):
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if pulp.LpStatus[prob.status] == "Optimal":
            selected_indices = [idx for idx in player_indices if player_vars[idx].varValue == 1.0]
            
            # Store this lineup
            lineup_df = df.loc[selected_indices].copy()
            lineup_df["Lineup #"] = i + 1
            generated_lineups.append(lineup_df)
            
            # Add constraint to prevent this specific combination from being picked again
            # Constraint: Sum of variables for these specific players must be <= roster_size - 1
            prob += pulp.lpSum([player_vars[idx] for idx in selected_indices]) <= roster_size - 1
        else:
            break # No more feasible solutions

    return generated_lineups

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("1. Fetch Boosts")
    selected_sports = st.multiselect(
        "Select Leagues",
        ["ncaam", "nba", "nhl", "mlb", "nfl"], 
        default=["nba"]
    )
    fetch_btn = st.button("Fetch Live Boosts")

    st.header("2. Upload Projections")
    st.info("Upload a CSV with columns: `Player Name` and `Fantasy Points`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# --- Main Logic ---

# Initialize session state to hold data across re-runs
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
            future_to_sport = {executor.submit(fetch_data_for_sport, sport): sport for sport in selected_sports}
            
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
            st.success(f"Fetched {len(st.session_state.boost_data)} players.")
        else:
            st.warning("No boosts found.")

# Step 2: Merge & Display
if not st.session_state.boost_data.empty:
    df_boosts = st.session_state.boost_data
    
    # -- Handle Projections Merge --
    if uploaded_file is not None:
        try:
            df_proj = pd.read_csv(uploaded_file)
            
            # Normalize column names for easier matching
            df_proj.columns = [c.strip() for c in df_proj.columns]
            
            # Identify critical columns (Case insensitive partial match)
            name_col = next((c for c in df_proj.columns if "player" in c.lower()), None)
            points_col = next((c for c in df_proj.columns if "fantasy" in c.lower()), None)

            if name_col and points_col:
                # Normalize names for merging
                df_boosts['join_key'] = df_boosts['Player Name'].apply(normalize_name)
                df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                
                # Merge
                merged_df = pd.merge(df_boosts, df_proj, on='join_key', how='inner')
                
                # Calculate Total Score
                merged_df['Total Score'] = merged_df['Boost'] * merged_df[points_col]
                
                # Clean up columns for display
                cols_to_show = ['Sport', 'Player Name', 'Boost', points_col, 'Total Score']
                
                # Sort by Total Score descending (Important for drop_duplicates later)
                final_df = merged_df[cols_to_show].sort_values(by="Total Score", ascending=False)
                
                # --- TABS INTERFACE ---
                tab1, tab2 = st.tabs(["📊 Data Browser", "🚀 Lineup Optimizer"])
                
                with tab1:
                    st.dataframe(final_df, use_container_width=True)
                
                with tab2:
                    st.subheader("Optimizer Settings")
                    
                    # Hardcoded Roster Size as requested
                    ROSTER_SIZE = 5
                    st.markdown(f"**Roster Size:** {ROSTER_SIZE} (Fixed)")
                    st.caption("Duplicates are automatically removed.")

                    col1, col2 = st.columns(2)
                    with col1:
                        num_lineups = st.slider("Number of Lineups to Generate", 1, 10, 1)

                    if st.button("Generate Optimal Lineups"):
                        lineups = run_optimization(final_df, ROSTER_SIZE, num_lineups)
                        
                        if lineups:
                            for idx, lineup in enumerate(lineups):
                                total_score = lineup['Total Score'].sum()
                                with st.expander(f"Lineup #{idx+1} (Total Score: {total_score:.2f})", expanded=(idx==0)):
                                    st.dataframe(lineup, use_container_width=True)
                        else:
                            st.error("Could not generate a lineup. Ensure you have enough players.")

            else:
                st.error("Could not find 'Player' or 'Fantasy' columns in your CSV. Please check your headers.")
                st.write("Columns found:", df_proj.columns.tolist())
                
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            
    else:
        # If no projections uploaded, just show the boosts
        st.info("Upload a CSV to enable Projections & Optimization. Showing raw boosts for now:")
        st.dataframe(df_boosts, use_container_width=True)

else:
    st.write("Waiting for data fetch...")
