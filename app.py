import streamlit as st
import requests
import datetime
import string
import pandas as pd
import concurrent.futures

# --- Page Configuration ---
st.set_page_config(page_title="Player Boost Viewer", layout="wide")

st.title("🏀 🏒 Player Boost Viewer")
st.write("Click the button below to fetch the latest player multipliers directly from the API.")

# --- Fetching Logic (Adapted from your script) ---
def fetch_data_for_sport(sport):
    """
    Fetches player data for a specific sport. 
    Returns a list of dictionaries.
    """
    letters = string.ascii_uppercase
    current_date = str(datetime.date.today())
    sport_data = []
    
    # We can use a session to speed up requests slightly
    session = requests.Session()

    # Progress placeholder for this specific sport (optional text update)
    # Note: interacting with st. elements from threads can be tricky, 
    # so we'll just return the data and handle UI in the main thread.
    
    for letter in letters:
        query = letter
        url = (
            f"https://api.real.vg/players/sport/{sport}/search"
            f"?day={current_date}&includeNoOneOption=false"
            f"&query={query}&searchType=ratingLineup"
        )
        
        try:
            r = session.get(url, timeout=10)
            if r.status_code != 200:
                continue

            data = r.json()
            players = data.get("players", [])

            if not players:
                continue

            for player in players:
                full_name = f"{player['firstName']} {player['lastName']}"
                boost_value = None # Use None for better sorting/filtering later

                details = player.get("details")
                
                if details and isinstance(details, list) and len(details) > 0 and "text" in details[0]:
                    text = details[0]["text"]
                    boost_str = text.replace("x", "").replace("+", "").strip()
                    
                    try:
                        boost_value = float(boost_str) 
                    except ValueError:
                        pass 
                
                # Only add if we actually found a boost value (optional cleanup)
                if boost_value is not None:
                    sport_data.append({
                        "Sport": sport.upper(),
                        "Player Name": full_name,
                        "Boost Value": boost_value
                    })
                
        except requests.RequestException:
            continue

    return sport_data

# --- Main App Interface ---

# 1. User Inputs
selected_sports = st.multiselect(
    "Select Leagues to Fetch",
    ["ncaam", "nba", "nhl", "mlb", "nfl"], # Added a few common ones just in case
    default=["ncaam", "nba", "nhl"]
)

if st.button("Fetch Boosts"):
    if not selected_sports:
        st.warning("Please select at least one sport.")
    else:
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # We use ThreadPoolExecutor to run sports in parallel, just like your original script
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_sports)) as executor:
            future_to_sport = {executor.submit(fetch_data_for_sport, sport): sport for sport in selected_sports}
            
            completed_count = 0
            total_sports = len(selected_sports)
            
            for future in concurrent.futures.as_completed(future_to_sport):
                sport = future_to_sport[future]
                try:
                    data = future.result()
                    all_results.extend(data)
                    status_text.text(f"Finished fetching {sport.upper()}...")
                except Exception as exc:
                    st.error(f"{sport} generated an exception: {exc}")
                
                completed_count += 1
                progress_bar.progress(completed_count / total_sports)

        # clear progress indicators
        status_text.text("Done!")
        progress_bar.progress(100)

        # --- Display Data ---
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Sort by Boost Value descending by default
            df = df.sort_values(by="Boost Value", ascending=False)
            
            st.success(f"Found {len(df)} players with boosts.")
            
            # Interactive Data Table
            st.dataframe(
                df,
                column_config={
                    "Boost Value": st.column_config.NumberColumn(
                        "Boost Multiplier",
                        format="%.2fx" # Display as "1.50x"
                    )
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Download Button (replaces the automatic file creation)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"player_boosts_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        else:
            st.warning("No players found with boost values.")
