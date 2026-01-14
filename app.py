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
    """Fetches player data from API using the selected date + lookahead."""
    session = requests.Session()
    sport_data = []
    
    # Determine Date Strategy
    # Fetch selected date AND next day (or next 7 for NFL) to account for timezone/server rollovers
    if sport.lower() == 'nfl':
        target_dates = [target_date + datetime.timedelta(days=i) for i in range(7)]
    else:
        target_dates = [target_date, target_date + datetime.timedelta(days=1)]

    # 1. Probe Dates (Find which days actually have data to avoid wasting requests)
    valid_dates = []
    for d in target_dates:
        d_str = str(d)
        # Probe with "a" to check existence
        probe_url = (
            f"https://api.real.vg/players/sport/{sport}/search"
            f"?day={d_str}&includeNoOneOption=false"
            f"&query=a&searchType=ratingLineup"
        )
        try:
            r = session.get(probe_url, timeout=3)
            if r.status_code == 200 and r.json().get("players"):
                valid_dates.append(d_str)
        except:
            pass
            
    # If no dates found via probe, fallback to target_date
    if not valid_dates:
        valid_dates = [str(target_date)]

    # 2. Parallel Fetch Alphabet for Valid Dates
    letters = string.ascii_uppercase
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_req = {}
        for d_str in valid_dates:
            for letter in letters:
                future_to_req[executor.submit(fetch_letter, session, sport, d_str, letter)] = d_str
        
        for future in concurrent.futures.as_completed(future_to_req):
            active_date_str = future_to_req[future]
            try:
                players = future.result()
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
                    
                    # Note: We don't filter duplicates here. We let the downstream merge logic
                    # (which uses a dictionary keyed by name) overwrite older entries with newer ones.

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
            except:
                continue
            
    return sport_data
