# --- SPLIT COLUMN DETECTION ---
            # 1. Slate
            slate_col = find_col(df_proj.columns, ["slate", "contest", "label"])
            # 2. Game - Look for a single "Game" column
            game_col = find_col(df_proj.columns, ["game", "matchup", "match"])
            # 3. Team/Opp Split
            team_col = find_col(df_proj.columns, ["team", "tm", "squad"])
            opp_col = find_col(df_proj.columns, ["opp", "opponent", "vs"])

            # --- HEURISTIC FALLBACK FOR GAME ---
            # If no explicit columns found, scan data for " v " or "@" patterns
            if not game_col and not (team_col and opp_col):
                for col in df_proj.columns:
                    # Check first few non-null values
                    sample = df_proj[col].dropna().astype(str).head(5)
                    if any(" v " in x.lower() or " vs " in x.lower() or "@" in x for x in sample):
                        game_col = col
                        break

            if name_col and points_col:
                df_boosts['join_key'] = df_boosts['Player Name'].apply(normalize_name)
                df_proj['join_key'] = df_proj[name_col].apply(normalize_name)
                
                merged_df = pd.merge(df_boosts, df_proj, on='join_key', how='inner')
                
                if merged_df.empty:
                    st.error("No players matched! This usually means names didn't match or the date is wrong.")
                else:
                    merged_df = merged_df.rename(columns={points_col: 'Projection'})
                    if pos_col:
                        merged_df['Position'] = merged_df[pos_col].fillna(merged_df['Position'])
                    merged_df['Position'] = merged_df['Position'].apply(normalize_position)
                    
                    # --- STANDARDIZE SLATE ---
                    if slate_col:
                        merged_df['Slate'] = merged_df[slate_col].fillna("Unknown")
                    else:
                        merged_df['Slate'] = "ALL"
                        
                    # --- STANDARDIZE GAME ---
                    # Priority: Construct from Team + Opp if available
                    if team_col and opp_col:
                        merged_df['Game'] = merged_df[team_col].astype(str) + " vs " + merged_df[opp_col].astype(str)
                    elif game_col:
                        merged_df['Game'] = merged_df[game_col].fillna("Unknown")
                    else:
                        merged_df['Game'] = "ALL"
