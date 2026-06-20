# FIFA World Cup Predictor — Dataset Sources

## Primary Datasets (Free)

### 1. International Football Results (1872–Present)
- **URL**: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
- **What's in it**: Every international match result including scoreline, date, tournament type, home/away
- **Use for**: Match outcome model + goals model

### 2. FIFA World Cup Complete Dataset
- **URL**: https://www.kaggle.com/datasets/abecklas/fifa-world-cup
- **What's in it**: All World Cup matches, players, goals from 1930–2022
- **Use for**: Tournament structure, group stage data, knockout round history

### 3. FIFA World Rankings (Historical)
- **URL**: https://www.kaggle.com/datasets/cashncarry/fifaworldranking
- **What's in it**: Monthly FIFA ranking points for every nation going back to 1993
- **Use for**: Team strength feature

### 4. Club Elo Ratings
- **URL**: http://clubelo.com/API  (free API, no signup)
- **What's in it**: Elo ratings for national teams updated weekly
- **Use for**: Stronger team strength signal than FIFA ranking alone

### 5. Football-Data.co.uk
- **URL**: https://www.football-data.co.uk/international.php
- **What's in it**: International match data with odds, goals, shots
- **Use for**: Cross-validation and betting odds as a calibration feature

### 6. StatsBomb Open Data
- **URL**: https://github.com/statsbomb/open-data
- **What's in it**: Detailed event-level data for selected World Cup tournaments
- **Use for**: Advanced features (xG, pressing stats) if you want deeper analysis

### 7. Transfermarkt Squad Values
- **URL**: https://www.transfermarkt.com (scrape with permission, or use)
- **Kaggle mirror**: https://www.kaggle.com/datasets/davidcariboo/player-scores
- **What's in it**: Squad market values, player ages, nationalities
- **Use for**: Squad quality feature

---

## Recommended Download Order

1. Start with **Dataset 1** (international results) — this is your core training data
2. Add **Dataset 3** (FIFA rankings) to merge in team strength
3. Add **Dataset 2** (World Cup specific) for tournament structure simulation
4. Later add **Dataset 7** (squad values) to improve model accuracy

---

## File Structure After Download

Place downloaded CSV files here:
```
fifa_predictor/
└── data/
    ├── results.csv              ← Dataset 1
    ├── world_cup_matches.csv    ← Dataset 2
    ├── fifa_rankings.csv        ← Dataset 3
    └── squad_values.csv         ← Dataset 7
```
