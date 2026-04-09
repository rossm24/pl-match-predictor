# ⚽ Premier League Match Predictor

A machine learning web app that predicts the outcome of any Premier League fixture using historical match data, team form, head-to-head records, and a Random Forest classifier.

---

## Features

- **Win/Draw/Loss probability** bar for any matchup
- **Expected goals (xG)** estimate for both sides
- **Season stats** – played, wins, draws, losses, goals for/against
- **Last 5 form** displayed as W/D/L bubbles
- **Head-to-head** record visualised as a proportional bar
- **Live standings** table pulled from the API
- Falls back to **demo data** automatically if no API key is set

---

## Setup

### 1. Get a free API key
Sign up at https://www.football-data.org/client/register → free tier gives you PL data.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
# macOS / Linux
export FOOTBALL_API_KEY=your_key_here

# Windows (PowerShell)
$env:FOOTBALL_API_KEY="your_key_here"
```

### 4. Run the app
```bash
python app.py
```

Then open http://localhost:5000 in your browser.

---

## Project Structure

```
pl_predictor/
├── app.py            # Flask routes
├── predictor.py      # Data fetching, feature engineering, ML model
├── requirements.txt
├── README.md
└── templates/
    └── index.html    # Dashboard UI
```

---

## How the Model Works

Features used per match:
| Feature | Description |
|---|---|
| Win rate | Season wins / games played |
| Form score | Last 5 games (W=3, D=1, L=0), normalised |
| Goals for/against per game | Attack & defence strength |
| Home/Away win rate | Venue-specific performance |
| Head-to-head record | Historical H2H win/draw/loss ratio |

The **Random Forest** (200 trees) trains on all completed matches this season and outputs three probabilities: Home Win, Draw, Away Win.

The **xG estimate** is a simple weighted average of the team's goals-per-game and the opponent's goals-conceded-per-game.

---

## Ideas for Extending the Project

- Add **Poisson distribution** for scoreline predictions
- Pull in **player injury data** (TransferMarkt scraper)
- Add a **bet value calculator** (compare predicted prob vs bookmaker odds)
- Store predictions in **SQLite** and track your accuracy over time
- Deploy to **Railway** or **Render** so friends can use it
