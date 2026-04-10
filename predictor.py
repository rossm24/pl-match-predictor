import os
import requests
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()
print("API KEY:", os.environ.get("FOOTBALL_API_KEY"))  # remove after testing 

API_KEY = os.environ.get("FOOTBALL_API_KEY", "YOUR_API_KEY_HERE")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}
PL_ID = 2021  # Premier League competition ID
YEAR = [2023, 2024, 2025]  # Change as needed for different seasons


def api_get(path):
    r = requests.get(f"{BASE_URL}{path}", headers=HEADERS)
    r.raise_for_status()
    return r.json()


class MatchPredictor:
    def __init__(self):
        self.matches = []
        self.teams = []
        self.model = None
        self.team_stats = {}
        self._load_data()
        self._train()
        self._load_standings_data()

    # Data loading

    def _load_data(self):
        """Fetch completed PL matches from every season."""
        try:
            self.matches = []
            for year in YEAR:
                data = api_get(f"/competitions/{PL_ID}/matches?status=FINISHED&season={year}")
                self.matches.extend(data.get("matches", []))

            #standings_data = api_get(f"/competitions/{PL_ID}/standings?season=2025")
            #table = standings_data["standings"][0]["table"]
            #self.teams = sorted([row["team"]["name"] for row in table])

            self._compute_team_stats()
        except Exception as e:
            print(f"[Warning] Could not fetch live data: {e}")
            print("Using demo data instead.")
            self._load_demo_data()

    def _load_standings_data(self):
        # fetch this years current standings, need to keep separate from historical matches to avoid data leakage
        try:
            standings_data = api_get(f"/competitions/{PL_ID}/standings?season={YEAR[-1]}")
            table = standings_data["standings"][0]["table"]
            self.teams = sorted([row["team"]["name"] for row in table])
            self.current_standings = table
        except Exception as e:
            print("Could not fetch standings data.")

        print(self.current_standings[0])
        print(list(self.team_stats.items())[0])


    def _load_demo_data(self):
        """Fallback demo data so the app works without an API key."""
        self.teams = [
            "Arsenal FC", "Aston Villa FC", "Brentford FC", "Brighton & Hove Albion FC",
            "Chelsea FC", "Crystal Palace FC", "Everton FC", "Fulham FC",
            "Ipswich Town FC", "Leicester City FC", "Liverpool FC", "Manchester City FC",
            "Manchester United FC", "Newcastle United FC", "Nottingham Forest FC",
            "Southampton FC", "Tottenham Hotspur FC", "West Ham United FC",
            "Wolverhampton Wanderers FC", "AFC Bournemouth"
        ]
        # Seed fake historical stats
        rng = np.random.default_rng(42)
        for team in self.teams:
            played = 30
            wins = int(rng.integers(5, 25))
            draws = int(rng.integers(2, 10))
            losses = played - wins - draws
            losses = max(0, losses)
            self.team_stats[team] = {
                "played": played, "wins": wins, "draws": draws, "losses": losses,
                "goals_for": int(rng.integers(20, 70)),
                "goals_against": int(rng.integers(15, 60)),
                "form": list(rng.choice(["W", "D", "L"], size=5)),
                "home_wins": int(wins * 0.6), "home_draws": int(draws * 0.5),
                "away_wins": int(wins * 0.4), "away_draws": int(draws * 0.5),
            }
        self._generate_demo_matches(rng)

    def _generate_demo_matches(self, rng):
        for home in self.teams:
            for away in self.teams:
                if home == away:
                    continue
                outcome = rng.choice(["HOME_WIN", "DRAW", "AWAY_WIN"], p=[0.46, 0.26, 0.28])
                self.matches.append({
                    "homeTeam": {"name": home},
                    "awayTeam": {"name": away},
                    "score": {"winner": outcome},
                })

    # Feature engineering

    def _compute_team_stats(self):
        stats = defaultdict(lambda: {
            "played": 0, "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0,
            "home_wins": 0, "home_draws": 0,
            "away_wins": 0, "away_draws": 0,
            "results": []  # chronological W/D/L
        })

        for m in self.matches:
            hw = m["homeTeam"]["name"]
            aw = m["awayTeam"]["name"]
            winner = m["score"].get("winner")
            hg = m["score"].get("fullTime", {}).get("home", 0) or 0
            ag = m["score"].get("fullTime", {}).get("away", 0) or 0

            for team in [hw, aw]:
                stats[team]["played"] += 1

            stats[hw]["goals_for"] += hg
            stats[hw]["goals_against"] += ag
            stats[aw]["goals_for"] += ag
            stats[aw]["goals_against"] += hg

            if winner == "HOME_TEAM":
                stats[hw]["wins"] += 1; stats[hw]["home_wins"] += 1
                stats[hw]["results"].append("W")
                stats[aw]["losses"] += 1
                stats[aw]["results"].append("L")
            elif winner == "AWAY_TEAM":
                stats[aw]["wins"] += 1; stats[aw]["away_wins"] += 1
                stats[aw]["results"].append("W")
                stats[hw]["losses"] += 1
                stats[hw]["results"].append("L")
            elif winner == "DRAW":
                stats[hw]["draws"] += 1; stats[hw]["home_draws"] += 1
                stats[aw]["draws"] += 1; stats[aw]["away_draws"] += 1
                stats[hw]["results"].append("D")
                stats[aw]["results"].append("D")

        # Compute rolling form (last 5 games)
        for team, s in stats.items():
            last5 = s["results"][-5:]
            s["form"] = last5

        self.team_stats = dict(stats)

    def _form_score(self, form_list):
        """Convert W/D/L list → numeric score (W=3, D=1, L=0)."""
        mapping = {"W": 3, "D": 1, "L": 0}
        if not form_list:
            return 1.5
        return sum(mapping.get(r, 0) for r in form_list) / (3 * len(form_list))

    def _h2h(self, home, away):
        """Return (home_wins, draws, away_wins) in head-to-head history."""
        hw = dw = aw = 0
        for m in self.matches:
            if m["homeTeam"]["name"] == home and m["awayTeam"]["name"] == away:
                w = m["score"].get("winner")
                if w == "HOME_TEAM": hw += 1
                elif w == "DRAW": dw += 1
                elif w == "AWAY_TEAM": aw += 1
        return hw, dw, aw

    def _build_features(self, home, away):
        hs = self.team_stats.get(home, {})
        as_ = self.team_stats.get(away, {})

        def safe(d, k, default=0):
            v = d.get(k, default)
            return v if v is not None else default

        hp = max(safe(hs, "played"), 1)
        ap = max(safe(as_, "played"), 1)

        home_win_rate = safe(hs, "wins") / hp
        away_win_rate = safe(as_, "wins") / ap
        home_form = self._form_score(hs.get("form", []))
        away_form = self._form_score(as_.get("form", []))
        home_gf_pg = safe(hs, "goals_for") / hp
        home_ga_pg = safe(hs, "goals_against") / hp
        away_gf_pg = safe(as_, "goals_for") / ap
        away_ga_pg = safe(as_, "goals_against") / ap
        home_home_wr = safe(hs, "home_wins") / hp
        away_away_wr = safe(as_, "away_wins") / ap
        h2h_hw, h2h_d, h2h_aw = self._h2h(home, away)
        h2h_total = max(h2h_hw + h2h_d + h2h_aw, 1)

        return [
            home_win_rate, away_win_rate,
            home_form, away_form,
            home_gf_pg, home_ga_pg,
            away_gf_pg, away_ga_pg,
            home_home_wr, away_away_wr,
            h2h_hw / h2h_total, h2h_d / h2h_total, h2h_aw / h2h_total,
        ]

    # Training

    def _train(self):
        X, y = [], []
        for m in self.matches:
            home = m["homeTeam"]["name"]
            away = m["awayTeam"]["name"]
            winner = m["score"].get("winner")
            if winner not in ("HOME_TEAM", "DRAW", "AWAY_TEAM"):
                continue
            label = {"HOME_TEAM": 0, "DRAW": 1, "AWAY_TEAM": 2}[winner]
            try:
                feats = self._build_features(home, away)
                X.append(feats)
                y.append(label)
            except Exception:
                continue

        if len(X) < 10:
            print("[Warning] Not enough data to train a reliable model.")
            self.model = None
            return

        self.model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.model.fit(X, y)
        print(f"[Model] Trained on {len(X)} matches.")

    # Prediction

    def predict(self, home, away):
        feats = self._build_features(home, away)
        hs = self.team_stats.get(home, {})
        as_ = self.team_stats.get(away, {})

        if self.model:
            probs = self.model.predict_proba([feats])[0]
            # model classes: 0=HOME_WIN, 1=DRAW, 2=AWAY_WIN
            classes = self.model.classes_
            prob_map = {c: p for c, p in zip(classes, probs)}
            home_prob = round(float(prob_map.get(0, 0)) * 100, 1)
            draw_prob = round(float(prob_map.get(1, 0)) * 100, 1)
            away_prob = round(float(prob_map.get(2, 0)) * 100, 1)
        else:
            home_prob, draw_prob, away_prob = 46.0, 26.0, 28.0

        # Predicted scoreline (simple Poisson-inspired estimate)
        hp = max(hs.get("played", 1), 1)
        ap = max(as_.get("played", 1), 1)
        home_xg = round((hs.get("goals_for", 0) / hp) * 0.9 +
                        (as_.get("goals_against", 0) / ap) * 0.1, 1)
        away_xg = round((as_.get("goals_for", 0) / ap) * 0.85 +
                        (hs.get("goals_against", 0) / hp) * 0.15, 1)

        h2h_hw, h2h_d, h2h_aw = self._h2h(home, away)

        return {
            "home_team": home,
            "away_team": away,
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "home_form": hs.get("form", []),
            "away_form": as_.get("form", []),
            "home_stats": {
                "played": hs.get("played", 0),
                "wins": hs.get("wins", 0),
                "draws": hs.get("draws", 0),
                "losses": hs.get("losses", 0),
                "goals_for": hs.get("goals_for", 0),
                "goals_against": hs.get("goals_against", 0),
            },
            "away_stats": {
                "played": as_.get("played", 0),
                "wins": as_.get("wins", 0),
                "draws": as_.get("draws", 0),
                "losses": as_.get("losses", 0),
                "goals_for": as_.get("goals_for", 0),
                "goals_against": as_.get("goals_against", 0),
            },
            "h2h": {"home_wins": h2h_hw, "draws": h2h_d, "away_wins": h2h_aw},
        }

    # Helpers

    def get_teams(self):
        return self.teams
    
    def get_standings(self):
        rows = []
        for team in self.current_standings:
            rows.append({"team": team["team"]["name"], "played": team["playedGames"], "wins": team["won"],
                         "draws": team["draw"], "losses": team["lost"],
                         "gf": team["goalsFor"], "ga": team["goalsAgainst"],
                         "gd": team["goalDifference"], "points": team["points"]})
        rows.sort(key=lambda r: (-r["points"], -r["gd"]))
        return rows