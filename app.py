from flask import Flask, render_template, jsonify, request
from predictor import MatchPredictor
import os

app = Flask(__name__)
predictor = MatchPredictor()

@app.route("/")
def index():
    teams = predictor.get_teams()
    return render_template("index.html", teams=teams)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    home = data.get("home_team")
    away = data.get("away_team")

    if not home or not away or home == away:
        return jsonify({"error": "Please select two different teams."}), 400

    result = predictor.predict(home, away)
    return jsonify(result)

@app.route("/standings")
def standings():
    data = predictor.get_standings()
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
