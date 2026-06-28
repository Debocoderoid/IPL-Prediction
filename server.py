from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
pipe = pickle.load(open('lr_pipe.pkl', 'rb'))

TEAMS = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bengaluru',
 'Kolkata Knight Riders',
 'Punjab Kings',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

CITIES = ['Chandigarh', 'Delhi', 'Jaipur', 'Chennai', 'Kolkata', 'Mumbai',
       'Cape Town', 'Durban', 'Port Elizabeth', 'Centurion',
       'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
       'Ahmedabad', 'Dharamsala', 'Pune', 'Bangalore', 'Hyderabad',
       'Raipur', 'Abu Dhabi', 'Ranchi', 'Cuttack', 'Visakhapatnam',
       'Indore', 'Dubai', 'Sharjah', 'Navi Mumbai', 'Guwahati', 'Mohali']

@app.route('/')
def index():
    return render_template('index.html', teams=TEAMS, cities=CITIES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    batting_team  = data['batting_team']
    bowling_team  = data['bowling_team']
    city          = data['city']
    target        = int(data['target'])
    current_score = int(data['current_score'])
    wickets_left  = int(data['wickets_left'])
    balls_left    = int(data['balls_left'])

    runs_left = target - current_score
    overs_done = (120 - balls_left) / 6
    crr = current_score / overs_done if overs_done > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    total_runs_x = target - 1  # target is score + 1

    input_df = pd.DataFrame({
        'batting_team':  [batting_team],
        'bowling_team':  [bowling_team],
        'city':          [city],
        'runs_left':     [runs_left],
        'balls_left':    [balls_left],
        'wickets_left':  [wickets_left],
        'total_runs_x':  [total_runs_x],
        'crr':           [round(crr, 2)],
        'rrr':           [round(rrr, 2)],
    })

    result = pipe.predict_proba(input_df)
    win  = round(float(result[0][1]) * 100, 1)
    lose = round(float(result[0][0]) * 100, 1)

    return jsonify({
        'win':          win,
        'lose':         lose,
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'runs_left':    runs_left,
        'balls_left':   balls_left,
        'crr':          round(crr, 2),
        'rrr':          round(rrr, 2),
    })

if __name__ == '__main__':
    app.run(debug=True)