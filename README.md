# 🏏 IPL Win Predictor

A machine learning web app that predicts the **live win probability** of IPL teams during the second innings, based on match situation — score, overs, wickets, target, and venue.

---

## 🗂️ Project Structure

```
IPL Prediction/
├── datas/
│   ├── deliveries.csv       # Ball-by-ball IPL delivery data
│   ├── matches.csv          # Match-level metadata
│   ├── delivery_df.csv      # Processed delivery data (generated)
│   └── final_df.csv         # Final feature-engineered dataset (generated)
├── templates/
│   └── index.html           # Frontend HTML served by Flask
├── lr_pipe.pkl              # Trained Logistic Regression pipeline (pickled)
├── main.ipynb               # Data processing, feature engineering & model training
└── server.py                # Flask backend server
```

---

## ⚙️ How It Works

1. **Data Processing** (`main.ipynb`)
   - Loads raw `matches.csv` and `deliveries.csv`
   - Filters to 8 current IPL franchises, merging old franchise names (e.g. `Delhi Daredevils` → `Delhi Capitals`)
   - Computes second-innings ball-by-ball features:
     - `runs_left`, `balls_left`, `wickets_left`
     - `crr` (current run rate), `rrr` (required run rate)
   - Labels each delivery with match result (`1` = batting team wins)

2. **Model Training**
   - `OneHotEncoder` on `batting_team`, `bowling_team`, `city`
   - `LogisticRegression` (liblinear solver) wrapped in a `sklearn Pipeline`
   - Pipeline serialized to `lr_pipe.pkl`

3. **Web App** (`server.py` + `templates/index.html`)
   - Built with **Flask**
   - `GET /` — serves the frontend (`index.html`) with team and city dropdowns
   - `POST /predict` — accepts JSON match state, runs `pipe.predict_proba()`, returns win % for both teams

---

## 🧠 ML Pipeline

```
Input Features
│
├── batting_team        ──┐
├── bowling_team        ──┤  OneHotEncoder (drop='first')
├── city                ──┘
│
├── runs_left           ──┐
├── balls_left          ──┤  Passthrough
├── wickets_left        ──┤
├── total_runs_x        ──┤  (target score)
├── crr                 ──┤  (current run rate)
└── rrr                 ──┘  (required run rate)
        │
        ▼
  LogisticRegression
  (solver='liblinear')
        │
        ▼
  Win Probability [0, 1]
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Debocoderoid/IPL-Prediction.git
cd IPL-Prediction
pip install -r requirements.txt
```

### Run the App

```bash
python server.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## 🌐 API Reference

### `POST /predict`

**Request body (JSON):**

```json
{
  "batting_team": "Chennai Super Kings",
  "bowling_team": "Mumbai Indians",
  "city": "Mumbai",
  "target": 180,
  "current_score": 120,
  "wickets_left": 5,
  "balls_left": 30
}
```

**Response (JSON):**

```json
{
  "batting_team": "Chennai Super Kings",
  "bowling_team": "Mumbai Indians",
  "win": 62.3,
  "lose": 37.7,
  "runs_left": 60,
  "balls_left": 30,
  "crr": 8.0,
  "rrr": 12.0
}
```

---

## 📦 Requirements

```
flask
pandas
numpy
scikit-learn
matplotlib
```

> Generate `requirements.txt` with: `pip freeze > requirements.txt`

---

## 🎮 Usage

| Field | Description |
|---|---|
| Batting Team | Team currently batting in the 2nd innings |
| Bowling Team | Team bowling in the 2nd innings |
| Host City | Venue city |
| Target | Runs set by the 1st innings team |
| Current Score | Runs scored so far by the batting team |
| Wickets Left | Wickets remaining (10 − fallen wickets) |
| Balls Left | Balls remaining in the innings |

Click **Predict** to get live win probability for both teams.

---

## 🏟️ Supported Teams

- Chennai Super Kings
- Delhi Capitals
- Kolkata Knight Riders
- Mumbai Indians
- Punjab Kings
- Rajasthan Royals
- Royal Challengers Bengaluru
- Sunrisers Hyderabad

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | ~82% |
| Random Forest | ~84% |

> Logistic Regression was chosen for deployment due to faster inference and interpretability.

---

## 📁 Data Source

- [IPL Dataset on Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- Covers IPL seasons from **2008 to 2020**

---

## 🙋 Author

**Debojyoti** — IIT Hyderabad  
[GitHub](https://github.com/Debocoderoid) · [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
