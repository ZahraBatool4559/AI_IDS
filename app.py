from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("ids_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        dur = float(request.form["dur"])
        sbytes = int(request.form["sbytes"])
        dbytes = int(request.form["dbytes"])

        df = pd.DataFrame([{
            "dur": dur,
            "sbytes": sbytes,
            "dbytes": dbytes
        }])

        pred = model.predict(df)[0]
        result = "🚨 Intrusion Detected" if pred == 1 else "✅ Normal Traffic"

    return f"""
    <h2>AI Intrusion Detection System</h2>
    <form method="post">
      Duration: <input name="dur"><br>
      Source Bytes: <input name="sbytes"><br>
      Destination Bytes: <input name="dbytes"><br>
      <button type="submit">Detect</button>
    </form>
    <h3>{result or ""}</h3>
    """

if __name__ == "__main__":
    app.run(debug=True)
