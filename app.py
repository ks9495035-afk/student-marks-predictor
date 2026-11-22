from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Sample student data
data = {
    "StudyHours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 75, 80, 82, 85, 90, 92, 95],
    "PreviousMarks": [40, 45, 50, 55, 60, 65, 70, 75, 82, 88],
    "Marks": [42, 47, 53, 58, 64, 69, 74, 80, 87, 93]
}

df = pd.DataFrame(data)

X = df[["StudyHours", "Attendance", "PreviousMarks"]]
y = df["Marks"]

model = LinearRegression()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])

        student = pd.DataFrame({
            "StudyHours": [study_hours],
            "Attendance": [attendance],
            "PreviousMarks": [previous_marks]
        })

        prediction = round(model.predict(student)[0], 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)