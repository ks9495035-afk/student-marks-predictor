
# import pandas as pd
# from sklearn.linear_model import LinearRegression

# # Sample student data
# data = {
#     "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     "Marks": [35, 40, 45, 50, 55, 60, 65, 70, 80, 90]
# }

# df = pd.DataFrame(data)

# # Input and output
# X = df[["Hours"]]
# y = df["Marks"]

# # Train the ML model
# model = LinearRegression()
# model.fit(X, y)

# # Take input from user
# hours = float(input("Enter study hours: "))

# # Predict marks
# prediction = model.predict([[hours]])

# print(f"Predicted Marks: {prediction[0]:.2f}")
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample student data
data = {
    "StudyHours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Attendance": [60, 65, 70, 75, 80, 82, 85, 90, 92, 95],
    "PreviousMarks": [40, 45, 50, 55, 60, 65, 70, 75, 82, 88],
    "Marks": [42, 47, 53, 58, 64, 69, 74, 80, 87, 93]
}

df = pd.DataFrame(data)

# Input features
X = df[["StudyHours", "Attendance", "PreviousMarks"]]

# Target
y = df["Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("========================================")
print("       STUDENT MARKS PREDICTOR")
print("========================================")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_marks = float(input("Enter previous marks: "))

# Create input with same feature names
student = pd.DataFrame({
    "StudyHours": [study_hours],
    "Attendance": [attendance],
    "PreviousMarks": [previous_marks]
})

prediction = model.predict(student)

print("\nPredicted Marks:", round(prediction[0], 2))