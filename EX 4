import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = {
    'Study_Hours': [1, 2, 2, 3, 4, 5, 6, 6, 7, 8],
    'Attendance':  [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    'Result':      [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]   
}
df = pd.DataFrame(data)
X = df[['Study_Hours', 'Attendance']]
y = df['Result']
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X, y)
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
new_student = pd.DataFrame({
    'Study_Hours': [hours],
    'Attendance': [attendance]
})
prediction = model.predict(new_student)

print("\nPrediction Result")
if prediction[0] == 1:
    print("The student is likely to PASS the exam.")
else:
    print("The student is likely to FAIL the exam.")
plt.figure(figsize=(12, 7))
plot_tree(
    model,
    feature_names=['Study_Hours', 'Attendance'],
    class_names=['Fail', 'Pass'],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree Classification (ID3 - Entropy)")
plt.show()
