import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
x = np.array([
    [20, 2, 10],
    [22, 3, 15],
    [25, 4, 20],
    [28, 5, 25],
    [30, 6, 30],
    [35, 7, 35],
    [40, 8, 40],
    [45, 9, 45],
    [27, 3, 18],
    [38, 7, 38]
])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
model = GaussianNB()
model.fit(x, y)
new_user = np.array([[32, 6, 32]])
prediction = model.predict(new_user)
probability = model.predict_proba(new_user)
if prediction[0] == 1:
    print("Prediction: Click")
else:
    print("Prediction: No Click")
print("Probability of no click:", probability[0][0])
print("Probability of click:", probability[0][1])
y_pred = model.predict(x)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy * 100, "%")
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0, 0], x[y == 0, 2], label="No Click")
plt.scatter(x[y == 1, 0], x[y == 1, 2], label="Click", color="orange")
plt.scatter(new_user[0, 0], new_user[0, 2], marker="*", s=200, label="New User", color="red")
plt.xlabel("Age")
plt.ylabel("Time spent (minutes)")
plt.title("Naive Bayes Classification")
plt.legend()
plt.grid(True)
plt.show()
