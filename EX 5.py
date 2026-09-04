import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x = np.array([
    [25, 120],
    [30, 125],
    [28, 118],
    [35, 130],
    [40, 135],
    [45, 140],
    [50, 145],
    [55, 150],
    [32, 128],
    [48, 142]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
k = 3
model = KNeighborsClassifier(
    n_neighbors=k,
    metric='euclidean'
)
model.fit(x_train, y_train)
new_patient = np.array([[38, 110]])
prediction = model.predict(new_patient)
if prediction[0] == 1:
    print("New patient: Heart disease")
else:
    print("New patient: No Heart disease")
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy * 100, "%")
plt.figure(figsize=(8, 6))
plt.scatter(
    x[y == 0, 0],
    x[y == 0, 1],
    label="No Heart Disease",
    marker="o"
)
plt.scatter(
    x[y == 1, 0],
    x[y == 1, 1],
    label="Heart Disease",
    marker="s"
)
plt.scatter(
    new_patient[0, 0],
    new_patient[0, 1],
    label="New Patient",
    marker="*",
    s=200
)
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("K-NN Classification")
plt.legend()
plt.grid(True)
plt.show()
