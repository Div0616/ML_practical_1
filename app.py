import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.title("ML Practical 1 – KNN Weather Classification")

# ---------------- Data ----------------
X = np.array([[30, 70], [25, 80], [27, 60], [31, 65], [23, 85], [28, 75]])
y = np.array([0, 1, 0, 0, 1, 1])  # 0 = Sunny, 1 = Rainy

# ---------------- KNN Model ----------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# New point
new_point = np.array([[26, 78]])
prediction = knn.predict(new_point)[0]

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7, 5))

# Sunny points
ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Sunny", s=100)

# Rainy points
ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Rainy", s=100)

# New predicted point
ax.scatter(
    new_point[0, 0],
    new_point[0, 1],
    marker="*",
    s=300,
    color="red",
    label="New Prediction",
)

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Humidity (%)")
ax.set_title("KNN Weather Classification")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# ---------------- Output ----------------
if prediction == 0:
    st.success("Predicted Weather: Sunny ☀️")
else:
    st.success("Predicted Weather: Rainy 🌧️")
