import numpy as np
from kalmenfilter import KalmanFilter

F = np.array([[1, 1], [0, 1]])
B = np.array([[0.5], [1]])
H = np.array([[1, 0]])
Q = np.array([[1, 0], [0, 1]])
R = np.array([[1]])
# Initial state and covariance
x0 = np.array([[0], [1]])
P0 = np.array([[1, 0], [0, 1]])
# Create Kalman Filter instance
kf = KalmanFilter(F, B, H, Q, R, x0, P0)
# Predict and update with the control input and measurement
u = np.array([[1]])
z = np.array([[1]])
# Predict step
predicted_state = kf.predict(u)
print("Predicted state:\n", predicted_state)
# Update step
updated_state = kf.update(z)
print("Updated state:\n", updated_state)
