import numpy as np
from numpy import NDArray


class KalmanFilter:
    """A class implementing a generic multidimensional kalman filter
    without any control inputs. The implementations follows the
    https://www.kalmanfilter.net/ tutorial and the variable names mostly
    adhere to the tutorial's notation.

    The Kalman filter is a recursive algorithm that estimates the hidden state
    of a linear dynamic system from a series of noisy measurements :math:`Z`
    over time. It is a two-step process consisting of a *prediction* and
    an *update* step.

    In the *prediction* step, the next state vector :math:`x_{n+1,n}` and its
    covariance :math:`P_{n+1,n}` are predicted from the previous state vector
    :math:`x_{n,n}` and its covariance :math:`P_{n,n}`, according to the
    following equations:

    .. math::
        x_{n+1,n} = F x_{n,n}

        P_{n+1,n} = F P_{n,n} F^T + Q

    The state transition matrix `F` governs the model dynamics. The process
    noise is assumed to be a zero-mean Gaussian with covariance :math:`Q`.

    In the *update* step, the previously predicted state vector :math:`x_{n,n}`
    and its covariance :math:`P_{n,n}` are corrected based on the new
    measurement :math:`z_n` according to the following equations:

    .. math::
        K_n = P_{n,n-1} H^T (H P_{n,n-1} H^T + R)^{-1}

        x_{n,n} = x_{n,n-1} + K_n (z_n - H x_{n,n-1})

        P_{n,n} = (I - K_n H) P_{n,n-1} (I - K_n H)^T + K_n R K_n^T

    The Kalman gain matrix :math:`K` determines the relative weight of the new
    measurement and the predicted state vector. The measurement noise is
    assumed to be a zero-mean Gaussian with covariance :math:`R`.
    The observation matrix :math:`H` maps the state vector to the measurement
    space. :math:`I` is the identity matrix.


    Attributes
    ----------
    Z : numpy.NDArray
        measurement matrix, shape (num_measures, num_timepoints).
        rows represent different measured parameters,
        columns represent different time points.
    R : numpy.NDArray
        measurement noise covariance matrix,
        shape (num_measures, num_measures).
    H : numpy.NDArray
        observation matrix, shape (num_measures, num_states).
        This matrix is used to map the state vector to the measurement space.
    F : numpy.NDArray
        state transition matrix, shape (num_states, num_states).
        Also known as the state extrapolation or the dynamic model matrix.
        It is used to predict the next state vector from the previous one.
    Q : numpy.NDArray
        process noise covariance matrix, shape (num_states, num_states).
    num_states : int
        number of hidden states to estimate.
    num_measures : int
        number of measured parameters.
    num_timepoints : int
        number of time points in the measurement matrix.
    X_est : numpy.NDArray
        estimated state vectors across time points,
        shape (num_states, num_timepoints).
    X_pred : numpy.NDArray
        predicted state vectors across time points,
        shape (num_states, num_timepoints + 1).
    P_est : numpy.NDArray
        estimated state covariance matrices across time points,
        shape (num_states, num_states, num_timepoints).
    P_pred : numpy.NDArray
        predicted state covariance matrices across time points,
        shape (num_states, num_states, num_timepoints + 1).
    K : numpy.NDArray
        Kalman gain matrices across time points,
        shape (num_states, num_measures, num_timepoints).
    """

    def __init__(
        self,
        Z: NDArray,
        R: NDArray,
        H: NDArray,
        F: NDArray,
        Q: NDArray,
        x0: NDArray,
        P0: NDArray,
    ):
        """
        Parameters
        ----------
        Z : numpy.NDArray
            measurement matrix, shape (num_measures, num_timepoints).
            rows represent different measured parameters,
            columns represent different time points.
        R : numpy.NDArray
            measurement noise covariance matrix,
            shape (num_measures, num_measures).
        H : numpy.NDArray
            observation matrix, shape (num_measures, num_states).
        F : numpy.NDArray
            state transition matrix, shape (num_states, num_states).
        Q : numpy.NDArray
            process noise covariance matrix, shape (num_states, num_states).
        x0 : numpy.NDArray
            initial state vector at time 0, shape (num_states,).
        P0 : numpy.NDArray
            initial state covariance matrix at time 0,
            shape (num_states, num_states).
        """
        self.Z = Z
        self.R = R
        self.H = H
        self.F = F
        self.Q = Q
        self.x0 = x0.squeeze()
        self.P0 = P0

        self.num_states = self.x0.shape[0]
        self.num_measures, self.num_timepoints = self.Z.shape

        # Initialize the output arrays with zeros
        self.X_est = np.zeros((self.num_states, self.num_timepoints))
        self.X_pred = np.zeros((self.num_states, self.num_timepoints + 1))
        self.P_est = np.zeros(
            (self.num_states, self.num_states, self.num_timepoints)
        )
        self.P_pred = np.zeros(
            (self.num_states, self.num_states, self.num_timepoints + 1)
        )
        self.K = np.zeros(
            (self.num_states, self.num_measures, self.num_timepoints)
        )

    def run(self):
        """Run the Kalman filter on the measurement matrix `Z`.
        Recursively predict and update the state vector and its covariance
        at each time point.
        """

        # Initialize the state vector and its covariance matrix
        x_current = self.x0
        P_current = self.P0

        # Loop over time points
        for n in range(self.num_timepoints):
            # Prediction step
            x_next = self.F @ x_current
            P_next = self.F @ P_current @ self.F.T + self.Q

            # Update step
            P_matmul_HT = P_next @ self.H.T  # precompute for efficiency
            k = P_matmul_HT @ np.linalg.inv(
                (self.H @ P_matmul_HT + self.R)
            )  # Kalman gain

            x_current = x_next + k @ (
                self.Z[:, n] - self.H @ x_next
            )  # update state vector

            k_c = (
                np.eye(self.num_states) - k @ self.H
            )  # precompute for efficiency
            P_current = (
                k_c @ P_next @ k_c.T + k @ self.R @ k.T
            )  # update state covariance matrix

            # Save the results
            self.X_est[:, n] = x_current
            self.P_est[:, :, n] = P_current[..., np.newaxis]
            self.X_pred[:, n] = x_next
            self.P_pred[:, :, n] = P_next[..., np.newaxis]
            self.K[:, :, n] = k[..., np.newaxis]

        # Predict at the last time point
        self.X_pred[:, -1] = self.F @ x_current
        P_next = self.F @ P_current @ self.F.T + self.Q
        self.P_pred[:, :, -1] = P_next[..., np.newaxis]
