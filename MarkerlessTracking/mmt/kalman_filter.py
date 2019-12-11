import numpy as np
from numpy.linalg import inv, multi_dot


class LinearKalmanFilter:

    def __init__(self, timestep, sigma=1, model_velocity=False):
        """
        If model_velocity is False, the system is modelled as static and
        affected by white noise. If it is true, velocity is included in the
        state and position is not assumed to be static.
        """

        if not model_velocity:
            self.sigma = sigma
            self.state_transition = np.eye(6)
            self.measurement_model = np.eye(6)
            self.current_state = np.zeros(6)
            self.state_covariance = np.zeros((6, 6))
            self.process_covariance = np.eye(6)
            self.measurement_covariance = np.eye(6)
            # Diagnostic variables
            self.Kalman_gain = None
            self.x_prior = None # A-priori state estimate
            self.P_prior = None # A-priori state covariance estimate

        else:
            dt = timestep
            self.sigma = sigma
            self.current_state = np.zeros(12)
            self.state_transition = np.eye(12)
            self.state_transition[:6, 6:] = dt*np.eye(6)
            self.measurement_model = np.zeros((6, 12))
            self.measurement_model[:6, :6] = np.eye(6)
            self.state_covariance = np.zeros((12, 12))

            # Gtop = 0.5*(dt**2)*np.ones(6)
            # Gbottom = dt*np.ones(6)
            # G = np.concatenate((Gtop, Gbottom)).reshape((12, 1))
            Q = np.zeros((12, 12))
            Q[:6, :6] = np.diag(0.25*dt**4*np.ones(6))
            Q[:6, 6:] = np.diag(0.5*dt**3*np.ones(6))
            Q[6:, 6:] = np.diag(dt**2*np.ones(6))
            Q[6:, :6] = np.diag(0.5*dt**3*np.ones(6))
            self.process_covariance = Q
            self.measurement_covariance = np.eye(12)
            # Diagnostic variables
            self.Kalman_gain = None
            self.x_prior = None # A-priori state estimate
            self.P_prior = None # A-priori state covariance estimate

    @property
    def state(self):
        return self.current_state

    @property
    def pose(self):
        """If model_velocity is False, this is same as the state."""
        return self.current_state[:6]

    def step(self, measurement, measurement_covariance):
        """
        Applies Kalman filter, given measurement and measurement covariance.
        """
        self.measurement_covariance = measurement_covariance
        F = self.state_transition
        H = self.measurement_model
        P = self.state_covariance
        Q = self.sigma*self.process_covariance
        R = self.measurement_covariance

        # Prediciton step
        x_prior = np.dot(F, self.current_state)
        P_prior = multi_dot([F, P, F.T]) + Q
        I = np.eye(P_prior.shape[0])
        # Observation
        innovation = measurement - np.dot(H, x_prior)
        innovation_covariance = multi_dot([H, P_prior, H.T]) + R
        # Update step
        K = multi_dot([P_prior, H.T, inv(innovation_covariance)])
        x_post = x_prior + np.dot(K, innovation)
        P_post = np.dot((np.eye(P_prior.shape[0]) - np.dot(K, H)), P_prior)

        self.Kalman_gain = K
        self.x_prior = x_prior
        self.P_prior = P_prior

        self.current_state = x_post
        self.state_covariance = P_post

    def save_diagnostics(self, output_filename):
        """
        Saves Kalman gain, prior state estimate, and prior covariance estimate
        to npz.
        """
        np.savez(output_filename, K=self.Kalman_gain,
                 R=self.measurement_covariance,
                 Q=self.sigma*self.process_covariance,
                 x_prior=self.x_prior,
                 P_prior=self.P_prior,
                 x_post=self.current_state)


if __name__ == '__main__':
    LKF = LinearKalmanFilter(1, model_velocity=True)
    print(LKF.process_covariance)
