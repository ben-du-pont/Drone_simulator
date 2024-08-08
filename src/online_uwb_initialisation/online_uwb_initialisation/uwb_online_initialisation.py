import numpy as np
from scipy.optimize import least_squares, minimize

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans

from drone_uwb_simulator.UWB_protocol import Anchor
from online_uwb_initialisation.trajectory_optimisation import TrajectoryOptimization

from drone_uwb_simulator.drone_simulator import DroneSimulation

from drone_uwb_simulator.drone_dynamics import Waypoint, Trajectory


class UwbOnlineInitialisation:

    def __init__(self):
        

        # Data from simulator to calculate errors and generate measurements
        self.base_anchors = []
        self.unknown_anchors = []
        self.drone_position = []

        # Dictionnaries containining all the measurements for the anchors to run the least squares
        self.known_anchor_measurements = {}
        self.unknown_anchor_measurements = {}

        # Default structure of the above dictionnary for one single anchor
        self.default_anchor_structure = {
            # Measurements from the initial rough estimation
            "distances_pre_rough_estimate": [],
            "positions_pre_rough_estimate": [],

            # Measurements from the optimal trajectory
            "distances_post_rough_estimate": [],
            "positions_post_rough_estimate": [],

            # Rough estimator
            "estimator_rough_linear": [0, 0, 0, 0, 0],
            "estimated_position_rough_linear": [0.0, 0.0, 0.0],
            "covariance_matrix_rough_linear": [],

            # Refined non linear rough estimator
            "estimator_rough_non_linear": [0, 0, 0, 0, 0],
            "estimated_position_rough_non_linear": [],
            "covariance_matrix_rough_non_linear": [],

            # Final estimator
            "estimator": [0, 0, 0, 0, 0],
            "estimated_position": [0, 0, 0],
            "covariance_matrix": [],

            # Variables for the thresholding
            "FIM": [np.zeros((3,3))],
            "GDOP": [float('inf')],
            "residuals": [float('inf')],
            "condition_number": [float('inf')],
            "covariances": [[float('inf'),float('inf'),float('inf')]]

        }

        # Optimiser class instance to call to run the optimisation procedure
        self.optimiser = TrajectoryOptimization()

        # Dictionnaries containing anchor IDs and the status of the intiialisation for those anchors
        self.anchor_rough_estimate_found = {}
        self.all_optimal_measurements_collected = {}

        # Trajectory to follow
        self.trajectory_waypoints = []
        self.optimised_trajectory_waypoints = []

        # Trajectory discretised
        self.spline_x = None
        self.spline_y = None
        self.spline_z = None


        # Tuning parameters
        self.params = {
            # Measurement gathering parameters
            'distance_to_anchor_ratio_threshold': 0.01, # tangent ratio between consecutive measurements -> smaller is more measurements
            'number_of_redundant_measurements': 1, # Number of measurements to take at the same place
            'distance_rejection_threshold': 50, # Minimum distance to the anchor to not reject the measurement
            
            # Least squares parameters
            'use_linear_bias': True, # Use linear bias in the linear least squares
            'use_constant_bias': True, # Use constant bias in the linear least squares
            'non_linear_optimisation_type': "IRLS",

            # Stopping criterion parameters
            'FIM_thresh': 0.2,
            'GDOP_thresh': 2,
            'residuals_thresh': 1,
            'condition_number_thresh': 500,
            'covariance_thresh': 0.3,
            'number_of_measurements_thresh': 1000,

            'FIM_ratio_thresh': 0.1,
            'GDOP_ratio_thresh': 0.1,
            'residuals_ratio_thresh': 0.1,
            'condition_number_ratio_thresh': 0.1,
            'covariance_ratio_thresh': 0.1,

            # Outlier rejection parameters
            "z_score_threshold": 2,

        }

        self.error = 100

    def process_anchor_info(self, known_anchors, unknown_anchors):
        """Process the anchor information and store it in the UWB Initialisation class variables
        
        Parameters:
        - known_anchors: list of Anchor objects, the known anchors.
        - unknown_anchors: list of Anchor objects, the unknown anchors.
        """

        self.base_anchors = []
        self.unknown_anchors = []

        for anchor in known_anchors:
            self.base_anchors.append(anchor)
        for anchor in unknown_anchors:
            self.unknown_anchors.append(anchor)

    def get_distance_to_anchors(self, x, y, z, reach_distance = 100):
        """Given a drone position, returns the distance to all the anchors it can reach and their IDs
        
        Parameters:
        - x: float, the x-coordinate of the drone.
        - y: float, the y-coordinate of the drone.
        - z: float, the z-coordinate of the drone.
        - reach_distance: float, the maximum distance at which the drone can measure the anchors.
        
        Returns:
        - known_anchor_distances: list of lists, the distances to the known anchors and their IDs.
        - unknown_anchor_distances: list of lists, the distances to the unknown anchors and their IDs.
        """

        known_anchor_distances = []
        unknown_anchor_distances = []

        for anchor in self.base_anchors:
            distance = anchor.request_distance(x, y, z)
            if distance < reach_distance:
                known_anchor_distances.append([distance, anchor.anchor_ID])

        for unknown_anchor in self.unknown_anchors:
            distance = unknown_anchor.request_distance(x, y, z)
            if distance < reach_distance:
                unknown_anchor_distances.append([distance, unknown_anchor.anchor_ID])

        return known_anchor_distances, unknown_anchor_distances
    
    def get_distance_to_anchors_gt(self, x, y, z, reach_distance = 100):
        """Given a drone position, returns the distance to all the anchors it can reach and their IDs
        
        Parameters:
        - x: float, the x-coordinate of the drone.
        - y: float, the y-coordinate of the drone.
        - z: float, the z-coordinate of the drone.
        - reach_distance: float, the maximum distance at which the drone can measure the anchors.
        
        Returns:
        - known_anchor_distances: list of lists, the distances to the known anchors and their IDs.
        - unknown_anchor_distances: list of lists, the distances to the unknown anchors and their IDs.
        """

        known_anchor_distances = []
        unknown_anchor_distances = []

        for anchor in self.base_anchors:
            distance = anchor.request_distance_gt(x, y, z)
            if distance < reach_distance:
                known_anchor_distances.append([distance, anchor.anchor_ID])

        for unknown_anchor in self.unknown_anchors:
            distance = unknown_anchor.request_distance_gt(x, y, z)
            if distance < reach_distance:
                unknown_anchor_distances.append([distance, unknown_anchor.anchor_ID])

        return known_anchor_distances, unknown_anchor_distances
    
    def setup_linear_least_square(self, measurements):
        """Setup the linear least squares problem given a set of measurements, using the parameters set in the class for deciding on wether on not to include bias terms

        Parameters:
        - measurements: list of lists, the measurements to use for the linear least squares problem where each line is of the form [x, y, z, distance]

        Returns:
        - A: numpy array, the matrix A of the linear least squares problem
        - b: numpy array, the vector b of the linear least squares problem
        """

        use_linear_bias = self.params['use_linear_bias']
        use_constant_bias = self.params['use_constant_bias']

        if use_constant_bias and use_linear_bias:
            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([2*x, 2*y, 2*z, measured_dist**2, -2*measured_dist, 1])
                b.append(norm_squared)

        elif use_constant_bias:
            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([-2*x, -2*y, -2*z, 2*measured_dist, 1])
                b.append(measured_dist**2 - norm_squared)

        elif use_linear_bias:
            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([2*x, 2*y, 2*z, measured_dist**2, 1])
                b.append(norm_squared)

        else:

            A = []
            b = []
            for measurement in measurements:
                x, y, z, measured_dist = measurement[0:4]
                norm_squared = x**2 + y**2 + z**2
                A.append([-2*x, -2*y, -2*z, 1])
                b.append(measured_dist**2 - norm_squared)


        return np.array(A), np.array(b)

    def compute_GDOP(self, measurements, target_coords):
        """Compute the Geometric Dilution of Precision (GDOP) given a set of measurements corresponding to drone positions in space, and the target coordinates of the anchor to relate the measurements to
        
        Parameters:
        - measurements: list of lists, the measurements to use for the GDOP calculation where each line is of the form [x, y, z, distance]
        - target_coords: list of floats, the target coordinates of the anchor to relate the measurements to [x, y, z]
        
        Returns:
        - gdop: float, the computed GDOP, or infinity if the matrix is singular
        """
        
        if len(measurements) < 4:
            return float('inf') # Not enough points to calculate GDOP
    
        x,y,z = target_coords
        A = []
        for measurement in measurements:
            x_i, y_i, z_i, _ = measurement
            R = np.linalg.norm([x_i-x, y_i-y, z_i-z])
            A.append([(x_i-x)/R, (y_i-y)/R, (z_i-z)/R, 1])

        A = np.array(A)

        try:
            inv_at_a = np.linalg.inv(A.T @ A)
            gdop = np.sqrt(np.trace(inv_at_a))
            if gdop is not None:
                return gdop
            else:
                return float('inf')
        except np.linalg.LinAlgError:
            return float('inf')  # Matrix is singular, cannot compute GDOP

    def compute_FIM(self, measurements, target_coords, noise_variance = 0.2):
        """Compute the Fisher Information Matrix (FIM) given a set of measurements corresponding to drone positions in space, and the target coordinates of the anchor to relate the measurements to
        
        Parameters:
        - measurements: list of lists, the measurements to use for the FIM calculation where each line is of the form [x, y, z, distance]
        - target_coords: list of floats, the target coordinates of the anchor to relate the measurements to [x, y, z]

        Returns:
        - FIM: numpy array, the computed FIM
        """

        x, y, z = target_coords  # Define the target coordinates
        FIM = np.zeros((3, 3))  # Initialize the FIM matrix
        
        for measurement in measurements:
            x_i, y_i, z_i, dist_i = measurement
            d_i = np.sqrt((x - x_i)**2 + (y - y_i)**2 + (z - z_i)**2)
            
            if d_i == 0:
                continue  # Avoid division by zero

            # Jacobian of the distance with respect to the anchor position
            jacobian = (np.array([x_i, y_i, z_i]) - np.array([x, y, z])) / d_i

            # Update FIM
            FIM += (1 / (noise_variance * (1 + d_i**2))) * np.outer(jacobian, jacobian)
        
        return FIM

    def compute_residuals(self, measurements, estimator):
        """Compute the residuals of the linear least squares problem given a set of measurements and an estimator computed from the least squares
        
        Parameters:
        - measurements: list of lists, the measurements to use for the residuals calculation where each line is of the form [x, y, z, distance]
        - estimator: numpy array, the estimator computed from the linear least squares
        
        Returns:
        - residuals: numpy array, the computed residuals
        """

        A, b = self.setup_linear_least_square(measurements)

        x_hat = np.linalg.lstsq(A, b, rcond=None)[0]
        y_hat = A @ x_hat

        # Compute the residuals
        residuals = b - y_hat

        return residuals

    def compute_z_score(self, residuals):
        """Compute the z-score of the residuals from the linear least squares problem
        
        Parameters:
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - z_scores: numpy array, the computed z-scores
        """
        
        mean_residuals = np.mean(residuals)
        std_residuals = np.std(residuals)
        z_scores = (residuals - mean_residuals) / std_residuals

        return z_scores

    def compute_condition_number(self, measurements):
        """Compute the condition number of the linear least squares A matrix given a set of measurements
        
        Parameters:
        - measurements: list of lists, the measurements to use for the condition number calculation where each line is of the form [x, y, z, distance]
        
        Returns:
        - condition_number: float, the computed condition number
        """

        A, _ = self.setup_linear_least_square(measurements)
        U, s, Vt = np.linalg.svd(A)
        sigma_max = max(s)
        sigma_min = min(s)
        condition_number = sigma_max / sigma_min

        # condition_number = np.linalg.cond(A)
        
        return condition_number



    def estimate_anchor_position_linear_least_squares(self, measurements):
        """Estimate the position of an anchor given distance measurements fron different drone positions using linear least squares optimization depending on the parameters set in the class for bias usage
        
        Parameters:
        - measurements: list of lists, the measurements to use for the linear least squares problem where each line is of the form [x, y, z, distance]
        
        Returns:
        - estimator: numpy array, the estimated position of the anchor and the bias terms if used
        - covariance_matrix: numpy array, the covariance matrix of the linear least squares estimation
        - residuals: numpy array, the residuals of the linear least squares estimation
        - x: numpy array, the estimated parameters of the linear least squares estimation, as it is setup
        """

        use_constant_bias = self.params['use_constant_bias']
        use_linear_bias = self.params['use_linear_bias']

        def compute_covariance_matrix_linear_least_squares(A, b, x):
            """Compute the covariance matrix of the parameters for linear least squares
            
            Parameters:
            - A: numpy array, the matrix A of the linear least squares problem
            - b: numpy array, the vector b of the linear least squares problem
            - x: numpy array, the estimated parameters of the linear least squares problem
            
            Returns:
            - covariance_matrix: numpy array, the computed covariance matrix
            """
            
            residual_sum_of_squares = np.sum((b - A @ x)**2)
            dof = A.shape[0] - A.shape[1]
            sigma_hat_squared = residual_sum_of_squares / dof
            covariance_matrix = sigma_hat_squared * np.linalg.pinv(A.T @ A)

            return covariance_matrix
        
        def compute_residuals(A, b, x):
            """Compute the residuals of the linear least squares problem given the matrix A, the vector b and the estimated parameters x

            Parameters:
            - A: numpy array, the matrix A of the linear least squares problem
            - b: numpy array, the vector b of the linear least squares problem
            - x: numpy array, the estimated parameters of the linear least squares problem

            Returns:
            - residuals: numpy array, the computed residuals
            """

            return np.array((b - A @ x))

        A, b = self.setup_linear_least_square(measurements)

        if use_constant_bias and use_linear_bias:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            squared_linear_bias = x[3] if x[3] > 0 else 1
            linear_bias = np.sqrt(1/squared_linear_bias)

            bias = x[4]/squared_linear_bias
            position = x[:3]

            return np.concatenate((position, [bias, linear_bias])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x
        
        elif use_constant_bias:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            position = x[:3]
            bias = x[3]

            return np.concatenate((position, [bias, 1])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x
        
        elif use_linear_bias:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            squared_linear_bias = x[3] if x[3] > 0 else 1
            linear_bias = np.sqrt(1/squared_linear_bias)
            position = x[:3]

            return np.concatenate((position, [0, linear_bias])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x
        
        else:

            x = np.linalg.lstsq(A, b, rcond=None)[0]
            #x = np.linalg.inv(A.T @ A) @ A.T @ b
            position = x[:3]

            return np.concatenate((position, [0, 1])), compute_covariance_matrix_linear_least_squares(A, b, x), compute_residuals(A, b, x), x

    def estimate_anchor_position_non_linear_least_squares(self, measurements, initial_guess=[0, 0, 0, 0, 0], max_iterations=10, tol=1e-6, alpha=1.0, sigma=1.0):
        """Estimate the position of an anchor given distance measurements fron different drone positions using non lienar ptimization
        
        Parameters:
        - measurements: list of lists, the measurements to use for the non linear least squares problem where each line is of the form [x, y, z, distance]
        - initial_guess: list of floats, the initial guess for the non linear least squares optimization, i.e [x, y, z, bias, linear_bias]
        - max_iterations: int, the maximum number of iterations for the optimization if using IRLS
        - tol: float, the tolerance for the optimization if using IRLS
        - alpha: float, the regularization parameter for the KRR model
        - sigma: float, the kernel width for the KRR model

        Returns:
        - estimator: numpy array, the estimated position of the anchor and the bias terms if used
        - covariance_matrix: numpy array, the covariance matrix of the non linear least squares estimation
        """
        
        measurements = np.array(measurements)

        optimisation_type = self.params["non_linear_optimisation_type"]

        if optimisation_type == "LM":
            # Define the loss function and its Jacobian
            def loss_function(x0, measurements):
                residuals = []
                J = []  # Jacobian matrix
                guess = x0[0:3]
                bias = x0[3]
                linear_bias = x0[4]
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[0:4]
                    distance_vector = np.array([guess[0] - x, guess[1] - y, guess[2] - z])
                    distance = np.linalg.norm(distance_vector)
                    estimated_dist = distance * (linear_bias) + bias
                    residual = estimated_dist - measured_dist
                    residuals.append(residual)

                    # Jacobian calculation
                    if distance != 0:  # to avoid division by zero
                        J_row = np.zeros(5)
                        J_row[0:3] = (linear_bias) * (distance_vector / distance)
                        J_row[3] = 1  # bias term
                        J_row[4] = distance  # linear_bias term
                        J.append(J_row)

                return residuals, J

            # Perform least squares optimization and calculate Jacobian at solution
            result = least_squares(
            lambda x0, measurements: loss_function(x0, measurements)[0],
            initial_guess,
            args=(measurements,),
            jac=lambda x0, measurements: np.array(loss_function(x0, measurements)[1]),
            method='lm')
            
            # Calculate covariance matrix from the Jacobian at the last evaluated point
            J = np.array(loss_function(result.x, measurements)[1])

            # Residual sum of squares
            residual_sum_of_squares = np.sum(result.fun**2)

            # Degrees of freedom: number of data points minus number of parameters
            dof = len(measurements) - len(result.x)

            # Estimate of the error variance
            sigma_hat_squared = residual_sum_of_squares / dof

            # Covariance matrix of the parameters
            cov_matrix = sigma_hat_squared * np.linalg.inv(J.T @ J)


            return result.x, cov_matrix
        

        elif optimisation_type == "IRLS":

            def loss_function(params, measurements):
                x_a, y_a, z_a, beta, gamma = params
                anchor_pos = np.array([x_a, y_a, z_a])
                residuals = []
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[:4]
                    distance_vector = anchor_pos - np.array([x, y, z])
                    distance = np.linalg.norm(distance_vector)
                    estimated_dist = beta * distance + gamma
                    residual = measured_dist - estimated_dist
                    residuals.append(residual)
                return np.array(residuals)
            
            def jacobian(params, measurements):
                x_a, y_a, z_a, beta, gamma = params
                anchor_pos = np.array([x_a, y_a, z_a])
                J = []
                for measurement in measurements:
                    x, y, z, measured_dist = measurement[:4]
                    distance_vector = anchor_pos - np.array([x, y, z])
                    distance = np.linalg.norm(distance_vector)
                    if distance == 0:
                        continue
                    J_row = np.zeros(5)
                    J_row[0:3] = -beta * (distance_vector / distance)
                    J_row[3] = -distance
                    J_row[4] = -1
                    J.append(J_row)
                return np.array(J)
            
            def compute_mad(residuals):
                median = np.median(residuals)
                mad = np.median(np.abs(residuals - median))
                return mad

            def huber_weights_function(residuals, delta=0.01):
                """
                Compute Huber weights for the residuals.
                
                Parameters:
                - residuals: array-like, the residuals of the model.
                - delta: float, the threshold parameter where the function switches from quadratic to linear.
                
                Returns:
                - weights: array-like, the computed weights.
                """
                abs_residuals = np.abs(residuals)
                weights = np.where(abs_residuals <= delta, 1, delta / abs_residuals)
                return weights
            
            params = np.array(initial_guess)
            
            for i in range(max_iterations):
                residuals = loss_function(params, measurements)

                # Compute the Median Absolute Deviation (MAD)
                mad = compute_mad(residuals)
                
                # Set delta dynamically based on MAD
                k = 3
                delta = k * mad
                # Calculate weights using the dynamic delta
                weights = huber_weights_function(residuals, delta)

                sqrt_weights = np.sqrt(weights)
                
                weighted_residuals = sqrt_weights * residuals
                weighted_J = sqrt_weights[:, np.newaxis] * jacobian(params, measurements)
                
                # Perform the weighted least squares optimization
                result = least_squares(lambda p: sqrt_weights * loss_function(p, measurements), params, jac=lambda p: sqrt_weights[:, np.newaxis] * jacobian(p, measurements), method='lm')
                
                new_params = result.x
                if np.linalg.norm(new_params - params) < tol:
                    break
                params = new_params
            
            # Compute the final covariance matrix
            J_final = jacobian(params, measurements)
            residual_sum_of_squares = np.sum(loss_function(params, measurements)**2)
            dof = len(measurements) - len(params)
            sigma_hat_squared = residual_sum_of_squares / dof
            cov_matrix = sigma_hat_squared * np.linalg.pinv(J_final.T @ J_final)
            

            return params, cov_matrix
        
        elif optimisation_type == "KRR":

            positions = measurements[:, :3]
            distances = measurements[:, 3]

            krr = KernelRidge(kernel='rbf', alpha=1.0, gamma=10)

            # Train the model
            krr.fit(positions, distances)

            def objective_function(anchor_position, krr_model):
                # Predict the range using the KRR model
                predicted_range = krr_model.predict([anchor_position])[0]
                # Since we want the range to be as close to 0 as possible, we return the absolute value of the predicted range
                return np.abs(predicted_range)
            
            result = minimize(objective_function, initial_guess[:3], args=(krr))

            estimated_anchor_position = result.x

            def fit_bias_model(anchor_position, measurements):

                true_distances = np.linalg.norm(positions - anchor_position, axis=1)

                X = true_distances.reshape(-1, 1)
                y = distances
                model = LinearRegression()
                model.fit(X, y)

                beta = model.coef_[0]
                gamma = model.intercept_

                return beta, gamma
            
            beta, gamma = fit_bias_model(estimated_anchor_position, measurements)

            estimator = np.concatenate((estimated_anchor_position, [beta, gamma]))

            cov_matrix = np.zeros((3,3))
            # def rbf_kernel(x1, x2, sigma=1.0):
            #     return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * sigma**2))

            # def kernel_matrix(X, kernel_func, **kwargs):
            #     n_samples = X.shape[0]
            #     K = np.zeros((n_samples, n_samples))
            #     for i in range(n_samples):
            #         for j in range(n_samples):
            #             K[i, j] = kernel_func(X[i], X[j], **kwargs)
            #     return K

            # def kernel_ridge_regression(K, y, alpha=10.0):
            #     n_samples = K.shape[0]
            #     I = np.eye(n_samples)
            #     return np.linalg.inv(K + alpha * I) @ y

            # def predict(K_train, K_test, alpha, y_train):
            #     alpha_vec = kernel_ridge_regression(K_train, y_train, alpha)
            #     return K_test @ alpha_vec

            # def fit_bias_model(anchor_position, measurements):
            #     positions = measurements[:, :3]
            #     measured_distances = measurements[:, 3]

            #     true_distances = np.linalg.norm(positions - anchor_position, axis=1)

            #     X = true_distances.reshape(-1, 1)
            #     y = measured_distances
            #     model = LinearRegression()
            #     model.fit(X, y)

            #     beta = model.coef_[0]
            #     gamma = model.intercept_

            #     return beta, gamma

            # positions = measurements[:, :3]
            # distances = measurements[:, 3]

            # K_train = kernel_matrix(positions, rbf_kernel, sigma=sigma)

            # def loss_function(params):
            #     params = params[:3]
            #     K_test = np.array([rbf_kernel(params, pos, sigma=sigma) for pos in positions])
            #     estimated_distances = predict(K_train, K_test, alpha, distances)
            #     residuals = distances - estimated_distances
            #     return np.sum(residuals**2)

            # result = minimize(loss_function, initial_guess, method='L-BFGS-B')
            # anchor_position = result.x[:3]

            # beta, gamma = fit_bias_model(anchor_position, measurements)

            # estimator = np.concatenate((anchor_position, [beta, gamma]))

            # # Calculate residuals
            # K_test = np.array([rbf_kernel(anchor_position, pos, sigma=sigma) for pos in positions])
            # estimated_distances = predict(K_train, K_test, alpha, distances)
            # residuals = distances - estimated_distances

            # # Residual sum of squares
            # residual_sum_of_squares = np.sum(residuals**2)

            # # Degrees of freedom: number of data points minus number of parameters
            # dof = len(measurements) - len(estimator)

            # # Estimate of the error variance
            # sigma_hat_squared = residual_sum_of_squares / dof

            # # Calculate the Jacobian matrix
            # def jacobian(params):
            #     params = params[:3]
            #     J = np.zeros((len(measurements), 5))
            #     for i, pos in enumerate(positions):
            #         distance_vector = params - pos
            #         distance = np.linalg.norm(distance_vector)
            #         J[i, :3] = (distance_vector / distance) * beta if distance != 0 else 0
            #         J[i, 3] = 1  # bias term
            #         J[i, 4] = distance  # linear_bias term
            #     return J

            # J = jacobian(anchor_position)

            # # Covariance matrix of the parameters
            # cov_matrix = sigma_hat_squared * np.linalg.inv(J.T @ J)

            return estimator, cov_matrix
        


    def measurement_callback(self, drone_position, drone_progress_waypoint_idx):
        """Callback function that is called every time a new drone position is reached, which triggers the UWB measurements
        The function processes the measurements and depending on the status of initialisation of the anchors, decides what to do with the measurements
        I.e run linear least squares, non-linear least squares, or optimise and update the trajectory
        
        Parameters:
        - drone_position: list of floats, the current drone position [x, y, z]"""
        drone_x, drone_y, drone_z = drone_position

        # Get new distance measurements from the anchors
        known_anchor_distances, unknown_anchor_distances = self.get_distance_to_anchors(drone_x, drone_y, drone_z)

        # # Add noise to x, y, z components
        # noise_x = np.random.normal(0, 0.005)
        # noise_y = np.random.normal(0, 0.005)
        # noise_z = np.random.normal(0, 0.005)

        # drone_x += noise_x
        # drone_y += noise_y
        # drone_z += noise_z

        # drone_position = [drone_x, drone_y, drone_z]
        

        ### 1st step, rough initialisation of the unknown anchors
        
        # Iterate over the unkown anchors in range
        for distance, anchor_id in unknown_anchor_distances: 

            self.anchor_rough_estimate_found.setdefault(anchor_id, False) # If it is the first time we see it, set it up in the status dictionary
            self.all_optimal_measurements_collected.setdefault(anchor_id, False)


            # If the rough estimate for this anchor is not yet found
            if not self.anchor_rough_estimate_found[anchor_id]:

                if not self.process_measurement(drone_position, distance, anchor_id): # Decide what to do with the measurement, if we choose not to add it, exit and go to the other anchors
                    continue
                
                anchor_measurement_dictionary = self.unknown_anchor_measurements[anchor_id] # Extract the anchor's dictionnary 
                number_of_measurements = len(anchor_measurement_dictionary["distances_pre_rough_estimate"])

                if number_of_measurements > 5: # If we have more then 4 measurements, the least square is overdetermined for all situations, and we can therefore calculate quantities of interest
                    
                    # Transform the measurements from the positions and range measurements to tuples
                    measurements = []
                    for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                        x, y, z = position
                        measurements.append([x, y, z, distance])

                    # Run the linear least squares estimation and compute the residuals
                    estimator, covariance_matrix, residuals, _ = self.estimate_anchor_position_linear_least_squares(measurements)

                    # Compute the stopping criterion variables
                    FIM = self.compute_FIM(measurements, estimator[:3])
                    GDOP = self.compute_GDOP(measurements, estimator[:3])
                    rmse_residuals = np.sqrt(np.mean(residuals))
                    condition_number = self.compute_condition_number(measurements)
                    covariances = np.diag(covariance_matrix)

                    # Add them to the dictionnary for plotting
                    anchor_measurement_dictionary["FIM"].append(FIM)
                    anchor_measurement_dictionary["GDOP"].append(GDOP)
                    anchor_measurement_dictionary["residuals"].append(rmse_residuals)
                    anchor_measurement_dictionary["condition_number"].append(condition_number)
                    anchor_measurement_dictionary["covariances"].append(covariances)

                    # Update the estimated position of the anchor
                    anchor_measurement_dictionary["estimated_position_rough_linear"] = estimator[:3]
                    self.error = self.calculate_position_error([anchor for anchor in self.unknown_anchors if anchor.anchor_ID == anchor_id][0].get_anchor_coordinates(), anchor_measurement_dictionary["estimated_position"])
                    if self.outlier_filtering(anchor_measurement_dictionary, residuals):
                        continue
                    # Check if the stopping criterion is achieved based on the variables calculated above
                    if self.stopping_criterion_check(number_of_measurements, anchor_measurement_dictionary):

                        self.anchor_rough_estimate_found[anchor_id] = True # Can set the status of the anchor to confirm a rough estimate is available
                        self.drone_position = drone_position # Save the drone position at which the rough estimate was calculated

                        # Update the dictionnary with the linear rough estimate
                        anchor_measurement_dictionary["estimator_rough_linear"] = estimator
                        anchor_measurement_dictionary["estimated_position_rough_linear"] = np.array(anchor_measurement_dictionary["estimator_rough_linear"][:3])

                        # Refine the linear rough estimate with the non-linear rough estimate using the linear as an intial guess
                        estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_linear"])

                        # Update the dictionnary with the non-linear refined rough estimate
                        anchor_measurement_dictionary["estimator_rough_non_linear"] = estimator
                        anchor_measurement_dictionary["estimated_position_rough_non_linear"] = np.array(anchor_measurement_dictionary["estimator_rough_non_linear"][:3])

                        # Update the error in position to plot it
                        self.error = self.calculate_position_error([anchor for anchor in self.unknown_anchors if anchor.anchor_ID == anchor_id][0].get_anchor_coordinates(), anchor_measurement_dictionary["estimated_position_rough_non_linear"])




                        # 2nd step: optimise the trajectory
                        # Skip for now
                        """
                        remaining_waypoints = self.trajectory_waypoints[drone_progress_waypoint_idx:]

                        self.optimiser.update_anchor_positions(anchor_measurement_dictionary["estimated_position_rough_non_linear"])

                        self.optimiser.update_waypoints(remaining_waypoints)

                        # self.optimiser.update_weights(weights)

                        self.optimised_trajectory_waypoints = self.optimiser.run_optimization()

                        self.spline_x, self.spline_y, self.spline_z = self.optimiser.create_optimised_trajectory(self.drone_position ,self.optimised_trajectory_waypoints, drone_progress_waypoint_idx, self.trajectory_waypoints)
                        """

                    
                else: # Number of measurements not enough to perform the stopping criterion analysis
                    continue
                
            # If the rough estimate for this anchor was already found and therefore the drone is following an optimised trajectory    
            elif not self.all_optimal_measurements_collected[anchor_id]:
                
                if drone_position in self.optimised_trajectory_waypoints: # The drone is at one of the optimal waypoints
                    for distance, anchor_id in unknown_anchor_distances:
                        anchor_measurement_dictionary = self.unknown_anchor_measurements[anchor_id]

                        anchor = self.unknown_anchors[0]
                        for i in range(self.params["number_of_redundant_measurements"]):
                            
                            distance = anchor.request_distance(drone_x, drone_y, drone_z)

                            anchor_measurement_dictionary["distances_post_rough_estimate"].append(distance)
                            anchor_measurement_dictionary["positions_post_rough_estimate"].append((drone_x, drone_y, drone_z))

                    if (np.array(drone_position) == self.optimised_trajectory_waypoints[-2]).all():
                        self.all_optimal_measurements_collected[anchor_id] = True
                        

                        measurements = []
                        for distance, position in zip(anchor_measurement_dictionary["distances_pre_rough_estimate"]+anchor_measurement_dictionary["distances_post_rough_estimate"], anchor_measurement_dictionary["positions_pre_rough_estimate"] + anchor_measurement_dictionary["positions_post_rough_estimate"]):
                            x, y, z = position
                            measurements.append([x, y, z, distance])

                        estimator, covariance_matrix = self.estimate_anchor_position_non_linear_least_squares(measurements, initial_guess=anchor_measurement_dictionary["estimator_rough_non_linear"])

                        anchor_measurement_dictionary["estimator"] = estimator
                        anchor_measurement_dictionary["estimated_position"] = np.array(anchor_measurement_dictionary["estimator"][:3])

                        self.error = self.calculate_position_error(self.unknown_anchors[0].get_anchor_coordinates(), anchor_measurement_dictionary["estimated_position"])

            else: # The anchor is in fact initialised so new measurements should be used for localisation and not initialisation
                pass

    def process_measurement(self, drone_position, distance, anchor_id):
        """Process a new measurement and decide whether or not to add it to the dictionnary of measurements for the anchor
        
        Parameters:
        - drone_position: list of floats, the current drone position [x, y, z]
        - distance: float, the measured distance to the anchor
        - anchor_id: int, the ID of the anchor
        
        Returns:
        - bool, True if the measurement was added, False if it was not added
        """
        
        drone_x, drone_y, drone_z = drone_position

        anchor_measurement_dictionary = self.unknown_anchor_measurements.setdefault(anchor_id, self.default_anchor_structure)

        # Reject measurements that are too far away
        if distance > self.params["distance_rejection_threshold"]:
            return False # We did not add a measurement
        
        
        # I we have no measurements yet, or if we decide to gather mutiple measurements at points very close to each other for robustness, add the measurement to the dictionnary
        if (len(anchor_measurement_dictionary["distances_pre_rough_estimate"]) % self.params["number_of_redundant_measurements"] != 0) or len(anchor_measurement_dictionary["distances_pre_rough_estimate"])  == 0:

            anchor_measurement_dictionary["distances_pre_rough_estimate"].append(distance)
            anchor_measurement_dictionary["positions_pre_rough_estimate"].append((drone_x, drone_y, drone_z))

        # If the drone has moved significantly since the last measurement, add the new measurement to the dictionnary
        elif np.linalg.norm(np.array(anchor_measurement_dictionary["positions_pre_rough_estimate"][-1]) - np.array((drone_x, drone_y, drone_z))) / distance > self.params["distance_to_anchor_ratio_threshold"]:
            anchor_measurement_dictionary["distances_pre_rough_estimate"].append(distance)
            anchor_measurement_dictionary["positions_pre_rough_estimate"].append((drone_x, drone_y, drone_z))

        else: 
            return False # No measurement was added
        
        return True # A measurement was added
    
    def stopping_criterion_check(self, number_of_measurements, anchor_meas_dictionnary):
        """Check if the stopping criterion is met based on the variables computed during the initialisation process to decide when to stop collecting measurements

        Parameters:
        - number_of_measurements: int, the number of measurements collected for the anchor
        - anchor_meas_dictionnary: dict, the dictionnary containing the measurements for the anchor
        
        Returns:
        - bool, True if the stopping criterion is met, False if it is not met
        """

        GDOP_k_prev , GDOP_k = anchor_meas_dictionnary["GDOP"][-2], anchor_meas_dictionnary["GDOP"][-1]
        FIM_k_prev , FIM_k = anchor_meas_dictionnary["FIM"][-2], anchor_meas_dictionnary["FIM"][-1]
        sum_of_res_k_prev , sum_of_res_k = anchor_meas_dictionnary["residuals"][-2], anchor_meas_dictionnary["residuals"][-1]
        condition_num_k_prev, condition_num_k = anchor_meas_dictionnary["condition_number"][-2], anchor_meas_dictionnary["condition_number"][-1]
        covariances_k_prev , covariances_k = anchor_meas_dictionnary["covariances"][-2], anchor_meas_dictionnary["covariances"][-1]
        
        GDOP_ratio = np.absolute(GDOP_k - GDOP_k_prev) / GDOP_k_prev if GDOP_k_prev != 0 and not np.isinf(GDOP_k_prev) else float('inf')

        # Options would be to look at the eigenvalues, and their range. If the range is big it means that the guess is bad or the measurements do not capture a good range of information 
        # condition number of the FIM, if it is too high, the matrix is ill-conditioned and the solution is not reliable
        # CRLB, if the CRLB is too high, the solution is not reliable
        FIM_ratio = np.absolute(np.linalg.det(FIM_k) - np.linalg.det(FIM_k_prev)) / np.linalg.det(FIM_k_prev) if np.linalg.det(FIM_k_prev) != 0 and not np.isinf(np.linalg.det(FIM_k_prev)) else float('inf')

        sum_of_res_ratio = np.absolute(sum_of_res_k - sum_of_res_k_prev) / sum_of_res_k_prev if sum_of_res_k_prev != 0 and not np.isinf(sum_of_res_k_prev) else float('inf')
        condition_num_ratio = np.absolute(condition_num_k - condition_num_k_prev) / condition_num_k_prev if condition_num_k_prev != 0 and not np.isinf(condition_num_k_prev) else float('inf')
        covariances_ratio = np.absolute(np.array(covariances_k[:3]) - np.array(covariances_k_prev[:3])) / np.array(covariances_k_prev[:3]) if np.array(covariances_k_prev[:3]).any() != 0 and not np.isinf(np.array(covariances_k_prev[:3])).any() else [float('inf'),float('inf'),float('inf')]
        
        if GDOP_ratio < self.params["GDOP_ratio_thresh"] and GDOP_k < self.params["GDOP_thresh"]:
            print("GDOP criterion met")
            return True
        
        if FIM_ratio < self.params["FIM_ratio_thresh"] and np.linalg.det(FIM_k) > self.params["FIM_thresh"]:
            print("FIM criterion met")
            return True
        
        if sum_of_res_ratio < self.params["residuals_ratio_thresh"] and sum_of_res_k < self.params["residuals_thresh"]:
            print("Sum of residuals criterion met")
            return True
        
        if condition_num_ratio < self.params["condition_number_ratio_thresh"] and condition_num_k < self.params["condition_number_thresh"]:
            print("Condition number criterion met")
            return True
        
        if max(covariances_ratio) < self.params["covariance_ratio_thresh"] and max(covariances_k) < self.params["covariance_thresh"]:
            print("Covariance criterion met")
            return True
        
        if number_of_measurements >= self.params["number_of_measurements_thresh"]:
            print("Number of measurements criterion met")
            return True

        return False

    

    def outlier_finder(self, residuals):
        """Find the outliers in the residuals using the z-score method
        
        Parameters:
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - outliers: numpy array, the indices of the outliers in the residuals
        """

        z_score = self.compute_z_score(residuals)

        outliers = np.where(np.abs(z_score) > self.params["z_score_threshold"])[0]
        
        return outliers

    def cluster_measurements(self, measurements, number_of_clusters=15):
        """Cluster the measurements using the KMeans algorithm

        Parameters:
        - measurements: numpy array, the measurements to cluster
        - number_of_clusters: int, the number of clusters to use for the KMeans algorithm

        Returns:
        - centroids: numpy array, the centroids of the clusters
        """

        kmeans = KMeans(n_clusters=number_of_clusters)
        kmeans.fit_predict(measurements)
        centroids = kmeans.cluster_centers_

        return centroids

    def outlier_filtering(self, anchor_measurement_dictionary, residuals):
        """Filter the outliers in the measurements using the z-score method
        
        Parameters:
        - anchor_measurement_dictionary: dict, the dictionnary containing the measurements for the anchor
        - residuals: numpy array, the residuals of the linear least squares problem
        
        Returns:
        - bool, True if outliers were found and removed, False if no outliers were found
        """
        
        outliers = self.outlier_finder(residuals)

        anchor_measurement_dictionary["distances_pre_rough_estimate"] = np.delete(anchor_measurement_dictionary["distances_pre_rough_estimate"], outliers).tolist()
        anchor_measurement_dictionary["positions_pre_rough_estimate"] = np.delete(anchor_measurement_dictionary["positions_pre_rough_estimate"], outliers, axis=0).tolist()

        if outliers.size > 0:
            return True
        else:
            return False
    






    def calculate_position_error(self, real_positions, estimated_positions):
            """Calculate the error between the estimated anchor position and the real ground truth anchor position
            Can be used for one or multiple anchors at once
            
            Parameters:
            - real_positions: numpy array, the real ground truth positions of the anchors
            - estimated_positions: numpy array, the estimated positions of the anchors
            
            Returns:
            - error: float, the error between the estimated and real positions
            """

            actual_position = np.array([(a.x, a.y, a.z) for a in self.unknown_anchors])
            error = np.linalg.norm(estimated_positions - actual_position)
            return error
    
    def calculate_estimator_error(self, real_anchor_values, estimator):
        """Calculate the error between the estimated anchor estimator and the real ground truth anchor characteristics
        Can be used for one or multiple anchors at once
        The estimator contains the position of the anchor and the bias and linear bias
        
        Parameters:
        - real_anchor_values: numpy array, the real ground truth values of the anchors
        - estimator: numpy array, the estimated values of the anchors
        """
        
        actual_position = np.array([(a.x, a.y, a.z, a.bias, a.linear_bias) for a in self.unknown_anchors])
        error = np.linalg.norm(estimator - actual_position)

        return error
    







    ### Helper functions for quickly computing the anchor initialisation without relying on the simulator to run real time

    def reset_measurements_pre_rough_initialisation(self, anchor_id):
        """Reset the measurements in the dictionnary that were gathered for the rough estimation of the anchor
        
        Parameters:
        - anchor_id: int, the ID of the anchor
        """

        self.unknown_anchor_measurements[anchor_id]["distances_pre_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["positions_pre_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["estimator_rough_linear"] = [0, 0, 0, 0, 0]
        self.unknown_anchor_measurements[anchor_id]["estimated_position_rough_linear"] = []
        self.unknown_anchor_measurements[anchor_id]["covariance_matrix_rough_linear"] = []
        self.unknown_anchor_measurements[anchor_id]["estimator_rough_non_linear"] = [0, 0, 0, 0, 0]
        self.unknown_anchor_measurements[anchor_id]["estimated_position_rough_non_linear"] = []
        self.unknown_anchor_measurements[anchor_id]["covariance_matrix_rough_non_linear"] = []

    def reset_measurements_post_rough_initialisation(self, anchor_id):
        """Reset the measurements in the dictionnary that were gathered after the rough estimation of the anchor from the optimal waypoints
        
        Parameters:
        - anchor_id: int, the ID of the anchor
        """

        self.unknown_anchor_measurements[anchor_id]["distances_post_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["positions_post_rough_estimate"] = []
        self.unknown_anchor_measurements[anchor_id]["estimator"] = [0, 0, 0, 0, 0]
        self.unknown_anchor_measurements[anchor_id]["estimated_position"] = []
        self.unknown_anchor_measurements[anchor_id]["covariance_matrix"] = []
        
    def reset_all_measurements(self, anchor_id):
        """Reset all the measurements in the dictionnary for a specific anchor
        
        Parameters:
        - anchor_id: int, the ID of the anchor"""

        self.unknown_anchor_measurements[anchor_id] = self.default_anchor_structure
    