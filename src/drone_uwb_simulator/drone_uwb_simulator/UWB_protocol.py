import numpy as np

class Anchor:
    """ Class representing an anchor in the UWB protocol."""
    def __init__(self, ID, x, y, z, bias = 0, linear_bias = 1, noise_variance = 0.1, outlier_probability = 0.0, outlier_factor_range = [1.5, 3]):
        self.anchor_ID = ID # Unique identifier of the anchor
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.bias = float(bias) # Constant bias
        self.linear_bias = float(linear_bias)
        self.noise_variance = float(noise_variance) # Variance of the noise in the distance measurement (suppose to be zero mean Gaussian)
        self.outlier_probability = float(outlier_probability) # Probability of generating an outlier
        self.outlier_factor_range_min, self.outlier_factor_range_max = outlier_factor_range 

    def request_distance(self, x, y, z):
        """ Simulates the distance measurement between the anchor and the point (x, y, z), by adding bias model, noise, and outliers."""
        # Simulate the distance measurement
        distance = 0
        # Add constant bias
        distance += self.bias
        # Add linear bias
        distance += self.linear_bias * np.linalg.norm([self.x - x, self.y - y, self.z - z])
        # Add noise
        distance += np.random.normal(0, self.noise_variance)
        
        if np.random.rand() < self.outlier_probability: # Generate an outlier distance
            factor = np.random.uniform(self.outlier_factor_range_min, self.outlier_factor_range_max)
            distance *= factor

        return distance
    
    def request_distance_gt(self, x, y, z):
        distance = np.linalg.norm([self.x - x, self.y - y, self.z - z])
        return distance

    def change_outlier_generation(self, outlier_probability = 0.1, outlier_factor_range = [1.5, 3]):
        self.outlier_probability = outlier_probability
        self.outlier_factor_range_min, self.outlier_factor_range_max = outlier_factor_range
        
    def get_anchor_coordinates(self):
        """ Returns the coordinates of the anchor. """
        return self.x, self.y, self.z
    
    def get_anchor_gt_estimator(self):
        """ Returns the coordinates of the anchor and the bias. Is used to compare the estimator to the ground truth."""
        return self.x, self.y, self.z, self.bias, self.linear_bias
