import numpy as np

class Anchor:
    def __init__(self, ID, x, y, z, bias = 0, linear_bias = 0, noise_variance = 0.1, generate_outliers = False, outlier_probability = 0.0005, outlier_factor_range = [1.5, 3]):
        self.anchor_ID = ID
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.bias = float(bias)
        self.linear_bias = float(linear_bias)
        self.noise_variance = float(noise_variance)
        self.generate_outliers = generate_outliers
        self.outlier_probability = outlier_probability
        self.outlier_factor_range_min, self.outlier_factor_range_max = outlier_factor_range

    def request_distance(self, x, y, z):
        
        # Simulate the distance measurement
        distance = np.linalg.norm([self.x - x, self.y - y, self.z - z])
        # Add constant bias
        distance += self.bias
        # Add linear bias
        distance += self.linear_bias * np.linalg.norm([self.x - x, self.y - y, self.z - z])
        # Add noise
        distance += np.random.normal(0, self.noise_variance)
        
        if self.generate_outliers and np.random.rand() < self.outlier_probability: # Generate an outlier distance
            factor = np.random.uniform(self.outlier_factor_range_min, self.outlier_factor_range_max)
            distance *= factor

        return distance
    
    def get_anchor_coordinates(self):
        return self.x, self.y, self.z
