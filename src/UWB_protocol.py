import numpy as np

class Anchor:
    def __init__(self, x, y, z, bias = 0, linear_bias = 0, noise_variance = 0.1):
        self.x = x
        self.y = y
        self.z = z
        self.bias = bias
        self.linear_bias = linear_bias
        self.noise_variance = noise_variance

    def request_distance(self, x, y, z):
        # Simulate the distance measurement
        distance = np.linalg.norm([self.x - x, self.y - y, self.z - z])
        # Add constant bias
        distance += self.bias
        # Add linear bias
        distance += self.linear_bias * np.linalg.norm([self.x - x, self.y - y, self.z - z])
        # Add noise
        distance += np.random.normal(0, self.noise_variance)
        
        return distance
    
