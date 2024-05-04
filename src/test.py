import numpy as np
from scipy.optimize import least_squares

def multilateration(known_anchor_positions, distances, unknown_distance):
    """
    Estimate the position of the unknown anchor using multilateration.
    
    Parameters:
        known_anchor_positions (array-like): Array of known anchor positions (x, y, z).
        distances (array-like): Array of distances from the known anchors to the moving point.
        unknown_distance (float): Distance from the unknown anchor to the moving point.
        
    Returns:
        estimated_position (array): Estimated position of the unknown anchor (x, y, z).
    """
    # Define the residual function for least squares optimization
    def residual(position):
        return np.linalg.norm(known_anchor_positions - position, axis=1) - distances
    
    # Add the distance from the unknown anchor to the residual function
    def residual_with_unknown_distance(position):
        return np.append(residual(position), np.linalg.norm(position) - unknown_distance)
    
    # Initial guess for the position of the unknown anchor (average of known anchor positions)
    initial_guess = np.mean(known_anchor_positions, axis=0)
    
    # Use least squares optimization to estimate the position
    result = least_squares(residual_with_unknown_distance, initial_guess)
    
    return result.x

# Example usage
known_anchor_positions = np.array([
    [0, 0, 0],  # Anchor 1 position (x, y, z)
    [1, 0, 0],  # Anchor 2 position
    [0, 1, 0]   # Anchor 3 position
])
distances = np.array([1, 1.5, 1.2])  # Distances from anchors to the moving point
unknown_distance = 2.0  # Distance from the unknown anchor to the moving point

estimated_position = multilateration(known_anchor_positions, distances, unknown_distance)
print("Estimated position of the unknown anchor:", estimated_position)
