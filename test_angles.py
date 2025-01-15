import numpy as np

def normalize_angle(angle):
    """Normalize an angle to the range [0, 2*pi) in radians."""
    return angle % (2 * np.pi)

def normalize_interval(a, b):
    """Normalize an interval such that if a > b, subtract 2π from a to handle wrap-around."""
    if a > b:
        a -= 2 * np.pi
    return a, b

def intersect_intervals(interval1, interval2):
    """Find the intersection of two intervals [A, B] and [C, D] in the unwrapped space."""
    A, B = interval1
    C, D = interval2
    intersection_start = max(A, C)
    intersection_end = min(B, D)
    if intersection_start < intersection_end:
        return intersection_start, intersection_end
    else:
        return None, None

def adjust_if_first_bigger(a, b):
    """
    If the first angle (a) is greater than the second angle (b),
    subtract 2π from the first angle (a).
    
    Returns:
        Tuple of adjusted angles (a, b).
    """
    if a > b:
        a -= 2 * np.pi
    return a, b


def intersect_phi_intervals(phi1_min, phi1_max, phi2_min, phi2_max):
    """Find the intersection of two angular intervals [phi1_min, phi1_max] and [phi2_min, phi2_max].
    
    Returns:
        (interval_low, interval_high) if they intersect, normalized back to [0, 2π],
        (None, None) if no intersection exists.
    """

    # Normalize the angles
    phi1_min = normalize_angle(phi1_min)
    phi1_max = normalize_angle(phi1_max)
    phi2_min = normalize_angle(phi2_min)
    phi2_max = normalize_angle(phi2_max)

    # Normalize both intervals (handles wrap-around)
    A, B = normalize_interval(phi1_min, phi1_max)
    C, D = normalize_interval(phi2_min, phi2_max)

    # Shift intervals based on comparison between A and C
    if A < C:
        E, F = A + 2 * np.pi, B + 2 * np.pi
    else:
        E, F = A - 2 * np.pi, B - 2 * np.pi

    # Calculate both intersections
    I_1 = intersect_intervals([A, B], [C, D])
    I_2 = intersect_intervals([E, F], [C, D])

    # Normalize intersections back to [0, 2π] if they exist
    if I_1[0] is not None:
        I_1 = I_1 # normalize_angle(I_1[0]), normalize_angle(I_1[1])
        return I_1
    elif I_2[0] is not None:
        I_2 = I_2 # normalize_angle(I_2[0]), normalize_angle(I_2[1])
        return I_2
    else:
        return None, None

print([0,0,0] == [])
# Example usage
phi1_min, phi1_max = -np.pi/2, np.pi/2
phi2_min, phi2_max = np.pi, 2*np.pi
print(intersect_phi_intervals(phi1_min, phi1_max, phi2_min, phi2_max))
