import numpy as np


# Create a function to calculate the angle between three points
def calculate_angle(point_a, point_b, point_c):
    ba = point_a - point_b
    bc = point_c - point_b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
