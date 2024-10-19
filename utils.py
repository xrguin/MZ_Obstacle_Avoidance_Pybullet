import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection

def generate_random_coords(num_coords, origins, radius, test_site_size):
    coords = np.zeros((num_coords, 2))
    for i in range(num_coords):
        while True:
            x = np.random.rand() * test_site_size
            y = np.random.rand() * test_site_size
            if np.sqrt((x - origins[i, 0])**2 + (y - origins[i, 1])**2) > radius:
                coords[i, :] = [x, y]
                break
    return coords

def generate_intercepting_path(ref_start, ref_goal, test_site_size, safe_radius):
    while True:
        dir2 = ref_goal - ref_start
        t = np.random.rand() * 0.5

        intercept_point = ref_start + t * dir2

        if np.all(intercept_point <= test_site_size):
            distance = np.linalg.norm(intercept_point - ref_start)
            new_start = generate_random_point_on_circle(intercept_point, distance, test_site_size)

            if np.linalg.norm(new_start - ref_start) > safe_radius:
                dir3 = intercept_point - new_start
                new_goal = intercept_point + dir3
                return new_start, new_goal, intercept_point

def generate_random_point_on_circle(center, radius, test_site_size):
    while True:
        theta = 2 * np.pi * np.random.rand()
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        
        if 0 <= x <= test_site_size and 0 <= y <= test_site_size:
            return np.array([x, y])

def generate_ego_robot_path(ally_start, enemy_start, intercept_point, test_site_size, safe_radius):
    max_attempts = 10000
    for _ in range(max_attempts):
        distance = np.linalg.norm(intercept_point - ally_start)
        
        vect_to_intercept = intercept_point - enemy_start
        vect_to_intercept = vect_to_intercept / np.linalg.norm(vect_to_intercept)
        
        angle = -np.pi * np.random.rand()
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], 
                                    [np.sin(angle), np.cos(angle)]])
        ego_start = enemy_start + 2 * distance * np.dot(vect_to_intercept, rotation_matrix)
        
        if (np.linalg.norm(ego_start - ally_start) > safe_radius and 
            np.linalg.norm(ego_start - enemy_start) > safe_radius):
            dir_vector = intercept_point - ego_start
            ego_goal = intercept_point + dir_vector
            
            if np.all((ego_goal >= 0) & (ego_goal <= test_site_size)):
                return ego_start, ego_goal
    
    raise ValueError(f"Unable to find a suitable configuration for ego robot after {max_attempts} attempts")

def plot_robot(ax, robot, frame_size):
    # Extract robot position and orientation
    x, y = robot.current_coord
    theta = robot.head_angle

    # Robot body
    body = Circle((x, y), frame_size/2, fill=False, color='black')
    ax.add_artist(body)

    # Robot direction indicator
    direction = Wedge((x, y), frame_size/2, np.degrees(theta)-30, np.degrees(theta)+30, width=frame_size/4)
    ax.add_artist(direction)

    # Robot safe zone
    safe_zone = Circle((x, y), robot.safe_radius, fill=True, alpha=0.1, color=robot.light_color[:3])
    ax.add_artist(safe_zone)

    # Robot detection zone
    detection_zone = Circle((x, y), robot.detection_radius, fill=True, alpha=0.05, color=robot.dark_color[:3])
    ax.add_artist(detection_zone)

    # Force vectors
    scale = frame_size * 2  # Adjust this to change the length of the force arrows
    
    # Attraction force (green)
    ax.arrow(x, y, robot.attraction_force[0]*scale, robot.attraction_force[1]*scale,
             head_width=frame_size/4, head_length=frame_size/4, fc='g', ec='g', alpha=0.6)
    
    # Repulsion force (red)
    ax.arrow(x, y, robot.repulsion_force[0]*scale, robot.repulsion_force[1]*scale,
             head_width=frame_size/4, head_length=frame_size/4, fc='r', ec='r', alpha=0.6)
    
    # Combined force (blue)
    ax.arrow(x, y, robot.combined_force[0]*scale, robot.combined_force[1]*scale,
             head_width=frame_size/4, head_length=frame_size/4, fc='b', ec='b', alpha=0.6)

    return ax

