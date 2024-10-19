import numpy as np
from math import atan2, cos, sin, pi

class MbRobot:
    def __init__(self):
        self.linear_velocity = 1
        self.rotation_angle = 0
        self.current_pose = np.zeros(4)
        self.current_coord = np.zeros(2)
        self.head_angle = 0
        self.angular_velocity = 0
        self.start = np.zeros(2)
        self.goal = np.zeros(2)
        self.waypoints = None
        self.light_color = None
        self.dark_color = None
        self.goal_radius = 0.5
        self.safe_radius = 0
        self.detection_radius = 0
        self.obstacle = None
        self.attraction_force = np.zeros(2)
        self.repulsion_force = np.zeros(2)
        self.combined_force = np.zeros(2)
        self.name = ""

    def bug_rotate(self, sample_time):
        previous_head_angle = self.head_angle
        target_point = self.goal
        obstacle_distances = np.linalg.norm(self.obstacle - self.current_coord.reshape(-1, 1), axis=0)
        
        min_distance = np.min(obstacle_distances)
        nearest_obstacle_index = np.argmin(obstacle_distances)
        
        if min_distance < self.safe_radius:
            obstacle_vector = self.obstacle[:, nearest_obstacle_index] - self.current_coord
            tangent_vector = np.array([-obstacle_vector[1], obstacle_vector[0]])
            tangent_vector /= np.linalg.norm(tangent_vector)
            self.head_angle = atan2(tangent_vector[1], tangent_vector[0])
        else:
            self.head_angle = atan2(target_point[1] - self.current_coord[1],
                                    target_point[0] - self.current_coord[0])

        self.current_coord += self.linear_velocity * \
            np.array([cos(self.head_angle), sin(self.head_angle)]) * sample_time
        self.angular_velocity = (self.head_angle - previous_head_angle) / sample_time
        self.current_pose = np.array([*self.current_coord, self.head_angle, self.angular_velocity])

    def apf(self, sample_time, zeta, eta):
        obstacle_distances = np.linalg.norm(self.obstacle - self.current_coord.reshape(-1, 1), axis=0)
        previous_head_angle = self.head_angle
        q = self.current_coord
        theta = self.goal

        Q_star = self.detection_radius

        F_att = np.zeros(2)
        F_rep = np.zeros(2)

        d_goal = np.linalg.norm(theta - q)
        U_att = 0.5 * zeta * d_goal**2
        F_att = zeta * (q - theta)
        self.attraction_force = -F_att

        D = np.min(obstacle_distances)
        nearest_obstacle_index = np.argmin(obstacle_distances)

        if D <= Q_star:
            obstacle_vector = q - self.obstacle[:, nearest_obstacle_index]
            U_rep = 0.5 * eta * (1/D - 1/Q_star)**2
            F_rep = eta * (1/Q_star - 1/D) * (obstacle_vector / D)
            self.repulsion_force = -F_rep

        F = -F_att - F_rep

        F = F / np.linalg.norm(F)
        self.combined_force = F

        self.head_angle = atan2(F[1], F[0])

        self.current_coord += self.linear_velocity * np.array([cos(self.head_angle), sin(self.head_angle)]) * sample_time
        self.angular_velocity = (self.head_angle - previous_head_angle) / sample_time
        self.current_pose = np.array([self.current_coord[0], self.current_coord[1], self.head_angle, self.angular_velocity])

    def artificial_potential_field(self, sample_time, attraction_factor, repulsive_factor):
        previous_head_angle = self.head_angle
        obstacle_distances = np.linalg.norm(self.obstacle - self.current_coord.reshape(-1, 1), axis=0)
        target_point = self.goal

        min_distance = np.min(obstacle_distances)
        nearest_obstacle_index = np.argmin(obstacle_distances)

        obstacle_vector = self.current_coord - self.obstacle[:, nearest_obstacle_index]
        self.repulsion_force = obstacle_vector / np.linalg.norm(obstacle_vector)
        self.repulsion_force *= (self.detection_radius - min_distance) / self.detection_radius

        goal_vector = target_point - self.current_coord
        self.attraction_force = goal_vector / np.linalg.norm(goal_vector)

        self.combined_force = attraction_factor * self.attraction_force + repulsive_factor * self.repulsion_force
        self.combined_force /= np.linalg.norm(self.combined_force)

        self.head_angle = atan2(self.combined_force[1], self.combined_force[0])

        self.current_coord += self.linear_velocity * np.array([cos(self.head_angle), sin(self.head_angle)]) * sample_time
        self.angular_velocity = (self.head_angle - previous_head_angle) / sample_time
        self.current_pose = np.array([self.current_coord[0], self.current_coord[1], self.head_angle, self.angular_velocity])

    def at_goal(self):
        distance_to_goal = np.linalg.norm(self.current_pose[:2] - self.goal)
        return distance_to_goal < self.goal_radius

    @staticmethod
    def calc_dist(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))