# %%
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from tqdm import tqdm

from mbrobot import MbRobot
from IPython.display import HTML, clear_output
from utils import generate_random_coords, generate_intercepting_path, plot_robot


# %%
np.random.seed(42)

simulation = True
sim_on_pybullet = False
sim_on_plt = True
record_data = False

r_a = 4
r_d = 0
distance = 8
detection_radius = 15
test_site_size = 15

samples = 5
# Initialize cell-like structures
all_poses2 = np.empty((samples,), dtype=object)
all_poses3 = np.empty((samples,), dtype=object)

# %%
if sim_on_pybullet:
    import pybullet as p
    import pybullet_data

    physicsClient = p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    # Set up camera
    p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-70, cameraTargetPosition=[7.5,7.5,0])


# %%
if simulation:
        for i in tqdm(range(samples), desc="Running simulations"):
            # Generate ally robot (robot2)
            robot2 = MbRobot()
            robot2.start = np.random.rand(2) * test_site_size
            # robot2.start = np.array([1.0, 1.0])
            robot2.goal = generate_random_coords(1, robot2.start.reshape(1, -1), distance, test_site_size)[0]
            # robot2.goal = np.array([14.0, 14.0])
            robot2.safe_radius = r_a
            robot2.light_color = [0.8, 1, 0.8, 0.2]
            robot2.dark_color = [0.5, 1, 0.5, 0.2]
            robot2.head_angle = np.arctan2(robot2.goal[1] - robot2.start[1], robot2.goal[0] - robot2.start[0])
            robot2.current_pose = np.array([*robot2.start, robot2.head_angle, robot2.angular_velocity])
            robot2.current_coord = robot2.current_pose[:2]
            robot2.detection_radius = detection_radius
            
            
            # Generate enemy robot (robot3)
            robot3 = MbRobot()
            robot3.start, robot3.goal, intercept_point = generate_intercepting_path(robot2.start, robot2.goal, test_site_size, r_a)

            # robot3.start, robot3.goal  = np.array([14.0, 1.0]), np.array([1.0, 14.0])
            
            robot3.safe_radius = r_a
            robot3.head_angle = np.arctan2(robot3.goal[1] - robot3.start[1], robot3.goal[0] - robot3.start[0])
            robot3.current_pose = np.array([*robot3.start, robot3.head_angle, robot3.angular_velocity])
            robot3.current_coord = robot3.current_pose[:2]
            robot3.detection_radius = detection_radius
            robot3.light_color = [0.6, 0.8, 1, 0.2]
            robot3.dark_color = [0.4, 0.6, 1, 0.2]
            

            if sim_on_pybullet:
                robot2_Orientation = p.getQuaternionFromEuler([0, 0, robot2.head_angle])
                robot3_Orientation = p.getQuaternionFromEuler([0, 0, robot3.head_angle])
                robot2_id = p.loadURDF('turtlebot.urdf', [*robot2.current_coord,0], robot2_Orientation)
                robot3_id = p.loadURDF('turtlebot.urdf', [*robot3.current_coord, 0], robot3_Orientation)

            if sim_on_plt:
                plt.ion()
                fig, ax = plt.subplots(figsize=(10, 10))

            poses2 = []
            poses3 = []

            sample_time = 0.1
            simulation_time = 30

            for t in np.arange(0, simulation_time, sample_time):
                robot2_pose = robot2.current_pose
                robot3_pose = robot3.current_pose
                
                # Update robot2 (ally robot)
                robot2.obstacle = robot3_pose[:2].reshape(2, 1)
                robot2.artificial_potential_field_uncertainty(sample_time = sample_time, 
                                                              var_observation = 0.05, 
                                                              attraction_factor = 1, 
                                                              var_attraction = 0.01, 
                                                              repulsive_factor = 1, 
                                                              var_repulsive = 0.01)

                # Update robot3 (enemy robot)
                robot3.obstacle = robot2_pose[:2].reshape(2, 1)
                robot3.artificial_potential_field_uncertainty(sample_time = sample_time, 
                                                              var_observation = 0.05, 
                                                              attraction_factor = 1, 
                                                              var_attraction = 0.01, 
                                                              repulsive_factor = 1.2, 
                                                              var_repulsive = 0.01)
                
                
                    
                if sim_on_plt:
                    # fig, ax = plt.subplots(figsize = (10,10))

                    ax.clear()
                    plot_robot(ax, robot2, 1)
                    plot_robot(ax, robot3, 1)
                    ax.plot([p[0] for p in poses2], [p[1] for p in poses2], '-', color=[0, 1, 0])
                    ax.plot([p[0] for p in poses3], [p[1] for p in poses3], '-', color=[0, 0, 1])

                    ax.plot(robot2.start[0], robot2.start[1], 'ko', markersize=10, markerfacecolor='g')
                    ax.plot(robot2.goal[0], robot2.goal[1], 'kp', markersize=10, markerfacecolor='g')
                    ax.plot(robot3.start[0], robot3.start[1], 'ko', markersize=10, markerfacecolor='b')
                    ax.plot(robot3.goal[0], robot3.goal[1], 'kp', markersize=10, markerfacecolor='b')
                    ax.set_xlim([-5, 20])
                    ax.set_ylim([-5, 20])
                    ax.set_aspect('equal', 'box')
                    plt.draw()
                    plt.pause(0.01)
                    
                
                if robot2.at_goal() or robot3.at_goal():
                    break
                
                poses2.append(robot2.current_pose)
                poses3.append(robot3.current_pose)
                
                # Update PyBullet robot positions
                if sim_on_pybullet:
                    p.resetBasePositionAndOrientation(robot2_id, [*robot2.current_coord, 0], 
                                                    p.getQuaternionFromEuler([0, 0, robot2.head_angle]))
                    p.resetBasePositionAndOrientation(robot3_id, [*robot3.current_coord, 0], 
                                                    p.getQuaternionFromEuler([0, 0, robot3.head_angle]))
                    p.stepSimulation()
                    time.sleep(sample_time)


                
            
            all_poses2[i] = np.array(poses2)
            all_poses3[i] = np.array(poses3)
            
            

# %%

if record_data:
    # Save data in a cell-like structure
    np.save('robot2_trajectories.npy', all_poses2)
    np.save('robot3_trajectories.npy', all_poses3)
    
    print("Data collection complete. Files saved: robot2_trajectories.npy, robot3_trajectories.npy")


if sim_on_pybullet:
    p.disconnect()


