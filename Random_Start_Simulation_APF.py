#%%
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, clear_output
import pandas as pd
from tqdm import tqdm

from mbrobot import MbRobot
from utils import generate_random_coords, generate_intercepting_path, plot_robot
import h5py

#%% Initialization
class RobotSimulation:
    def __init__(self, 
                 uncertainties_robot2 = None,
                 uncertainties_robot3 = None,
                 start_pos_robot2 = None, 
                 goal_pos_robot2 = None,
                 start_pos_robot3 = None,
                 goal_pos_robot3 = None):
        
        # Basic parameters
        self.r_a = 4
        self.r_d = 0
        self.detection_radius = 15
        self.test_site_size = 15
        self.sample_time = 0.1
        self.simulation_time = 30

        self.start_2 = start_pos_robot2
        self.goal_2 = goal_pos_robot2
        self.start_3 = start_pos_robot3
        self.goal_3 = goal_pos_robot3

        # Default unvertainty parameters
        # Default uncertainty parameters
        self.default_uncertainty = {
            'var_observation': 0.05,
            'attraction_factor': 1.0,
            'var_attraction': 0.01,
            'repulsive_factor': 1.0,
            'var_repulsive': 0.01
        }

        # Store custom uncertainty parameters for each robot
        self.uncertainty_params_robot2 = uncertainties_robot2 if uncertainties_robot2 is not None else self.default_uncertainty
        self.uncertainty_params_robot3 = uncertainties_robot3 if uncertainties_robot3 is not None else self.default_uncertainty

        # Simulation mode flags
        self.visualization_mode = 'matplotlib'  # 'matplotlib' or 'pybullet'

        # Storage
        self.robot2 = None
        self.robot3 = None
        self.poses2 = []
        self.poses3 = []


        # Animation storage
        self.all_frames = []
        
        # PyBullet attributes
        self.physicsClient = None
        self.robot2_id = None
        self.robot3_id = None

    #%% Setups for visualization using pybullet
    def setup_pybullet(self):
        """Initialize PyBullet simulation"""
        import pybullet as p
        import pybullet_data

        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        
        # Set up camera
        p.resetDebugVisualizerCamera(
            cameraDistance=15,
            cameraYaw=0,
            cameraPitch=-70,
            cameraTargetPosition=[7.5, 7.5, 0]
        )

        # Load robots
        robot2_orientation = p.getQuaternionFromEuler([0, 0, self.robot2.head_angle])
        robot3_orientation = p.getQuaternionFromEuler([0, 0, self.robot3.head_angle])
        
        self.robot2_id = p.loadURDF('turtlebot.urdf', 
                                   [*self.robot2.current_coord, 0], 
                                   robot2_orientation)
        self.robot3_id = p.loadURDF('turtlebot.urdf', 
                                   [*self.robot3.current_coord, 0], 
                                   robot3_orientation)
        
        return p

    #%% Setups for visualization using matplotlib
    def setup_matplotlib(self):
        """Initialize Matplotlib visualization"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        # Create empty line objects for trajectories
        self.line2, = self.ax.plot([], [], '-', color=[0, 1, 0])
        self.line3, = self.ax.plot([], [], '-', color=[0, 0, 1])
        
        # Plot static elements (start and goal positions)
        self.ax.plot(self.robot2.start[0], self.robot2.start[1], 'ko', markersize=10, markerfacecolor='g')
        self.ax.plot(self.robot2.goal[0], self.robot2.goal[1], 'kp', markersize=10, markerfacecolor='g')
        self.ax.plot(self.robot3.start[0], self.robot3.start[1], 'ko', markersize=10, markerfacecolor='b')
        self.ax.plot(self.robot3.goal[0], self.robot3.goal[1], 'kp', markersize=10, markerfacecolor='b')
        
        self.ax.set_xlim([-5, 20])
        self.ax.set_ylim([-5, 20])
        self.ax.set_aspect('equal', 'box')
        return self.fig, self.ax
    
    #%% Setups for robots
    def setup_robots(self):
        # Generate ally robot (robot2)
        self.robot2 = MbRobot()
        self.robot2.start = self.start_2
        self.robot2.goal = self.goal_2
        self.robot2.safe_radius = self.r_a
        self.robot2.light_color = [0.6, 1, 0.6, 0.2]
        self.robot2.dark_color = [0.5, 1, 0.5, 0.2]
        self.robot2.head_angle = np.arctan2(
            self.robot2.goal[1] - self.robot2.start[1], 
            self.robot2.goal[0] - self.robot2.start[0]
        )
        self.robot2.current_pose = np.array([
            *self.robot2.start, self.robot2.head_angle, self.robot2.angular_velocity
        ])
        self.robot2.current_coord = self.robot2.current_pose[:2]
        self.robot2.detection_radius = self.detection_radius

        # Generate enemy robot (robot3)
        self.robot3 = MbRobot()
        self.robot3.start = self.start_3
        self.robot3.goal = self.goal_3
        self.robot3.safe_radius = self.r_a
        self.robot3.head_angle = np.arctan2(
            self.robot3.goal[1] - self.robot3.start[1], 
            self.robot3.goal[0] - self.robot3.start[0]
        )
        self.robot3.current_pose = np.array([
            *self.robot3.start, self.robot3.head_angle, self.robot3.angular_velocity
        ])
        self.robot3.current_coord = self.robot3.current_pose[:2]
        self.robot3.detection_radius = self.detection_radius
        self.robot3.light_color = [0.6, 0.8, 1, 0.2]
        self.robot3.dark_color = [0.4, 0.6, 1, 0.2]

    #%% Get the user defined uncertainties 
    def get_uncertainty_params(self, robot_type):
        """Get uncertainty parameters for a specific sample index and robot"""
        if robot_type == 'robot2':
            params = self.uncertainty_params_robot2
        else:  # robot3
            params = self.uncertainty_params_robot3
            
        return params

    #%% One simulation step 
    def simulate_step(self):
        """Simulate one step of robot movement"""
        if self.robot2.at_goal() or self.robot3.at_goal():
            return False
        
        # Get uncertainty parameters for each robot
        params_robot2 = self.get_uncertainty_params('robot2')
        params_robot3 = self.get_uncertainty_params('robot3')

        # Update robot2 (ally robot)
        self.robot2.obstacle = self.robot3.current_pose[:2].reshape(2, 1)
        self.robot2.artificial_potential_field_uncertainty(
            sample_time=self.sample_time,
            var_observation=params_robot2['var_observation'],
            attraction_factor=params_robot2['attraction_factor'],
            var_attraction=params_robot2['var_attraction'],
            repulsive_factor=params_robot2['repulsive_factor'],
            var_repulsive=params_robot2['var_repulsive']
        )

        # Update robot3 (enemy robot)
        self.robot3.obstacle = self.robot2.current_pose[:2].reshape(2, 1)
        self.robot3.artificial_potential_field_uncertainty(
            sample_time=self.sample_time,
            var_observation=params_robot3['var_observation'],
            attraction_factor=params_robot3['attraction_factor'],
            var_attraction=params_robot3['var_attraction'],
            repulsive_factor=params_robot3['repulsive_factor'],
            var_repulsive=params_robot3['var_repulsive']
        )

        self.poses2.append(self.robot2.current_pose.copy())
        self.poses3.append(self.robot3.current_pose.copy())

        # Store frame data
        frame_data = {
            'robot2': self.robot2.current_coord.copy(),
            'robot3': self.robot3.current_coord.copy(),
            'poses2': np.array(self.poses2).copy(),
            'poses3': np.array(self.poses3).copy(),
            'time': len(self.poses2) * self.sample_time,
            'goals': {
                'robot2': self.robot2.goal.copy(),
                'robot3': self.robot3.goal.copy()
            },
            'starts': {
                'robot2': self.robot2.start.copy(),
                'robot3': self.robot3.start.copy()
            },
            'params': {
                'robot2': params_robot2.copy(),
                'robot3': params_robot3.copy()
            },
            'forces': {
                'robot2': {
                    'attraction': self.robot2.attraction_force.copy(),
                    'repulsion': self.robot2.repulsion_force.copy(),
                    'combined': self.robot2.combined_force.copy()
                },
                'robot3': {
                    'attraction': self.robot3.attraction_force.copy(),
                    'repulsion': self.robot3.repulsion_force.copy(),
                    'combined': self.robot3.combined_force.copy()
                }
            }
        }
        self.all_frames.append(frame_data)

        return True
    
    #%% Visualize the updated one step simulation using pybullet
    def update_pybullet(self, p):
        """Update PyBullet visualization"""
        p.resetBasePositionAndOrientation(
            self.robot2_id,
            [*self.robot2.current_coord, 0],
            p.getQuaternionFromEuler([0, 0, self.robot2.head_angle])
        )
        p.resetBasePositionAndOrientation(
            self.robot3_id,
            [*self.robot3.current_coord, 0],
            p.getQuaternionFromEuler([0, 0, self.robot3.head_angle])
        )
        p.stepSimulation()

    #%% Visualize the updated one step simulation using matplotlib
    def update_matplotlib(self, frame):
        """Update function for Matplotlib animation"""
        ax = self.ax
        ax.clear()
        
        frame_data = self.all_frames[frame]

        # Plot trajectories
        if len(frame_data['poses2']) > 0:
            ax.plot(frame_data['poses2'][:, 0], frame_data['poses2'][:, 1], '-', 
                   color='green', label='Ally Trajectory')
            ax.plot(frame_data['poses3'][:, 0], frame_data['poses3'][:, 1], '-', 
                   color='blue', label='Adversarial Trajectory')
            
        # Update robot states including forces
        self.robot2.current_coord = frame_data['robot2']
        self.robot2.current_pose = frame_data['poses2'][-1]
        self.robot2.head_angle = frame_data['poses2'][-1][2]
        self.robot2.attraction_force = frame_data['forces']['robot2']['attraction']
        self.robot2.repulsion_force = frame_data['forces']['robot2']['repulsion']
        self.robot2.combined_force = frame_data['forces']['robot2']['combined']
        
        self.robot3.current_coord = frame_data['robot3']
        self.robot3.current_pose = frame_data['poses3'][-1]
        self.robot3.head_angle = frame_data['poses3'][-1][2]
        self.robot3.attraction_force = frame_data['forces']['robot3']['attraction']
        self.robot3.repulsion_force = frame_data['forces']['robot3']['repulsion']
        self.robot3.combined_force = frame_data['forces']['robot3']['combined']
        
        # Plot robots using plot_robot
        plot_robot(ax, self.robot2, frame_size=1)
        plot_robot(ax, self.robot3, frame_size=1)


        ax.plot(frame_data['starts']['robot2'][0], 
                   frame_data['starts']['robot2'][1], 'ko', markersize=10, markerfacecolor='g')
        ax.plot(frame_data['goals']['robot2'][0], 
                   frame_data['goals']['robot2'][1], 'kp', markersize=10, markerfacecolor='g')
        ax.plot(frame_data['starts']['robot3'][0], 
                   frame_data['starts']['robot3'][1], 'ko', markersize=10, markerfacecolor='b')
        ax.plot(frame_data['goals']['robot3'][0], 
                   frame_data['goals']['robot3'][1], 'kp', markersize=10, markerfacecolor='b')

         # Add parameter information to title
        params2 = frame_data['params']['robot2']
        params3 = frame_data['params']['robot3']
        title = (
                f'Time: {frame_data["time"]:.1f}s\n'
                f'Ally: std obs={params2["var_observation"]:.3f}, '
                f'std weights={params2["var_attraction"]:.3f}, '
                f'att={params2["attraction_factor"]:.1f}, '
                f'rep={params2["repulsive_factor"]:.1f}\n'
                f'Adv: std obs={params3["var_observation"]:.3f}, '
                f'std weights={params3["var_attraction"]:.3f}, '
                f'att={params3["attraction_factor"]:.1f}, '
                f'rep={params3["repulsive_factor"]:.1f}'
        )
        
        ax.set_xlim([-5, 20])
        ax.set_ylim([-5, 20])
        ax.grid(True)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(title)
    
    #%% Run the pybullet simulation
    def run_pybullet_simulation(self):
        """Run simulation with PyBullet visualization"""
        p = self.setup_pybullet()

        self.setup_robots()
        
        for t in tqdm(np.arange(0, self.simulation_time, self.sample_time)):
            if not self.simulate_step():
                break
                
            self.update_pybullet(p)
            time.sleep(self.sample_time)  # Maintain real-time simulation
            
        p.disconnect()

        return np.array(self.poses2), np.array(self.poses3)
    
    #%% Run the simulation without any visualization
    def run_simulation_no_animation(self):
        self.setup_robots()

        for t in np.arange(0, self.simulation_time, self.sample_time):
            if not self.simulate_step():
                break
        
        return np.array(self.poses2), np.array(self.poses3)


    #%% Run the matplotlib simulation
    def run_matplotlib_simulation(self, filename = None):
        """Run simulation with Matplotlib animation"""
        plt.ioff()  # Turn off interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.setup_robots()

        # Simulate and collect all frames first
        while self.simulate_step():
            pass
        
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_matplotlib,
            frames=len(self.all_frames),
            interval=self.sample_time * 1000,
            blit=False,
            repeat=False
        )


        if filename is not None:
            writer = animation.FFMpegWriter(
                fps=30, 
                metadata=dict(artist='Me'),
                bitrate=2000
            )
            self.animation.save(
                f'{filename}.mp4', 
                writer=writer,
                dpi=300,
                savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0}
            )
        
        plt.close()

        return np.array(self.poses2), np.array(self.poses3)
    

    #%% Save start and goal positions for ally and adversarial robots
    def return_start(self):
        return self.robot2.start, self.robot3.start
    
    def return_goal(self):
        return self.robot2.goal, self.robot3.goal
    

#%% Set uncertainties, number of running samples in the main function
def main(seed_number):

    np.random.seed(seed_number)
    
    params_no_error_robot2 = {
        'var_observation': 0.0,
        'attraction_factor': 1.0,
        'var_attraction': 0.0,
        'repulsive_factor': 1.0,
        'var_repulsive': 0.0
    }

    params_no_error_robot3 = {
        'var_observation': 0.0,
        'attraction_factor': 1.0,
        'var_attraction': 0.0,
        'repulsive_factor': 1.2,
        'var_repulsive': 0.0
    }
    uncertainty_params_robot2 = {
        'var_observation': 0.05,
        'attraction_factor': 1.0,
        'var_attraction': 0.02,
        'repulsive_factor': 1.0,
        'var_repulsive': 0.02
    }

    uncertainty_params_robot3 = {
        'var_observation': 0.05,
        'attraction_factor': 1.0,
        'var_attraction': 0.02,
        'repulsive_factor': 1.2,
        'var_repulsive': 0.02
    }

    samples = 500
    all_poses2 = np.empty((samples), dtype=object)
    all_poses3 = np.empty((samples), dtype=object)

    test_site_size = 15
    r_a = 4

    start_pos_2 = np.random.rand(2) * test_site_size
    goal_pos_2 = generate_random_coords(num_coords=1, 
                                        origins=start_pos_2.reshape(1,-1),
                                        radius = 8,
                                        test_site_size=15 )[0]

    start_pos_3, goal_pos_3, _ = generate_intercepting_path(ref_start = start_pos_2,
                                                            ref_goal = goal_pos_2,
                                                            test_site_size = 15,
                                                            safe_radius = r_a
        )


    for i in tqdm(range(samples), desc="Running simulations"):
        # if i == 0:
        #     sim = RobotSimulation(
        #         uncertainties_robot2 = params_no_error_robot2,
        #         uncertainties_robot3 = params_no_error_robot3,
        #         start_pos_robot2 = start_pos_2,
        #         goal_pos_robot2 = goal_pos_2,
        #         start_pos_robot3 = start_pos_3,
        #         goal_pos_robot3 = goal_pos_3
        #     )
            
        #     # poses2, poses3 = sim.run_matplotlib_simulation(filename = f'trail{seed_number},sample{i}')
        #     poses2, poses3 = sim.run_simulation_no_animation()
        
        # else:
        sim = RobotSimulation(
            uncertainties_robot2 = uncertainty_params_robot2,
            uncertainties_robot3 = uncertainty_params_robot3,
            start_pos_robot2 = start_pos_2,
            goal_pos_robot2 = goal_pos_2,
            start_pos_robot3 = start_pos_3,
            goal_pos_robot3 = goal_pos_3
        )
        # poses2, poses3 = sim.run_matplotlib_simulation(filename = f'trail{seed_number},sample{i}')
        poses2, poses3 = sim.run_simulation_no_animation()
        robot2_start, robot3_start = sim.return_start()
        robot2_goal, robot3_goal = sim.return_goal()


        all_poses2[i] = poses2
        all_poses3[i] = poses3


    def calculate_means_var_std(lists, element):

        max_length = max(len(lst) for lst in lists)
        arr = np.ma.empty((len(lists), max_length))
        arr.mask = True

        for i, lst in enumerate(lists):
            for j in range(len(lst)):
                arr[i, j] = lst[j][element]
                arr.mask[i,j] = False

        mean_values = np.mean(arr, axis = 0)
        var_values = np.var(arr, axis = 0)
        std_values = np.std(arr, axis = 0)

        return mean_values, std_values
    
    mean_x_2, std_x_2 = calculate_means_var_std(all_poses2, 0)
    mean_y_2, std_y_2 = calculate_means_var_std(all_poses2, 1)
    mean_u_2, std_u_2 = calculate_means_var_std(all_poses2, 2)

    mean_x_3, std_x_3 = calculate_means_var_std(all_poses3, 0)
    mean_y_3, std_y_3 = calculate_means_var_std(all_poses3, 1)
    mean_u_3, std_u_3 = calculate_means_var_std(all_poses3, 2)

    mean_poses_2 = np.array(list(zip(mean_x_2, mean_y_2, mean_u_2)))
    mean_poses_3 = np.array(list(zip(mean_x_3, mean_y_3, mean_u_3)))

    std_poses_2 = np.array(list(zip(std_x_2, std_y_2, std_u_2)))
    std_poses_3 = np.array(list(zip(std_x_3, std_y_3, std_u_3)))

    robot2_data = {
        'trajectory_mean': mean_poses_2,
        'trajectory_std': std_poses_2,
        'start': robot2_start,
        'goal': robot2_goal
    }

    robot3_data = {
        'trajectory_mean': mean_poses_3,
        'trajectory_std': std_poses_3,
        'start': robot3_start,
        'goal': robot3_goal
    }


       

    return robot2_data, robot3_data

    
#%%
if __name__ == "__main__":

    import joblib

    trials = 5000

    robot2_mean_all = np.empty((trials,), dtype=object)
    robot3_mean_all = np.empty((trials,), dtype=object)

    robot2_std_all = np.empty((trials,), dtype=object)
    robot3_std_all = np.empty((trials,), dtype=object)

    record_data = True

    for n in tqdm(range(trials), desc="Trials completed: "):

        robot2_data, robot3_data = main(n)
        mean_poses_2 = robot2_data['trajectory_mean']
        mean_poses_3 = robot3_data['trajectory_mean']

        std_poses_2 = robot2_data['trajectory_std']
        std_poses_3 = robot3_data['trajectory_std']

        robot2_mean_all[n] = mean_poses_2
        robot3_mean_all[n] = mean_poses_3

        robot2_std_all[n] = std_poses_2
        robot3_std_all[n] = std_poses_3
        

    if record_data:
        # Create a dictionary to hold your data
        data = {
            "robot2": {
                "mean_trajectories": robot2_mean_all,
                "std_trajectories": robot2_std_all
            },
            "robot3": {
                "mean_trajectories": robot3_mean_all,
                "std_trajectories": robot3_std_all
            }
        }

        # Save the data to a file using joblib
        file_name = 'uncertainty_robot_trajectories_5000.pkl'
        joblib.dump(data, file_name)
            

        
        print(f"Data collection complete. Files saved: {file_name}")

# %%
