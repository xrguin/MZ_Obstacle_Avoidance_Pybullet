import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import clear_output
import pandas as pd
from tqdm import tqdm

from mbrobot import MbRobot
from utils import generate_random_coords, generate_intercepting_path, plot_robot, plot_robot_artist

class RobotSimulation:
    def __init__(self, uncertainty_params_robot2=None, uncertainty_params_robot3=None):
        # Basic parameters
        self.r_a = 4
        self.r_d = 0
        self.distance = 8
        self.detection_radius = 15
        self.test_site_size = 15
        self.samples = 1
        self.sample_time = 0.1
        self.simulation_time = 30

        # Default uncertainty parameters
        self.default_uncertainty = {
            'var_observation': 0.05,
            'attraction_factor': 1.0,
            'var_attraction': 0.01,
            'repulsive_factor': 1.0,
            'var_repulsive': 0.01
        }
        
        # Store custom uncertainty parameters for each robot
        self.uncertainty_params_robot2 = uncertainty_params_robot2 if uncertainty_params_robot2 is not None else []
        self.uncertainty_params_robot3 = uncertainty_params_robot3 if uncertainty_params_robot3 is not None else []

        # Storage
        self.robot2 = None
        self.robot3 = None
        self.poses2 = []
        self.poses3 = []
        self.all_poses2 = np.empty((self.samples,), dtype=object)
        self.all_poses3 = np.empty((self.samples,), dtype=object)
        self.all_frames = []

        np.random.seed(42)

    def setup_robots(self):
        """Initialize robot configurations"""
        # Generate ally robot (robot2)
        self.robot2 = MbRobot()
        self.robot2.start = np.array([1.0, 1.0])
        self.robot2.goal = np.array([14.0, 14.0])
        self.robot2.safe_radius = self.r_a
        self.robot2.light_color = [0.8, 1, 0.8, 0.2]
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
        self.robot3.start = np.array([14.0, 1.0])
        self.robot3.goal = np.array([1.0, 14.0])
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

    def get_uncertainty_params(self, sample_index, robot_type):
        """Get uncertainty parameters for a specific sample index and robot"""
        if robot_type == 'robot2':
            params_list = self.uncertainty_params_robot2
        else:  # robot3
            params_list = self.uncertainty_params_robot3
            
        if sample_index < len(params_list):
            return params_list[sample_index]
        return self.default_uncertainty

    def simulate_step(self, current_sample):
        """Simulate one step of robot movement"""
        if self.robot2.at_goal() or self.robot3.at_goal():
            return False

        # Get uncertainty parameters for each robot
        params_robot2 = self.get_uncertainty_params(current_sample, 'robot2')
        params_robot3 = self.get_uncertainty_params(current_sample, 'robot3')

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
            }
        }
        self.all_frames.append(frame_data)
        return True

    def update_matplotlib(self, frame):
        """Update function for animation"""
        ax = self.ax
        ax.clear()
        
        frame_data = self.all_frames[frame]

        # Plot trajectories
        if len(frame_data['poses2']) > 0:
            ax.plot(frame_data['poses2'][:, 0], frame_data['poses2'][:, 1], '-', 
                   color='green', label='Ally Robot')
            ax.plot(frame_data['poses3'][:, 0], frame_data['poses3'][:, 1], '-', 
                   color='blue', label='Adversarial Robot')
        
        # Plot current positions
        ax.scatter(frame_data['robot2'][0], frame_data['robot2'][1], 
                  c='green', s=100, label='Ally')
        ax.scatter(frame_data['robot3'][0], frame_data['robot3'][1], 
                  c='blue', s=100, label='Adversary')
        
        # Plot start and goal positions
        for robot in ['robot2', 'robot3']:
            ax.plot(frame_data['starts'][robot][0], 
                   frame_data['starts'][robot][1], 'ko', markersize=10)
            ax.plot(frame_data['goals'][robot][0], 
                   frame_data['goals'][robot][1], 'k*', markersize=10)
        
        # Add parameter information to title
        params2 = frame_data['params']['robot2']
        params3 = frame_data['params']['robot3']
        title = f'Time: {frame_data["time"]:.1f}s\n'
        title += f'Ally: obs={params2["var_observation"]:.3f}, att={params2["attraction_factor"]:.1f}, rep={params2["repulsive_factor"]:.1f}\n'
        title += f'Adv: obs={params3["var_observation"]:.3f}, att={params3["attraction_factor"]:.1f}, rep={params3["repulsive_factor"]:.1f}'
        
        ax.set_xlim([-5, 20])
        ax.set_ylim([-5, 20])
        ax.grid(True)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(title)

    def run_matplotlib_simulation(self, filename=None):
        """Run simulation with Matplotlib animation"""
        plt.ioff()  # Turn off interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Simulate and collect all frames first
        while self.simulate_step(self.current_sample):
            pass
            
        # Create animation with exact number of frames
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

    def run_simulation(self):
        """Main simulation loop"""
        for i in tqdm(range(self.samples), desc="Running simulations"):
            self.current_sample = i
            self.setup_robots()
            self.poses2 = []
            self.poses3 = []
            self.all_frames = []
            
            # Run simulation
            self.run_matplotlib_simulation(filename=f'simulation_{i}')
            
            # Store results
            self.all_poses2[i] = np.array(self.poses2)
            self.all_poses3[i] = np.array(self.poses3)

def main():
    # Example usage with different parameters for each robot
    uncertainty_params_robot2 = [
        # First sample with no uncertainty
        {
            'var_observation': 0.0,
            'attraction_factor': 1.0,
            'var_attraction': 0.0,
            'repulsive_factor': 1.0,
            'var_repulsive': 0.0
        },
        # Second sample with some uncertainty
        {
            'var_observation': 0.05,
            'attraction_factor': 1.0,
            'var_attraction': 0.01,
            'repulsive_factor': 1.0,
            'var_repulsive': 0.01
        }
    ]
    
    uncertainty_params_robot3 = [
        # First sample with moderate uncertainty
        {
            'var_observation': 0.05,
            'attraction_factor': 1.0,
            'var_attraction': 0.01,
            'repulsive_factor': 1.2,
            'var_repulsive': 0.01
        },
        # Second sample with high uncertainty
        {
            'var_observation': 0.1,
            'attraction_factor': 1.0,
            'var_attraction': 0.02,
            'repulsive_factor': 1.5,
            'var_repulsive': 0.02
        }
    ]

    sim = RobotSimulation(
        uncertainty_params_robot2=uncertainty_params_robot2,
        uncertainty_params_robot3=uncertainty_params_robot3
    )
    sim.samples = len(uncertainty_params_robot2)  # Make sure both parameter lists have same length
    sim.run_simulation()

if __name__ == "__main__":
    main()