# Config for continuous actions in highway-v0 env

config_dict = {
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "ContinuousAction",  # Continuous control (steering, acceleration)
        "steering_range": [-0.3, 0.3],
        "longitudinal": False,  # Enable acceleration/braking control
        "lateral": True,  # Enable steering control
    },

    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,  # [s]
    "initial_spacing": 2,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}

# config_dict = {
#     "observation": {
#         "type": "OccupancyGrid",
#         "vehicles_count": 10, # Increased from 10
#         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#         "features_range": {
#             "x": [-100, 100],
#             "y": [-100, 100],
#             "vx": [-20, 20],
#             "vy": [-20, 20],
#         },
#         "grid_size": [[-20, 20], [-20, 20]],
#         "grid_step": [5, 5],
#         "absolute": False,
#     },
#     "action": {
#         "type": "ContinuousAction",  # Continuous control (steering, acceleration)
#         "steering_range": [-0.1, 0.1],
#         "longitudinal": True,  # Enable acceleration/braking control
#         "lateral": True,  # Enable steering control
    
#     },
#     "lanes_count": 4,  # Increased from 4
#     "vehicles_count": 15, # Increased from 15
#     "duration": 60,  # [s]
#     "initial_spacing": 0,
#     "collision_reward": -1,  # The reward received when colliding with a vehicle.
#     "right_lane_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
#     # zero for other lanes.
#     "high_speed_reward": 0.1,  # Increased from 0.1 to try to encourage faster driving
#     "lane_change_reward": 0,
#     # "on_road_reward" : 100000,
#     "reward_speed_range": [
#         20,
#         30,
#     ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
#     "simulation_frequency": 5,  # [Hz]
#     "policy_frequency": 1,  # [Hz]
#     "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
#     "screen_width": 600,  # [px]
#     "screen_height": 150,  # [px]
#     "centering_position": [0.3, 0.5],
#     "scaling": 5.5,
#     "show_trajectories": True,
#     "render_agent": True,
#     "offscreen_rendering": False,
#     "disable_collision_checks": True,
# }

# env = gym.make("highway-fast-v0", render_mode="rgb_array")
# env.unwrapped.configure(config)
# print(env.reset())