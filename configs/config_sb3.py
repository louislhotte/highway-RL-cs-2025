import pickle

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "features": ['presence', 'on_road'],
        "grid_size": [[-18, 18], [-18, 18]],
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True
    },
    "action": {
        "type": "ContinuousAction",
        "longitudinal": False,
        "lateral": True
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 1,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "scaling": 7,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": False
}

# config_dict = {
#     "observation": {
#         "type": "OccupancyGrid",
#         "vehicles_count": 20,
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
#         "type": "DiscreteMetaAction",
#     },
#     "lanes_count": 8,
#     "vehicles_count": 30,
#     "duration": 60,  # [s]
#     "initial_spacing": 0,
#     "collision_reward": -1,  # The reward received when colliding with a vehicle.
#     "right_lane_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
#     # zero for other lanes.
#     "high_speed_reward": 0.1,  # The reward received when driving at full speed, linearly mapped to zero for
#     # lower speeds according to config["reward_speed_range"].
#     "lane_change_reward": 0,
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

# # with open("config.pkl", "wb") as f:
# #     pickle.dump(config_dict, f)

# # env = gym.make("highway-fast-v0", render_mode="rgb_array")
# # env.unwrapped.configure(config)
# # print(env.reset())