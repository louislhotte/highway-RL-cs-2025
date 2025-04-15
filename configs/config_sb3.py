import pickle

config_dict = {
    "observation": {
        "type": "TimeToCollision"
    },
    "action": {
        "type": "DiscreteMetaAction"
    },
    "incoming_vehicle_destination": None,
    "duration": 11, # [s] If the environment runs for 11 seconds and still hasn't done(vehicle is crashed), it will be truncated. "Second" is expressed as the variable "time", equal to "the number of calls to the step method" / policy_frequency.
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px] width of the pygame window
    "screen_height": 600,  # [px] height of the pygame window
    "centering_position": [0.5, 0.6],  # The smaller the value, the more southeast the displayed area is. K key and M key can change centering_position[0].
    "scaling": 5.5,
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