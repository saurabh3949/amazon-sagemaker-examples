def reward_function(params):

    distance_from_center = params['distance_from_center']
    track_width = params['track_width']

    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    reward = 1e-3
    if distance_from_center <= marker_1:
        reward = 2
    elif distance_from_center <= marker_2:
        reward = 1
    elif distance_from_center <= marker_3:
        reward = 0.5
    else:
        reward = 1e-3  # likely crashed/ close to off track

#     if params['speed'] < 2.0:
#         if reward > 0:
#             reward *= 0.5
#     if params['steering_angle'] > 15:
#         if reward > 0:
#             reward *= 0.5
        
    return float(reward)