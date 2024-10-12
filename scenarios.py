import numpy as np

def get_scenario(scenario_id: str, cycle_time_s: float):
    T = cycle_time_s
    if scenario_id == '1':
        # CV crossing
        scenario = {
            'duration_s': 20.0, 
            'x_init': np.array([-50, 20, 5, 0, 0, 0]),
            'state': 'cartesian'
        }
    elif scenario_id == '2':
        # CA crossing
        scenario = {
            'duration_s': 20.0, 
            'x_init': np.array([-70, 0, 10, 0, 0, 0]),
            'state': 'cartesian',
            'manx_frames': [[0, 0, 0]] # initial frame, n_frames, jerk value
        }
    elif scenario_id == '3':
        # Turn
        scenario = {
            'duration_s': 20.0, 
            'x_init': np.array([-70, 20, 5, 0, 0, 0]),
            'state': 'cartesian',
            'manx_frames': [[5/T, 20, 0.5], [7/T, 20, -0.5]] # initial frame, n_frames, jerk value
        }
    elif scenario_id == '4':
        # CTRV - single turn 
        scenario = {
            'duration_s': 20.0, 
            'x_init': np.array([-40, 10, 0, 5, 0]), # x, y, phi, v, w
            'state': 'polar', 
            'manx_frames': [[5/T, 10/T, np.radians(-90/5)]], # initial frame, n_frames, manouver value
        }
    elif scenario_id == '5':
        # CTRV - smooth turn and sharp turn 
        scenario = {
            'duration_s': 20.0, 
            'x_init': np.array([-40, 10, 0, 5, 0]), # x, y, phi, v, w
            'state': 'polar', 
            # 'manx_frames': [[5/T, 10/T, np.radians(-90/7)], [12/T, 15/T, np.radians(-90/3)]], # initial frame, n_frames, manouver value
            'accw_frames': [[5/T, 5/T+15, np.radians(-1)], [10/T, 10/T+15, np.radians(1)]], # initial frame, n_frames, increment value
        }
    else:
        raise ValueError('Scenario not found')
    scenario['n_frames'] = int(scenario['duration_s']/T)
    return scenario
