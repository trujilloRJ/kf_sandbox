import numpy as np

def get_scenario(scenario_id: str, cycle_time_s: float):
    T = cycle_time_s
    if scenario_id == '1':
        # CV crossing
        scenario = {
            'duration_s': 20.0, 
            'x_init': np.array([-50, 20, 5, 0, 0, 0]),
        }
    elif scenario_id == '2':
        # CA crossing
        scenario = {
            'duration_s': 20.0, 
            'x_init': np.array([-70, 20, 5, 0, 0, 0]),
            'ax_frames': [[5/T, 7/T, 3], [13/T, 15/T, -3]],
        }
    else:
        raise ValueError('Scenario not found')
    scenario['n_frames'] = int(scenario['duration_s']/T)
    return scenario
