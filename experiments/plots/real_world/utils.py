import numpy as np
def mu_to_v(mapping, svar_process, mask, names, intervention_effects):
    """
    Convert intervention effects to the force vector.
    """
    force_vector = np.zeros(len(names)).reshape(-1, 1)
    for col, value in intervention_effects.items():
        interventional_vector = np.zeros(len(names)).reshape(-1, 1)
        index = mapping[col]
        interventional_vector[index] = value
        force_vector += (svar_process.A_solve @ svar_process._char_mat @ interventional_vector) * mask
    return force_vector

def varprocess_additive_intervention(process, intervention_effects, names, intervention_steps, asymptotic=False):
    """
    Apply additive intervention effects to the VAR process.
    """
    mask = np.zeros(len(names)).reshape(-1, 1)
    mapping = {names[i]: i for i in range(len(names))}

    for col, value in intervention_effects.items():
        index = mapping[col]
        mask[index] = 1

    interventional_vector = np.zeros(len(names)).reshape(-1, 1)
    if asymptotic:
        interventional_vector = mu_to_v(mapping, process, mask, names, intervention_effects)
    else:
        for col, value in intervention_effects.items():
            index = mapping[col]
            interventional_vector[index] = value

    size = process.coefs.shape[1]
    interventional_vector = np.array(interventional_vector).reshape(-1, 1)
    effects_vector = np.zeros(shape=(0, size))
    effects = np.zeros(size).reshape(-1, 1)
    response_matrices = process.ma_rep(maxn=intervention_steps)
    for i in range(intervention_steps):
        if i <= np.inf:
            effects += response_matrices[i] @ interventional_vector
        else:
            effects = np.sum(response_matrices[i - np.inf:i + 1], axis=0) @ interventional_vector
        effects_vector = np.vstack([effects_vector, effects.T])
    return effects_vector

def scale_data_by_pop(data):
    """
    Scale data by population for each state.
    """
    sum_last_three_cols = np.sum(data[:, 0, -3:], axis=1)
    sum_last_three_cols = sum_last_three_cols[:, np.newaxis, np.newaxis]
    return data / sum_last_three_cols

def scale_data_by_pop_single_state(data):
    """
    Scale data by population for a single state.
    """
    sum_last_three_cols = np.sum(data[0, -3:])
    return data / sum_last_three_cols

def calculate_mean_age(data):
    """
    Calculate the mean age based on population data.
    """
    age_weights = np.array([7, 40, 80])
    populations = data[:, :, -3:]
    mean_age = np.sum(populations * age_weights, axis=2) / np.sum(populations, axis=2)
    return mean_age

def softplus(data, beta):
    """
    Apply the softplus function to the data.
    """
    return 1/beta * np.log(1 + np.exp(beta * data))