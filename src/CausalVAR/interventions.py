import numpy as np
import copy
from typing import Dict, List, Tuple
from statsmodels.tsa.vector_ar.svar_model import SVARProcess
from numpy.typing import NDArray


def apply_force(process: SVARProcess, int_vector: NDArray, steps: int, window: float = np.inf) -> NDArray:
    """
    Apply a force (intervention) to the SVAR process and calculate the effects over time.

    :param process: SVAR process
    :param int_vector: Intervention vector
    :param steps: Number of time steps
    :param window: Time window for force application (default: infinite)
    :return: Matrix of effects over time
    """
    size = process.A_solve.shape[0]
    interventional_vector = np.array(int_vector).reshape(-1, 1)
    effects_vector = np.zeros(shape=(0, size))
    effects = np.zeros(size).reshape(-1, 1)
    response_matrices = process.svar_ma_rep(maxn=steps)

    for i in range(steps):
        if i <= window:
            effects += response_matrices[i] @ interventional_vector
        else:
            effects = np.sum(response_matrices[i - window:i + 1], axis=0) @ interventional_vector
        effects_vector = np.vstack([effects_vector, effects.T])
    return effects_vector


def mu_to_v(mapping: Dict[str, int], svar_process: SVARProcess, mask: NDArray, names: List[str],
            intervention_effects: Dict[str, float]) -> NDArray:
    """
    Convert intervention effects to a force vector.

    :param mapping: Dictionary mapping variable names to indices
    :param svar_process: SVAR process
    :param mask: Mask for intervention application
    :param names: List of variable names
    :param intervention_effects: Dictionary of intervention effects
    :return: Force vector
    """
    force_vector = np.zeros(len(names)).reshape(-1, 1)
    for col, value in intervention_effects.items():
        interventional_vector = np.zeros(len(names)).reshape(-1, 1)
        index = mapping[col]
        interventional_vector[index] = value
        force_vector += (svar_process.A_solve @ svar_process._char_mat @ interventional_vector) * mask
    return force_vector


def additive_intervention(simulation: 'SVARSimulation', intervention_effects: Dict[str, float],
                          names: List[str], intervention_steps: int, asymptotic: bool = False) -> NDArray:
    """
    Apply an additive intervention to the SVAR simulation.

    :param simulation: SVAR simulation object
    :param intervention_effects: Dictionary of intervention effects
    :param names: List of variable names
    :param intervention_steps: Number of steps for intervention
    :param asymptotic: Whether to use asymptotic
    :return: Effects of the intervention
    """
    mask = np.zeros(len(names)).reshape(-1, 1)
    mapping = {names[i]: i for i in range(len(names))}

    for col, value in intervention_effects.items():
        index = mapping[col]
        mask[index] = 1

    interventional_vector = np.zeros(len(names)).reshape(-1, 1)
    if asymptotic:
        interventional_vector = mu_to_v(mapping, simulation.svar_process, mask, names, intervention_effects)
    else:
        for col, value in intervention_effects.items():
            index = mapping[col]
            interventional_vector[index] = value

    effects = apply_force(simulation.svar_process, interventional_vector, intervention_steps)
    return effects


def forcing_intervention(simulation: 'SVARSimulation', intervention_effects: Dict[str, float],
                         names: List[str], epsilon_scale: float, F: float, intervention_steps: int,
                         asymptotic: bool = False) -> Tuple['SVARSimulation', NDArray]:
    """
    Apply a forcing intervention to the SVAR simulation.

    :param simulation: SVAR simulation object
    :param intervention_effects: Dictionary of intervention effects
    :param names: List of variable names
    :param epsilon_scale: Scale factor for the error term
    :param F: Forcing factor
    :param intervention_steps: Number of steps for intervention
    :param asymptotic: Whether to use asymptotic effects
    :return: Tuple of new simulation object and effects of the intervention
    """
    mask = np.zeros(len(names)).reshape(-1, 1)
    mapping = {names[i]: i for i in range(len(names))}

    for col, value in intervention_effects.items():
        index = mapping[col]
        mask[index] = 1

    A_0 = simulation.A_0
    coefs = simulation.coefs

    F_matrix = np.diagflat(mask) * F
    new_A_0 = A_0 + F_matrix
    new_A_0_inv = np.linalg.inv(new_A_0)
    new_coefs = new_A_0_inv @ coefs
    new_sigma_u = new_A_0_inv @ np.eye(len(names)) * epsilon_scale @ new_A_0_inv.T

    new_simulation = copy.deepcopy(simulation)
    new_simulation.update_parameters(coefs=new_coefs, sigma_u=new_sigma_u, A_0=new_A_0)
    interventional_vector = np.zeros(len(names)).reshape(-1, 1)

    if asymptotic:
        interventional_vector = mu_to_v(mapping, new_simulation.svar_process, mask, names, intervention_effects)
    else:
        for col, value in intervention_effects.items():
            index = mapping[col]
            interventional_vector[index] = F * value

    effects = apply_force(new_simulation.svar_process, interventional_vector, intervention_steps)
    return new_simulation, effects