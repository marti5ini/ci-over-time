import numpy as np
from typing import List, Tuple


def add_memory(matrix: np.ndarray, memory_coeff_list: List[float]) -> np.ndarray:
    """
    Add memory coefficients to the diagonal of the matrix.

    :param matrix: Input matrix
    :param memory_coeff_list: List of memory coefficients
    :return: Modified matrix with memory coefficients
    """
    for i in range(len(memory_coeff_list)):
        matrix[i, i] = memory_coeff_list[i]
    return matrix


def initialize_matrices(relations: List[Tuple[int, int, int, float]], n_columns: int, n_lags: int) -> List[np.ndarray]:
    """
    Initialize matrices based on given relations.

    :param relations: List of relations (lag, source, target, weight)
    :param n_columns: Number of columns in each matrix
    :param n_lags: Number of lags
    :return: List of initialized matrices
    """
    A_matrices = [np.zeros((n_columns, n_columns)) for _ in range(n_lags + 1)]
    np.fill_diagonal(A_matrices[0], 1)
    for relation in relations:
        lag, source, target, weight = relation
        A_matrices[lag][target][source] = weight
    return A_matrices


def setup_german_simulation() -> Tuple[
    np.ndarray, np.ndarray, List[str], np.ndarray, int, List[Tuple[int, int, int, float]]]:
    """
    Set up a simulation for a German credit scoring model.

    :return: Tuple containing true matrices, true sigma_u, column names, A0 matrix, number of lags, and relations
    """
    relations_r = [
        # Super short run Effects
        (1, 4, 6, 0.3),  # Income -> Credit Score
        (1, 5, 6, 0.5),  # Savings -> Credit Score
        (1, 3, 6, -0.5),  # Duration -> Credit Score
        # Short-Term
        (2, 1, 4, 0.3),  # Responsibility -> Income
        (2, 2, 6, 0.5),  # Loan amount -> Credit Score
        (2, 4, 5, 0.2),  # Income -> Savings
        # Medium Term
        (3, 2, 3, 0.5),  # Loan amount -> Duration
        # Long Term
        (4, 0, 4, 0.8),  # Expertise -> Income
        (4, 0, 1, 0.3)  # Expertise -> Responsibility
    ]

    n_lags = 4
    column_names = ['Expertise', 'Responsibility', 'LoanAmount', 'LoanDuration', 'Income', 'Savings', 'CreditScore']
    memory = 0.95
    memory_coefficients = [memory] * 6 + [0]

    A_matrices = initialize_matrices(relations_r, n_columns=len(column_names), n_lags=n_lags)
    A_matrices[1] = add_memory(A_matrices[1], memory_coefficients)
    A0 = A_matrices[0]
    A_0_inv = np.linalg.inv(A0)

    epsilon_scale = np.ones(len(column_names))
    true_matrices = A_0_inv @ A_matrices[1:]
    true_sigma_u = A_0_inv @ np.eye(len(column_names)) * epsilon_scale @ A_0_inv.T
    relations_python = [
        # Instantaneous Effects
        (0, 4, 6, -0.003),  # Income -> Credit Score
        (0, 5, 6, -0.005),  # Savings -> Credit Score
        (0, 3, 6, 0.005),  # Duration -> Credit Score
        # Super short run Effects
        (1, 4, 6, 0.3),  # Income -> Credit Score
        (1, 5, 6, 0.5),  # Savings -> Credit Score
        (1, 3, 6, -0.5),  # Duration -> Credit Score
        # Short-Term
        (2, 1, 4, 0.3),  # Responsibility -> Income
        (2, 2, 6, 0.5),  # Loan amount -> Credit Score
        (2, 4, 5, 0.2),  # Income -> Savings
        # Medium Term
        (3, 2, 3, 0.5),  # Loan amount -> Duration
        # Long Term
        (4, 0, 4, 0.8),  # Expertise -> Income
        (4, 0, 1, 0.3)  # Expertise -> Responsibility
    ]

    return true_matrices, true_sigma_u, column_names, A0, n_lags, relations_python


def setup_inverted_pendulum_simulation() -> Tuple[
    np.ndarray, np.ndarray, List[str], np.ndarray, int, List[Tuple[int, int, int, float]]]:
    """
    Set up a simulation for an inverted pendulum system.

    :return: Tuple containing true matrices, true sigma_u, column names, A0 matrix, number of lags, and relations
    """
    column_names = ['x1', 'x2']
    relations_python = [(0, 0, 1, -0.0000003)]
    n_lags = 1
    n_variables = 2
    true_matrices = np.asarray([1.8 * np.array([[0.05, 0.5], [-0.5, 1]])])
    A0 = np.eye(n_variables)
    true_sigma_u = np.eye(n_variables)
    return true_matrices, true_sigma_u, column_names, A0, n_lags, relations_python
