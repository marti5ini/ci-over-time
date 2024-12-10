import numpy as np
from causalgraphicalmodels import StructuralCausalModel
from typing import Dict, Callable


class GermanDataGenerator:
    def __init__(self, base_seed: int = 3):
        """
        Initialize the German Data Generator.

        :param base_seed: Base seed for random number generation
        """
        self.base_seed: int = base_seed
        self.current_seed: int = base_seed
        self.rf: float = 0.3  # Random factor

    def next_seed(self) -> None:
        """Increment the current seed and set it for numpy."""
        self.current_seed += 1
        np.random.seed(self.current_seed)

    def normal(self, n_samples: int, k: float = 1) -> np.ndarray:
        """Generate normal distribution samples."""
        return np.random.normal(0, self.rf * k, size=n_samples)

    @staticmethod
    def linear(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Apply linear scaling to input."""
        return x / scale

    def f_G(self, n_samples: int) -> np.ndarray:
        """Generate Gender data."""
        np.random.seed(self.current_seed)
        return (np.random.binomial(1, 0.5, size=n_samples) - 0.5) * 2

    def f_A(self, n_samples: int) -> np.ndarray:
        """Generate Age data."""
        self.next_seed()
        return (-35 + np.random.gamma(10, scale=3.5, size=n_samples)) / 10

    def f_E(self, n_samples: int, G: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Generate Expertise data."""
        self.next_seed()
        l2norm = np.linalg.norm([1, 1, self.rf])
        noise = self.normal(n_samples)
        return self.linear(G + A + noise, scale=l2norm)

    def f_R(self, n_samples: int, G: np.ndarray, A: np.ndarray, E: np.ndarray) -> np.ndarray:
        """Generate Responsibility data."""
        np.random.seed(self.base_seed + 3)
        l2norm = np.linalg.norm([1, 2, 4, 2 * self.rf])
        noise = self.normal(n_samples, 2)
        return self.linear(G + 2 * A + 4 * E + noise, scale=l2norm)

    def f_L(self, n_samples: int, A: np.ndarray, G: np.ndarray) -> np.ndarray:
        """Generate Loan Amount data."""
        np.random.seed(self.base_seed + 4)
        l2norm = np.linalg.norm([1, 3 * self.rf])
        noise = self.normal(n_samples, 3)
        return self.linear(A + noise, scale=l2norm)

    def f_D(self, n_samples: int, G: np.ndarray, A: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Generate Loan Duration data."""
        np.random.seed(self.base_seed + 5)
        l2norm = np.linalg.norm([1, -0.5, 2, 2 * self.rf])
        noise = self.normal(n_samples, 2)
        return self.linear(G - 0.5 * A + 2 * L + noise, scale=l2norm)

    def f_I(self, n_samples: int, G: np.ndarray, A: np.ndarray, E: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Generate Income data."""
        np.random.seed(self.base_seed + 6)
        l2norm = np.linalg.norm([0.5, 5, 4, 1, 4 * self.rf])
        noise = self.normal(n_samples, 4)
        return self.linear(0.5 * G + A + 4 * E + 5 * R + noise, scale=l2norm)

    def f_S(self, n_samples: int, I: np.ndarray) -> np.ndarray:
        """Generate Savings data."""
        np.random.seed(self.base_seed + 7)
        l2norm = np.linalg.norm([5, 2 * self.rf])
        noise = self.normal(n_samples, 2)
        return self.linear(5 * I + noise, scale=l2norm)

    def f_Y(self, n_samples: int, I: np.ndarray, S: np.ndarray, L: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Generate Credit Score data."""
        np.random.seed(self.base_seed + 8)
        l2norm = np.linalg.norm([2, 3, -1, -1])
        return self.linear(2 * I + 3 * S + - L - D, scale=l2norm)


def load_german(n_samples: int = 1000, seed: int = 3, order: bool = True) -> pd.DataFrame:
    """
    Load German credit scoring dataset.

    :param n_samples: Number of samples to generate
    :param seed: Random seed
    :param order: Whether to reorder columns
    :return: DataFrame with generated data
    """
    column_names: Dict[str, str] = {
        'Gender': 'G', 'Age': 'A', 'Expertise': 'E', 'Responsibility': 'R',
        'Loan Amount': 'L', 'Loan Duration': 'D', 'Income': 'I', 'Savings': 'S', 'Credit Score': 'Y'
    }

    data_gen = GermanDataGenerator(base_seed=seed)

    structural_equations: Dict[str, Callable] = {
        abbreviation: getattr(data_gen, f'f_{abbreviation}')
        for abbreviation in column_names.values()
    }

    scm = StructuralCausalModel(structural_equations)
    df = scm.sample(n_samples=n_samples).astype(float)

    if order:
        # To ensure a specific position in the matrices corresponds
        # to a particular column, we arrange the df columns in a predetermined order.
        df['L'], df['R'] = df['R'], df['L']
        df = df.rename(columns={'L': 'R', 'R': 'L'})

    return df