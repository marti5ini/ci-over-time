import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

class GenerateTimeSeriesData:
    def __init__(self, dataframe: pd.DataFrame, n_lags: int = 3, epsilon_scale: float = 0.2,
                 relations: Optional[List[Tuple[int, int, int, float]]] = None):
        """
        Initialize the time series data generator.

        :param dataframe: Input dataframe
        :param n_lags: Number of lags to consider
        :param epsilon_scale: Scale of the error term
        :param relations: List of relations (lag, source, target, weight)
        """
        self.df = dataframe
        self.column_names = list(self.df.columns)
        self.n_columns = len(self.column_names)
        self.n_samples = len(self.df)
        self.n_lags = n_lags
        self.epsilon_scale = epsilon_scale
        self.relations = relations or []
        self.A_matrices = self.initialize_matrices()

    def initialize_matrices(self) -> List[np.ndarray]:
        """
        Initialize the A matrices based on the given relations.

        :return: List of initialized A matrices
        """
        A_matrices = [np.zeros((self.n_columns, self.n_columns)) for _ in range(self.n_lags + 1)]
        np.fill_diagonal(A_matrices[0], 1)
        for relation in self.relations:
            lag, source, target, weight = relation
            A_matrices[lag][target][source] = weight
        return A_matrices

    def add_memory(self, memory_coeff: List[float]) -> None:
        """
        Add memory coefficients to the A matrices.

        :param memory_coeff: List of memory coefficients
        """
        for i in range(len(memory_coeff)):
            self.A_matrices[1][i, i] = memory_coeff[i]

    def calculate_y_t(self, t: int, sample: int, epsilon: np.ndarray, df_time: pd.DataFrame, col: int) -> np.ndarray:
        """
        Calculate the values for a specific time step using the autoregressive process.

        :param t: Current time step
        :param sample: Sample index
        :param epsilon: Error term
        :param df_time: Time series dataframe
        :param col: Column index
        :return: Calculated values for the time step
        """
        A_0_inv = np.linalg.inv(self.A_matrices[0])

        if t == 0:
            return self.df.iloc[sample, :].values

        previous_values = [df_time[(df_time['Time'] == t - previous_time) & (df_time['ID'] == sample)].iloc[:,
                           col:-1].values.flatten().reshape(-1, 1) for previous_time in range(1, min(t, self.n_lags) + 1)]

        y_t = A_0_inv @ (sum(
            A @ y for A, y in zip(self.A_matrices[:len(previous_values)], previous_values)) + epsilon)

        return y_t.flatten()

    def simulate_data(self, age_range: List[int]) -> pd.DataFrame:
        """
        Simulate time series data based on the given age range.

        :param age_range: List of ages to simulate
        :return: Simulated time series data
        """
        starting_age = age_range[0]
        df_time = pd.DataFrame(columns=['Time', 'ID'] + self.column_names)
        min_s, max_s = 1, 1
        for t in range(0, len(age_range)):
            for sample in range(self.n_samples):
                epsilon = np.zeros(len(self.df.columns))
                for col in range(self.n_columns):
                    epsilon[col] = np.random.normal(0, scale=min_s + ((max_s - min_s) / self.n_columns) * col)
                epsilon = epsilon.reshape(-1, 1)

                row = {'Time': t, 'ID': sample, 'A': t + starting_age}
                n_col = len(row) - 1

                y_t = self.calculate_y_t(t, sample, epsilon, df_time, n_col)
                row.update({col: val for col, val in zip(self.df.columns, y_t)})
                df_time = pd.concat([df_time, pd.DataFrame([row])], ignore_index=True)
        return df_time

    @staticmethod
    def _add_time(group: pd.DataFrame, c_Age: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
        """
        Apply time-based adjustments to a group of data.

        :param group: Group of data
        :param c_Age: Age coefficients
        :param baseline: Baseline adjustments
        :return: Adjusted group of data
        """
        t = group['Time'].iloc[0]
        group.iloc[:, 2:-1] += c_Age * t * 0.1 + baseline
        return group

    def add_trends(self, time_series: pd.DataFrame, c_Age: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
        """
        Add trends to the simulated time series data.

        :param time_series: Base time series data
        :param c_Age: Age coefficients
        :param baseline: Baseline adjustments
        :return: Time series data with added trends
        """
        df_time = (time_series.groupby('Time', group_keys=True).apply(self._add_time, c_Age, baseline)
                   .reset_index(drop=True))
        return df_time.iloc[:, 2:]