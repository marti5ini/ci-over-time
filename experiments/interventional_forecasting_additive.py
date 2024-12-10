import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from experiments.utils import get_forecast, calculate_score, clone_effects
from src.CausalVAR.fitting import var_fit_with_r
from src.CausalVAR.interventions import additive_intervention
from src.CausalVAR.simulation import SVARSimulation
from src.generation.setup_simulation import setup_german_simulation, setup_inverted_pendulum_simulation


class AdditiveInterventions:
    def __init__(self, dataset_name: str, normalize: bool, results_csv_path: str, metric: str,
                 collapse: bool, axis: Tuple[int, ...], future_steps: int, train_steps_list: List[int],
                 repetitions: int = 10):
        """
        Initialize the Additive Interventions Experiment.

        :param dataset_name: Name of the dataset to use ('inverted' or 'german')
        :param normalize: Whether to normalize the data
        :param results_csv_path: Path to save results
        :param metric: Metric to use for evaluation
        :param collapse: Whether to collapse results
        :param axis: Axis for collapsing results
        :param future_steps: Number of future steps to forecast
        :param train_steps_list: List of training steps to experiment with
        """
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.results_csv_path = results_csv_path
        self.metric = metric
        self.collapse = collapse
        self.axis = axis
        self.future_steps = future_steps
        self.train_steps_list = train_steps_list

        self._setup_dataset()
        self.true_simulation = SVARSimulation(coefs=self.true_matrices, sigma_u=self.true_sigma_u,
                                              names=self.column_names, A_0=self.A0)
        self.results_df = self._load_results_df()
        self.repetitions = repetitions

    def _setup_dataset(self):
        """Set up the dataset based on the dataset name."""
        if self.dataset_name == 'pendulum':
            self._setup_inverted_pendulum()
        else:
            self._setup_german()

    def _setup_inverted_pendulum(self):
        """Set up the inverted pendulum dataset."""
        (self.true_matrices, self.true_sigma_u, self.column_names,
         self.A0, self.n_lags, self.relations_python) = setup_inverted_pendulum_simulation()
        self.n_simulations_list = [1]
        self.past_steps = 40
        self.n_simulation_test = 5
        self.intervention_dict = {'x2': 0.3}
        self.target = 0

    def _setup_german(self):
        """Set up the German credit dataset."""
        (self.true_matrices, self.true_sigma_u, self.column_names, self.A0,
         self.n_lags, self.relations_python) = setup_german_simulation()
        self.n_simulations_list = [1]
        self.past_steps = 40
        self.n_simulation_test = 5
        self.intervention_dict = {'Expertise': 0.2}
        self.target = 6

    def _load_results_df(self) -> pd.DataFrame:
        """Load or create the results DataFrame."""
        if os.path.exists(self.results_csv_path):
            return pd.read_csv(self.results_csv_path)
        else:
            return pd.DataFrame(columns=['dataset', 'train_steps', 'n_simulations',
                                         'score', 'score_type', 'repetition', 'future steps'])

    def run(self):
        """Run the experiment for all train steps and repetitions."""
        for train_steps in self.train_steps_list:
            print(f'Train steps: {train_steps}')
            for repetition in range(self.repetitions):
                print(f'Repetition: {repetition + 1}')
                if self._should_skip_iteration(train_steps, repetition):
                    print('Result already exists. Skipping...')
                    continue

                self._run_iteration(train_steps, repetition)

    def _should_skip_iteration(self, train_steps: int, repetition: int) -> bool:
        """Check if the current iteration should be skipped."""
        return not self.results_df[(self.results_df['train_steps'] == train_steps) &
                                   (self.results_df['dataset'] == self.dataset_name) &
                                   (self.results_df['n_simulations'] == 1) &
                                   (self.results_df['repetition'] == repetition) &
                                   (self.results_df['future steps'] == self.future_steps) &
                                   (self.results_df['score_type'] == self.metric)].empty

    @staticmethod
    def additive_forecasting(test_forecast, predicted_forecast, test_effects, predicted_effects):
        test_if = test_forecast + test_effects
        predicted_if = predicted_forecast + predicted_effects
        return test_if, predicted_if

    def _run_iteration(self, train_steps: int, repetition: int):
        """Run a single iteration of the experiment."""
        seed = repetition
        train_data = self._generate_train_data(train_steps, seed)
        fitted_simulation = self._fit_simulation(train_data)
        _ = fitted_simulation.generate_data(n_simulations=1, steps=2)
        total_data = self._generate_total_data()
        past_data = total_data[:, :self.past_steps, :]

        test_forecast = get_forecast(self.true_simulation, past_data, self.future_steps)
        predicted_forecast = get_forecast(fitted_simulation, past_data, self.future_steps)
        print('Forecast completed.')

        test_effects = self._get_effects(self.true_simulation)
        predicted_effects = self._get_effects(fitted_simulation)
        print('Effect Estimation completed.')

        true_additive, predicted_additive = self.additive_forecasting(test_forecast, predicted_forecast,
                                                                      test_effects, predicted_effects)

        score = calculate_score(test_effects, predicted_effects, axis=self.axis,
                                metric_type=self.metric,
                                mean_collapse=self.collapse)[self.target]

        self._save_results(train_steps, repetition, score)

    def _generate_train_data(self, train_steps: int, seed: int) -> np.ndarray:
        """Generate training data."""
        train_data, _ = self.true_simulation.generate_data(steps=train_steps,
                                                           n_simulations=1, seed=seed)
        return train_data

    def _fit_simulation(self, train_data: np.ndarray) -> SVARSimulation:
        """Fit a simulation model to the training data."""
        fitted_matrices, fitted_sigma_u = var_fit_with_r(train_data, n_lags=self.n_lags,
                                                         column_names=self.column_names)
        return SVARSimulation(fitted_matrices, fitted_sigma_u, self.column_names, A_0=self.A0)

    def _generate_total_data(self) -> np.ndarray:
        """Generate total data for the experiment."""
        total_data, _ = self.true_simulation.generate_data(steps=self.past_steps + self.future_steps,
                                                           n_simulations=self.n_simulation_test, seed=40)
        if self.normalize:
            cov_matrix = self.true_simulation.svar_process.acf(self.n_lags + 1)[0]
            self.std_array = np.diag(np.sqrt(np.abs(cov_matrix))).reshape(1, 1, -1)
            total_data = total_data / self.std_array
        return total_data

    def _get_effects(self, simulation: SVARSimulation) -> np.ndarray:
        """Get effects for a given simulation."""
        effects = additive_intervention(simulation, self.intervention_dict, self.column_names,
                                        self.future_steps)
        additive_effects = clone_effects(self.n_simulation_test, effects)
        return additive_effects / self.std_array[0] if self.normalize else additive_effects

    def _save_results(self, train_steps: int, repetition: int, score: float):
        """Save the results of the experiment."""
        results = [{
            'dataset': self.dataset_name,
            'train_steps': train_steps,
            'n_simulations': 1,
            'score_mean': score,
            'score_type': self.metric,
            'future steps': self.future_steps,
            'repetition': repetition,
        }]

        results_temp = pd.DataFrame(results)
        results_temp.to_csv(self.results_csv_path, index=False, mode='a',
                            header=not os.path.exists(self.results_csv_path))


if __name__ == '__main__':

    train_steps_list = [100, 500, 1000]
    results_csv_path = 'datasets/interventional_forecasting.csv'
    normalize = True
    collapse = False
    axis = (0, 1)

    for dataset_name in ['german', 'pendulum']:
        print(f"Running experiments for dataset: {dataset_name}")
        for metric in ['rmse', 'smape', 'mae']:
            for future_steps in [1, 10]:
                print(f"Metric: {metric}, Future steps: {future_steps}")
                experiment = AdditiveInterventions(
                    dataset_name=dataset_name,
                    normalize=normalize,
                    results_csv_path=results_csv_path,
                    metric=metric,
                    collapse=collapse,
                    axis=axis,
                    future_steps=future_steps,
                    train_steps_list=train_steps_list
                )
                experiment.run()
