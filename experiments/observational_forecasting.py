import os
from typing import List, Any
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TSMixerModel, TiDEModel, LinearRegressionModel

from experiments.utils import to_lagged_tensor, get_forecast, create_params
from src.generation.setup_simulation import setup_german_simulation, setup_inverted_pendulum_simulation
from src.CausalVAR.simulation import SVARSimulation
from src.CausalVAR.fitting import var_fit_with_r
from darts.metrics import mae, rmse, smape, r2_score

metrics_dict = {'r2': r2_score, 'rmse': rmse, 'smape': smape, 'mae': mae}

class ObservationalForecasting:
    def __init__(self, dataset_name: str = 'german', normalize: bool = True, repetitions: int = 10,
                 results_csv_path: str = 'datasets/observational_forecasting.csv',
                 metric: str = 'mae', collapse: bool = False, axis: tuple = (0)):
        """
        Initialize the experiment.

        :param dataset_name: Name of the dataset to use
        :param normalize: Whether to normalize the data
        :param repetitions: Number of repetitions for the experiment
        :param results_csv_path: Path to save results
        :param metric: Metric to use for evaluation
        :param collapse: Whether to collapse results
        :param axis: Axis for collapsing results
        """
        self.dataset_name = dataset_name
        self.normalize = normalize
        self.results_csv_path = results_csv_path
        self.metric_name = metric
        self.metric_function = metrics_dict[metric]
        self.train_steps_list = [100, 500, 1000]
        self.forecast_prediction_steps_list = [1, 10]
        self.test_steps = 2000
        self.collapse = collapse
        self.axis = axis
        self.seed = None
        self.n_repetitions = repetitions
        self.time_index = None

        self.models_dir = f'saved_models_{dataset_name}'
        os.makedirs(self.models_dir, exist_ok=True)

        self._setup_dataset()

        self.true_simulation = SVARSimulation(coefs=self.true_matrices, sigma_u=self.true_sigma_u,
                                              names=self.column_names, A_0=self.A0)
        self.results_df = self._load_results_df()
        self.models = ['TSMixer', 'VAR', 'Oracle', 'TiDE', 'DLinear']

    def _setup_dataset(self):
        """Set up the dataset based on the dataset name."""
        if self.dataset_name == 'pendulum':
            self._setup_inverted_pendulum()
        else:
            self._setup_german()

    def _setup_inverted_pendulum(self):
        """Set up the inverted pendulum dataset."""
        (self.true_matrices, self.true_sigma_u, self.column_names, self.A0,
         self.n_lags, self.relations_python) = setup_inverted_pendulum_simulation()
        self.n_columns = len(self.column_names)
        self.val_steps = 100
        self.forecast_steps = max(self.forecast_prediction_steps_list)
        self.target = 'x2'

    def _setup_german(self):
        """Set up the German credit dataset."""
        (self.true_matrices, self.true_sigma_u, self.column_names, self.A0,
         self.n_lags, self.relations_python) = setup_german_simulation()
        self.n_columns = len(self.column_names)
        self.val_steps = 300
        self.forecast_steps = max(self.forecast_prediction_steps_list)
        self.target = 'CreditScore'

    def _load_results_df(self) -> pd.DataFrame:
        """Load or create the results DataFrame."""
        if os.path.exists(self.results_csv_path):
            return pd.read_csv(self.results_csv_path)
        else:
            return pd.DataFrame(columns=['dataset', 'train_steps', 'n_simulations', 'forecast_prediction_steps',
                                         'model', 'score_mean', 'score_type', 'repetition'])

    def run(self):
        """Run the experiment."""
        for train_steps in self.train_steps_list:
            print(f"Train Steps {train_steps}")
            for repetition in range(self.n_repetitions):
                print(f"Repetition {repetition + 1}/{self.n_repetitions}")
                self._run_repetition(train_steps, repetition)

    def _run_repetition(self, train_steps: int, repetition: int):
        """Run a single repetition of the experiment."""
        missing_models = self._check_missing_models(repetition, train_steps)
        if not missing_models:
            print(f"All models for repetition {repetition} and train_steps {train_steps} exist. Skipping fitting.")
            return

        print(f"Training missing models: {', '.join(missing_models)}")

        self._generate_and_process_data(train_steps, repetition)
        self._fit_and_save_models(train_steps, repetition, missing_models)
        self._predict_and_evaluate(train_steps, repetition)

    def _check_missing_models(self, repetition: int, train_steps: int) -> List[str]:
        """Check which models are missing for the current repetition and train steps."""
        missing_models = []
        for model in self.models:
            if model == 'Oracle':
                continue
            if not os.path.exists(f"{self.models_dir}/{model}_{repetition}_{train_steps}.pkl"):
                missing_models.append(model)
        return missing_models

    def _generate_and_process_data(self, train_steps: int, repetition: int):
        """Generate and process data for the experiment."""
        self.seed = repetition
        self.data, _ = self.true_simulation.generate_data(n_simulations=1,
                                                          steps=train_steps + self.val_steps + self.test_steps,
                                                          seed=self.seed)
        if self.normalize:
            cov_matrix = self.true_simulation.svar_process.acf(self.n_lags + 1)[0]
            self.std_array = np.diag(np.sqrt(np.abs(cov_matrix))).reshape(1, 1, -1)
            self.data = self.data / self.std_array
        else:
            self.std_array = 1

    def _create_darts_series(self, data):
        df = pd.DataFrame(data=data[0], columns=self.column_names)
        df['Time'] = pd.date_range(start='2000-01-01', periods=len(df), freq='W')
        return TimeSeries.from_dataframe(df, time_col='Time', value_cols=self.column_names).astype('float32')

    def _save_models(self, fitted_models, repetition, train_steps):
        for model_name, model in fitted_models.items():
            if model_name == 'VAR':
                np.save(f"{self.models_dir}/VAR_matrices_{repetition}_{train_steps}.npy", model.coefs)
                np.save(f"{self.models_dir}/VAR_sigma_u_{repetition}_{train_steps}.npy", model.sigma_u)
            elif model_name == 'TSMixer':
                model.save(f"{self.models_dir}/TSMixer_{repetition}_{train_steps}.pkl")
            elif model_name == 'TiDE':
                model.save(f"{self.models_dir}/TiDE_{repetition}_{train_steps}.pkl")
            elif model_name == 'DLinear':
                model.save(f"{self.models_dir}/DLinear_{repetition}_{train_steps}.pkl")

    def _fit_var(self, train_steps):
        fitted_models = {}

        # VAR
        X_var_train = self.data[:, :train_steps, :]
        fitted_matrices, fitted_sigma_u = var_fit_with_r(X_var_train, n_lags=self.n_lags,
                                                         column_names=self.column_names)
        fitted_simulation = SVARSimulation(coefs=fitted_matrices, sigma_u=fitted_sigma_u, names=self.column_names,
                                           A_0=self.A0)
        fitted_models['VAR'] = fitted_simulation

        return fitted_models

    def _fit_models_darts(self, series, fitted_models, train_steps, repetition):
        train_series, temp = series.split_after(train_steps-1)
        val_series, _ = temp.split_after(self.test_steps-1)
        tsmixer_params = create_params(input_chunk_length=self.n_lags, output_chunk_length=1)
        tide_params = create_params(input_chunk_length=self.n_lags, output_chunk_length=1)

        tsmixer_model = TSMixerModel(**tsmixer_params, model_name=f"tsmixer_{repetition}_{train_steps}", use_static_covariates=False)
        tide_model = TiDEModel(**tide_params, model_name=f"tide_{repetition}_{train_steps}", use_static_covariates=False)

        tsmixer_model.fit(train_series, val_series=val_series)
        tide_model.fit(train_series, val_series=val_series)

        dlinear_model = LinearRegressionModel(lags=self.n_lags, output_chunk_length=1, use_static_covariates=False)

        fitted_models['TSMixer'] = tsmixer_model.load_from_checkpoint(model_name=tsmixer_model.model_name, best=True)
        fitted_models['TiDE'] = tide_model.load_from_checkpoint(model_name=tide_model.model_name, best=True)
        fitted_models['DLinear'] = dlinear_model.fit(train_series, val_series=val_series)
        return fitted_models

    def _fit_and_save_models(self, train_steps: int, repetition: int, missing_models: List[str]):
        """Fit and save missing models."""
        fitted_models = {}
        if 'VAR' in missing_models:
            fitted_models = self._fit_var(train_steps)

        series = self._create_darts_series(self.data)
        if set(missing_models) & {'TSMixer', 'TiDE', 'DLinear'}:
            fitted_models = self._fit_models_darts(series, fitted_models, train_steps, repetition)

        self._save_models(fitted_models, repetition, train_steps)

    def _load_models(self, repetition, train_steps):
        loaded_models = {}
        loaded_models['TSMixer'] = TSMixerModel.load(f"{self.models_dir}/TSMixer_{repetition}_{train_steps}.pkl")
        fitted_matrices = np.load(f"{self.models_dir}/VAR_matrices_{repetition}_{train_steps}.npy", allow_pickle=True)
        fitted_sigma_u = np.load(f"{self.models_dir}/VAR_sigma_u_{repetition}_{train_steps}.npy", allow_pickle=True)
        loaded_models['VAR'] = SVARSimulation(coefs=fitted_matrices, sigma_u=fitted_sigma_u, names=self.column_names,
                                              A_0=self.A0)

        loaded_models['TiDE'] = TiDEModel.load(f"{self.models_dir}/TiDE_{repetition}_{train_steps}.pkl")
        loaded_models['DLinear'] = LinearRegressionModel.load(f"{self.models_dir}/DLinear_{repetition}_{train_steps}.pkl")

        return loaded_models

    def _generate_test_data(self, new_seed=45):
        new_data, _ = self.true_simulation.generate_data(n_simulations=1,
                                                         steps=100 + self.val_steps + self.test_steps,
                                                         seed=new_seed)

        cov_matrix = self.true_simulation.svar_process.acf(self.n_lags + 1)[0]
        self.std_array = np.diag(np.sqrt(np.abs(cov_matrix))).reshape(1, 1, -1)
        new_data = new_data / self.std_array

        full_series = self._create_darts_series(new_data)
        train, test = full_series.split_after(0.05)

        return new_data, full_series, train, test

    def _predict_and_evaluate(self, train_steps: int, repetition: int):
        """Predict using fitted models and evaluate the results."""
        test_data, full_time_series, train_time_series_data, test_time_series_data = self._generate_test_data()

        loaded_models = self._load_models(repetition, train_steps)

        for model_name, model in loaded_models.items():
            self._evaluate_model(model_name, model, test_data, test_time_series_data, full_time_series, train_steps, repetition)

        self._evaluate_oracle(test_data, test_time_series_data, train_steps, repetition)

    def _evaluate_model(self, model_name: str, model: Any, test_data: np.ndarray, test_time_series_data: TimeSeries,
                        full_time_series: TimeSeries, train_steps: int, repetition: int):
        """Evaluate a single model."""
        for forecast_prediction_steps in self.forecast_prediction_steps_list:
            if self._should_skip_evaluation(train_steps, model_name, forecast_prediction_steps, repetition):
                print(f'Skipping Forecast Prediction Steps: {forecast_prediction_steps}.')
                continue

            predictions = self._get_predictions(model_name, model, test_data, test_time_series_data, full_time_series, forecast_prediction_steps)
            score = self.metric_function(test_time_series_data[self.target], predictions[self.target])

            self._save_results(train_steps, model_name, forecast_prediction_steps, score, repetition)

    def _evaluate_oracle(self, test_data: np.ndarray, test_time_series_data: TimeSeries, train_steps: int, repetition: int):
        """Evaluate the oracle model."""
        oracle_simulation = self.true_simulation
        for forecast_prediction_steps in self.forecast_prediction_steps_list:
            if self._should_skip_evaluation(train_steps, 'Oracle', forecast_prediction_steps, repetition):
                print(f'Skipping Forecast Prediction Steps: {forecast_prediction_steps}.')
                continue

            predictions = self._get_oracle_predictions(oracle_simulation, test_data, test_time_series_data, forecast_prediction_steps)
            score = self.metric_function(test_time_series_data[self.target], predictions[self.target])

            self._save_results(train_steps, 'Oracle', forecast_prediction_steps, score, repetition)

    def _should_skip_evaluation(self, train_steps: int, model_name: str, forecast_prediction_steps: int, repetition: int) -> bool:
        """Check if evaluation should be skipped."""
        return not self.results_df[(self.results_df['train_steps'] == train_steps) &
                                   (self.results_df['dataset'] == self.dataset_name) &
                                   (self.results_df['n_simulations'] == 1) &
                                   (self.results_df['model'] == model_name) &
                                   (self.results_df['forecast_prediction_steps'] == forecast_prediction_steps) &
                                   (self.results_df['score_type'] == self.metric_name) &
                                   (self.results_df['repetition'] == repetition)
                                   ].empty

    def _get_predictions(self, model_name: str, model: Any, test_data: np.ndarray, test_time_series_data: TimeSeries,
                         full_time_series: TimeSeries, forecast_prediction_steps: int) -> TimeSeries:
        """Get predictions for a given model."""
        if model_name == 'VAR':
            return self._get_var_predictions(model, test_data, test_time_series_data, forecast_prediction_steps)
        elif model_name in ['TSMixer', 'TiDE', 'DLinear']:
            return self._get_darts_predictions(model, full_time_series, test_time_series_data, forecast_prediction_steps)

    def _get_var_predictions(self, model: Any, test_data: np.ndarray, test_time_series_data: TimeSeries,
                             forecast_prediction_steps: int) -> TimeSeries:
        """Get predictions for VAR model."""
        X_test = to_lagged_tensor(
            test_data[:, -len(test_time_series_data) - self.n_lags - forecast_prediction_steps + 1:, :],
            lags=self.n_lags)
        X_test = X_test[0, :-1, :, :].transpose(1, 0, 2)
        _ = model.generate_data(n_simulations=1, steps=2, seed=4)
        predictions_var = get_forecast(model, X_test, forecast_prediction_steps)[
                          :len(test_time_series_data),
                          forecast_prediction_steps - 1, :]
        df_var = pd.DataFrame(data=predictions_var, columns=self.column_names)
        df_var['Time'] = self.time_index
        return TimeSeries.from_dataframe(df_var, value_cols=list(df_var.columns)[:-1],
                                         time_col='Time').astype('float32')

    def _get_darts_predictions(self, model: Any, full_time_series: TimeSeries, test_time_series_data: TimeSeries,
                               forecast_prediction_steps: int) -> TimeSeries:
        """Get predictions for Darts models."""
        predictions = model.historical_forecasts(
            full_time_series,
            start=0,
            forecast_horizon=forecast_prediction_steps,
            stride=1,
            retrain=False,
            last_points_only=True,
            num_samples=1
        )
        predictions = predictions.slice_intersect(test_time_series_data)
        if isinstance(model, TSMixerModel):
            self.time_index = predictions.time_index
        return predictions

    def _get_oracle_predictions(self, oracle_simulation: Any, test_data: np.ndarray, test_time_series_data: TimeSeries,
                                forecast_prediction_steps: int) -> TimeSeries:
        """Get predictions for the oracle model."""
        X_test = to_lagged_tensor(
            test_data[:, -len(test_time_series_data) - self.n_lags - forecast_prediction_steps + 1:, :],
            lags=self.n_lags)
        X_test = X_test[0, :-1, :, :].transpose(1, 0, 2)
        predictions_oracle = get_forecast(oracle_simulation, X_test * self.std_array[0],
                                          forecast_prediction_steps)[:len(test_time_series_data),
                             forecast_prediction_steps - 1, :] / self.std_array[0]
        df_oracle = pd.DataFrame(data=predictions_oracle, columns=self.column_names)
        df_oracle['Time'] = self.time_index
        return TimeSeries.from_dataframe(df_oracle, value_cols=list(df_oracle.columns)[:-1],
                                         time_col='Time').astype('float32')

    def _save_results(self, train_steps: int, model_name: str, forecast_prediction_steps: int, score: float, repetition: int):
        """Save the evaluation results."""
        results = [{
            'dataset': self.dataset_name,
            'train_steps': train_steps,
            'n_simulations': 1,
            'forecast_prediction_steps': forecast_prediction_steps,
            'model': model_name,
            'score': score,
            'score_type': self.metric_name,
            'repetition': repetition
        }]

        results_temp = pd.DataFrame(results)
        results_temp.to_csv(self.results_csv_path, mode='a', header=not os.path.exists(self.results_csv_path), index=False)

if __name__ == '__main__':
    experiment = ObservationalForecasting(dataset_name='german', metric='rmse')
    experiment.run()