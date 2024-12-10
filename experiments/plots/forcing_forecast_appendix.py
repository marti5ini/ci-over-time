import os
import sys
import numpy as np
import scipy as sp
from typing import Dict, Tuple

# Set up paths to import necessary modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from plotting_functions import plot_ai_fi_intervention_forecast
from src.generation.setup_simulation import setup_german_simulation
from experiments.utils import get_forecast, clone_effects
from src.CausalVAR.interventions import forcing_intervention
from src.CausalVAR.simulation import SVARSimulation
from src.CausalVAR.fitting import var_fit_with_r
from config import YEAR_START, FUTURE_STEPS, PAST_STEPS

# Mapping of column names to indices
column_names_german = {
    'Expertise': 0,
    'Responsibility': 1,
    'Loan Amount': 2,
    'LoanDuration': 3,
    'Income': 4,
    'Savings': 5,
    'Credit Score': 6
}


def update_column_name(index: int) -> str:
    """
    Returns the column name corresponding to a given index.

    Args:
        index (int): The index of the column.

    Returns:
        str: The name of the column.
    """
    for name, value in column_names_german.items():
        if value == index:
            return name
    return None


def forcing_forecasting(true_forecast: np.ndarray, true_effects: np.ndarray, predicted_effects: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Combines forecast data with intervention effects.

    Args:
        true_forecast (np.ndarray): The forecasted data before intervention.
        true_effects (np.ndarray): The actual effects of the intervention.
        predicted_effects (np.ndarray): The predicted effects of the intervention.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The updated true and predicted effects after forcing.
    """
    true_forcing_effects = true_forecast + true_effects
    pred_forcing_effects = true_forecast + predicted_effects
    return true_forcing_effects, pred_forcing_effects


class PlotForcing:
    def __init__(self, dataset_name: str = 'german', force: int = 3):
        self.dataset_name = dataset_name
        self.threshold = 0
        self.train_steps = 500
        self.n_simulations = 1
        self.n_simulation_test = 1
        self.year_start = int(YEAR_START)
        self.force = force
        self.asymptotic = True

        # Set up the SVAR model for the German dataset
        (self.true_matrices, self.true_sigma_u, self.column_names, self.A0,
         self.n_lags, self.relations_python) = setup_german_simulation()
        self.true_simulation = SVARSimulation(coefs=self.true_matrices, sigma_u=self.true_sigma_u,
                                              names=self.column_names, A_0=self.A0)
        self.future_steps = FUTURE_STEPS
        self.past_steps = PAST_STEPS
        self.simulation_index_to_plot = 0
        self.epsilon_scale = np.ones(len(self.column_names))

        # Define interventions to be applied during the simulation
        self.intervention_dict: Dict[str, float] = {'LoanDuration': 4, 'Income': 30}

    def run(self):
        # Generate training data using the true simulation model
        train_data, _ = self.true_simulation.generate_data(steps=self.train_steps,
                                                           n_simulations=self.n_simulations, seed=4)

        # Fit a VAR model to the training data
        fitted_matrices, fitted_sigma_u = var_fit_with_r(train_data, n_lags=self.n_lags,
                                                         column_names=self.column_names)

        # Set up a simulation model using the fitted parameters
        fitted_svar_simulation_r = SVARSimulation(fitted_matrices, fitted_sigma_u, self.column_names,
                                                  A_0=self.A0)
        fitted_data_r, _ = fitted_svar_simulation_r.generate_data(n_simulations=self.n_simulation_test)

        # Generate the full dataset (past and future) for the true model
        total_data, _ = self.true_simulation.generate_data(steps=self.past_steps + self.future_steps,
                                                           n_simulations=self.n_simulation_test, seed=17)

        # Smooth the data for plotting
        total_data[self.simulation_index_to_plot] = sp.ndimage.convolve1d(total_data[self.simulation_index_to_plot],
                                                                          np.ones(5) / 5, axis=0)
        past_data = total_data[:, :self.past_steps, :]

        # Generate forecasts based on past data
        true_forecast = get_forecast(self.true_simulation, past_data, self.future_steps)
        true_forecast_cov = self.true_simulation.svar_process.forecast_cov(self.future_steps)

        # Apply forcing interventions to the true simulation model
        true_forcing_simulation, true_forcing_single_effect = forcing_intervention(self.true_simulation,
                                                                                   self.intervention_dict,
                                                                                   self.column_names,
                                                                                   self.epsilon_scale, self.force,
                                                                                   self.future_steps,
                                                                                   asymptotic=self.asymptotic)

        # Generate forecasts after the intervention
        true_intervened_forecast = get_forecast(true_forcing_simulation, past_data, self.future_steps)
        true_intervened_forecast_cov = true_forcing_simulation.svar_process.forecast_cov(self.future_steps)

        # Apply forcing interventions to the fitted model
        _, predicted_forcing_single_effect = forcing_intervention(
            fitted_svar_simulation_r,
            self.intervention_dict,
            self.column_names,
            self.epsilon_scale,
            self.force,
            self.future_steps,
            asymptotic=self.asymptotic)

        # Clone effects across multiple simulation runs
        true_forcing_effects = clone_effects(self.n_simulation_test, true_forcing_single_effect)
        predicted_forcing_effects = clone_effects(self.n_simulation_test, predicted_forcing_single_effect)

        # Combine forecasted data with intervention effects
        true_forcing_forecast_and_effects, predicted_forcing_forecast_and_effects = forcing_forecasting(
            true_intervened_forecast,
            true_forcing_effects,
            predicted_forcing_effects)

        # Plot results for selected columns
        for index_col in [3, 4, 6]:
            col_name = update_column_name(index_col)
            if col_name != list(self.intervention_dict.keys())[0]:
                self.title = f'Forcing Intervention {col_name}, on {list(self.intervention_dict.keys())[0]}={list(self.intervention_dict.values())[0]}'
            else:
                self.title = f'Forcing Intervention on {list(self.intervention_dict.keys())[0]} with Force={list(self.intervention_dict.values())[0]}'

            # Plot the forecast results with the applied intervention
            plot_ai_fi_intervention_forecast(past_data[self.simulation_index_to_plot],
                                             true_forecast[self.simulation_index_to_plot],
                                             true_forecast_cov,
                                             true_forcing_forecast_and_effects[self.simulation_index_to_plot],
                                             predicted_forcing_forecast_and_effects[self.simulation_index_to_plot],
                                             true_intervened_forecast_cov,
                                             additive=False,
                                             intervention_dict=self.intervention_dict,
                                             column_name=col_name,
                                             column_index=index_col,
                                             title=self.title,
                                             year_end=str(self.year_start + self.past_steps + self.future_steps))


# Run the simulation and plotting process
if __name__ == '__main__':
    plot_forcing_instance = PlotForcing()
    plot_forcing_instance.run()
