import os
import sys
import numpy as np
import scipy as sp
from typing import Dict, Tuple

# Adjust the system path to include necessary modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import functions and classes
from plotting_functions import plot_ai_fi_intervention_forecast
from src.generation.setup_simulation import setup_german_simulation
from experiments.utils import get_forecast, clone_effects
from src.CausalVAR.interventions import forcing_intervention
from forcing_forecast_appendix import forcing_forecasting
from src.CausalVAR.simulation import SVARSimulation
from src.CausalVAR.fitting import var_fit_with_r
from config import YEAR_START, FUTURE_STEPS, PAST_STEPS


class PlotForcing:
    def __init__(self, dataset_name: str = 'german'):
        """
        Initialize the PlotForcing class with configuration for simulations and forecasts.

        Args:
            dataset_name (str): The name of the dataset used (default is 'german').
        """
        self.dataset_name = dataset_name
        self.train_steps = 200
        self.n_simulations = 5
        self.n_simulation_test = 5
        self.year_start = int(YEAR_START)
        self.force = 1
        self.asymptotic = True
        self.column_name_for_plot = 'Expertise'
        self.column_index_for_plot = 0

        # Load simulation setup
        (self.true_matrices, self.true_sigma_u, self.column_names, self.A0,
         self.n_lags, self.relations_python) = setup_german_simulation()
        self.true_simulation = SVARSimulation(coefs=self.true_matrices, sigma_u=self.true_sigma_u,
                                              names=self.column_names, A_0=self.A0)
        self.future_steps = FUTURE_STEPS
        self.past_steps = PAST_STEPS
        self.simulation_index_to_plot = 0
        self.intervention_dict: Dict[str, float] = {'Expertise': 5}
        self.epsilon_scale = np.ones(len(self.column_names))

        # Set plot title based on the intervention
        if self.column_name_for_plot == 'Credit Score':
            self.title = f'Forcing Intervention on {list(self.intervention_dict.keys())[0]}'
        else:
            self.title = f'Forcing Intervention with Hardness={self.force}'

    def run(self):
        """
        Run the simulation, forecast, and plotting processes.
        """
        # Generate training data
        train_data, _ = self.true_simulation.generate_data(steps=self.train_steps,
                                                           n_simulations=self.n_simulations, seed=4)

        # Fit VAR model to training data
        fitted_matrices, fitted_sigma_u = var_fit_with_r(train_data, n_lags=self.n_lags,
                                                         column_names=self.column_names)
        fitted_svar_simulation_r = SVARSimulation(fitted_matrices, fitted_sigma_u, self.column_names,
                                                  A_0=self.A0)

        # Generate data for simulation and forecasting
        total_data, _ = self.true_simulation.generate_data(steps=self.past_steps + self.future_steps,
                                                           n_simulations=self.n_simulation_test, seed=17)

        # Smooth the data for plotting
        total_data[self.simulation_index_to_plot] = sp.ndimage.convolve1d(total_data[self.simulation_index_to_plot],
                                                                          np.ones(5) / 5, axis=0)
        past_data = total_data[:, :self.past_steps, :]

        # Generate forecasts
        true_forecast = get_forecast(self.true_simulation, past_data, self.future_steps)
        true_forecast_cov = self.true_simulation.svar_process.forecast_cov(self.future_steps)

        # Apply forcing intervention to the true simulation
        true_forcing_simulation, true_forcing_single_effect = forcing_intervention(
            self.true_simulation,
            self.intervention_dict,
            self.column_names,
            self.epsilon_scale,
            self.force,
            self.future_steps,
            asymptotic=self.asymptotic
        )
        true_intervened_forecast = get_forecast(true_forcing_simulation, past_data, self.future_steps)
        true_intervened_forecast_cov = true_forcing_simulation.svar_process.forecast_cov(self.future_steps)

        # Predict the forecast using the fitted model
        _, predicted_forcing_single_effect = forcing_intervention(
            fitted_svar_simulation_r,
            self.intervention_dict,
            self.column_names,
            self.epsilon_scale,
            self.force,
            self.future_steps,
            asymptotic=self.asymptotic
        )

        # Clone effects for each simulation trajectory
        true_forcing_effects = clone_effects(self.n_simulation_test, true_forcing_single_effect)
        predicted_forcing_effects = clone_effects(self.n_simulation_test, predicted_forcing_single_effect)

        # Combine forecasted data with intervention effects
        true_forcing_forecast_and_effects, predicted_forcing_forecast_and_effects = forcing_forecasting(
            true_intervened_forecast,
            true_forcing_effects,
            predicted_forcing_effects
        )

        # Plot the results
        plot_ai_fi_intervention_forecast(
            past_data[self.simulation_index_to_plot],
            true_forecast[self.simulation_index_to_plot],
            true_forecast_cov,
            true_forcing_forecast_and_effects[self.simulation_index_to_plot],
            predicted_forcing_forecast_and_effects[self.simulation_index_to_plot],
            true_intervened_forecast_cov,
            additive=False,
            intervention_dict=self.intervention_dict,
            column_name=self.column_name_for_plot,
            column_index=self.column_index_for_plot,
            title=self.title,
            year_end=str(self.year_start + self.past_steps + self.future_steps)
        )


if __name__ == '__main__':
    # Instantiate and run the PlotForcing class
    plot_forcing_instance = PlotForcing()
    plot_forcing_instance.run()
