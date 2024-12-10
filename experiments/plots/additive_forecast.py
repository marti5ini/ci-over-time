import os
import sys
import scipy as sp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiments.plots.plotting_functions import plot_ai_fi_intervention_forecast
from src.generation.setup_simulation import setup_german_simulation
from experiments.utils import get_forecast, additive_forecasting
from experiments.interventional_forecasting_additive import additive_intervention
from src.CausalVAR.simulation import SVARSimulation
from src.CausalVAR.fitting import var_fit_with_r
from experiments.plots.config import YEAR_START, FUTURE_STEPS, PAST_STEPS

def clone_effects(n_simul_test, effects):
    trajectory = np.zeros((n_simul_test, *effects.shape))

    for i in range(n_simul_test):
        trajectory[i] = effects

    return trajectory

def get_effects(process, interv_eff, names, steps_future, n_simul_test):
    effects = additive_intervention(process, interv_eff, names, steps_future)
    return clone_effects(n_simul_test, effects)


class PlotAdditive:
    def __init__(self, dataset_name='german'):
        self.dataset_name = dataset_name
        self.threshold = 0
        self.train_steps = 480
        self.n_simulations = 2 #2
        self.n_simulation_test = 20
        self.year_start = int(YEAR_START)
        self.year_range_x_labels = 10
        (self.true_matrices, self.true_sigma_u, self.column_names,
         self.A0, self.n_lags, self.relations_python) = setup_german_simulation()
        self.future_steps = FUTURE_STEPS
        self.past_steps = PAST_STEPS
        self.simulation_index_to_plot = 0
        self.column_index_for_plot = 0
        self.column_name_for_plot = 'Expertise'
        self.intervention_dict = {'Expertise': 0.3}
        if self.column_name_for_plot == 'Credit Score':
            self.title=f'Additive Intervention on {list(self.intervention_dict.keys())[0]}'
        else:
            self.title=f'Additive Intervention with Force={list(self.intervention_dict.values())[0]}'
        self.true_simulation = None

    def run(self):
        self.true_simulation = SVARSimulation(coefs=self.true_matrices, sigma_u=self.true_sigma_u,
                                              names=self.column_names, A_0=self.A0)

        train_data, _ = self.true_simulation.generate_data(steps=self.train_steps,
                                                           n_simulations=self.n_simulations, seed=2)

        fitted_matrices, fitted_sigma_u = var_fit_with_r(train_data, n_lags=self.n_lags,
                                                         column_names=self.column_names)

        fitted_svar_simulation_r = SVARSimulation(fitted_matrices, fitted_sigma_u, self.column_names,
                                                  A_0=self.A0)
        fitted_data_r, _ = fitted_svar_simulation_r.generate_data(n_simulations=self.n_simulation_test)

        total_data, _ = self.true_simulation.generate_data(steps=self.past_steps + self.future_steps,
                                                           n_simulations=self.n_simulation_test, seed=17)

        total_data[self.simulation_index_to_plot] = sp.ndimage.convolve1d(total_data[self.simulation_index_to_plot],
                                                                          np.ones(5) / 5, axis=0)
        past_data = total_data[:, :self.past_steps, :]

        true_forecast = get_forecast(self.true_simulation, past_data, self.future_steps)
        true_forecast_cov = self.true_simulation.svar_process.forecast_cov(self.future_steps)
        predicted_forecast = get_forecast(fitted_svar_simulation_r, past_data, self.future_steps)
        predicted_forecast_cov = fitted_svar_simulation_r.svar_process.forecast_cov(self.future_steps)

        # EFFECT ESTIMATION
        test_effects = get_effects(self.true_simulation, self.intervention_dict, self.column_names,
                                   self.future_steps,
                                   self.n_simulation_test)
        predicted_effects = get_effects(fitted_svar_simulation_r, self.intervention_dict,
                                        self.column_names, self.future_steps,
                                        self.n_simulation_test)

        true_additive, predicted_additive = additive_forecasting(true_forecast, predicted_forecast,
                                                                 test_effects, predicted_effects)

        plot_ai_fi_intervention_forecast(past_data[self.simulation_index_to_plot],
                                         true_forecast[self.simulation_index_to_plot],
                                         true_forecast_cov,
                                         true_additive[self.simulation_index_to_plot],
                                         predicted_additive[self.simulation_index_to_plot],
                                         predicted_forecast_cov,
                                         intervention_dict=self.intervention_dict,
                                         column_index=self.column_index_for_plot,
                                         column_name=self.column_name_for_plot,
                                         additive=True,
                                         title=self.title,
                                         year_end=str(self.year_start + self.past_steps + self.future_steps))


if __name__ == '__main__':
    # fire.Fire(PlotAdditive)
    a = PlotAdditive()
    a.run()
