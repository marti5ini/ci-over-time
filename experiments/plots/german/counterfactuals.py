import os
import sys
import scipy as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiments.plots.plotting_functions import plot_cf_forecast2
from src.generation.setup_simulation import *
from experiments.plots.additive_forecast import get_effects
from src.CausalVAR.simulation import SVARSimulation
from src.CausalVAR.fitting import var_fit_with_r
from experiments.plots.config import YEAR_START_CF, FUTURE_STEPS, PAST_STEPS

column_names_german = {
    'Expertise': 0,
    'Responsibility': 1,
    'Loan Amount': 2,
    'Loan Duration': 3,
    'Income': 4,
    'Savings': 5,
    'Credit Score': 6
}

def retrospective_counterfactuals(test_factual, test_effects, predicted_effects):
    test_cf = test_factual + test_effects
    predicted_cf = test_factual + predicted_effects
    return test_cf, predicted_cf

# Function to update column name based on index
def update_column_name(index):
    for name, value in column_names_german.items():
        if value == index:
            return name
    return None

class PlotCounterfactuals:
    def __init__(self, dataset_name='german'):
        self.dataset_name = dataset_name
        self.threshold = 0
        self.train_steps = 480
        self.n_simulations = 2
        self.n_simulation_test = 5
        self.year_start = int(YEAR_START_CF)
        self.year_range_x_labels = 10
        (self.true_matrices, self.true_sigma_u, self.column_names,
         self.A0, self.n_lags, self.relations_python) = setup_german_simulation()
        self.future_steps = FUTURE_STEPS
        self.past_steps = PAST_STEPS
        self.simulation_index_to_plot = 0
        #self.column_index_for_plot = 0
        #self.column_name_for_plot = update_column_name(self.column_index_for_plot)
        self.intervention_dict = {'Expertise': 0.3}
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
                                                          n_simulations=self.n_simulation_test, seed=96)

        total_data[self.simulation_index_to_plot] = sp.ndimage.convolve1d(total_data[self.simulation_index_to_plot],
                                                                        np.ones(5) / 5, axis=0)
        past_data = total_data[:, :self.past_steps, :]

        # EFFECT ESTIMATION
        test_effects = get_effects(self.true_simulation, self.intervention_dict, self.column_names,
                                   self.future_steps,
                                   self.n_simulation_test)
        predicted_effects = get_effects(fitted_svar_simulation_r, self.intervention_dict,
                                        self.column_names, self.future_steps,
                                        self.n_simulation_test)

        test_factual = self.true_simulation.simulated_data[:, self.past_steps:, :]
        true_cf, predicted_cf = retrospective_counterfactuals(test_factual, test_effects, predicted_effects)


        for index_col in [0, 1, 2]:
            col_name = update_column_name(index_col)
            if col_name != list(self.intervention_dict.keys())[0]:
                self.title = f'Counterfactuals {col_name}, on {list(self.intervention_dict.keys())[0]}={list(self.intervention_dict.values())[0]}'
            else:
                self.title = f'Counterfactuals on {list(self.intervention_dict.keys())[0]} with Force={list(self.intervention_dict.values())[0]}'
            plot_cf_forecast2(past_data[self.simulation_index_to_plot],
                                         test_factual[self.simulation_index_to_plot],
                                         true_cf[self.simulation_index_to_plot],
                                         predicted_cf[self.simulation_index_to_plot],
                                         column_index=index_col,
                                         column_name=col_name,
                                         title=self.title,
                                         year_end=str(self.year_start + self.past_steps + self.future_steps))


if __name__ == '__main__':
    a = PlotCounterfactuals()
    a.run()
