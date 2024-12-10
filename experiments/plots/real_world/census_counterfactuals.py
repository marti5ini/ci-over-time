import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR, VARProcess

from experiments.plots.config import LINEWIDTH, FONTSIZE_TITLE, FONTSIZE_TICKS, GRID_ENABLED, GRID_COLOR, GRID_STYLE, \
    GRID_LINEWIDTH
from src.CausalVAR.fitting_var_bounded import estimate_bounded

FIGSIZE = (9, 6)

def calculate_mean_age(data):
    age_weights = np.array([7, 40, 80])
    populations = data[..., -3:]
    return np.sum(populations * age_weights, axis=-1) / np.sum(populations, axis=-1)


def scale_data_by_pop(data):
    # Reshape data to (n_states, n_years, n_features)
    n_states = len(np.unique(data[:, 0]))  # Assuming first column is state identifier
    n_years = len(data) // n_states
    n_features = data.shape[1] - 1  # Subtract 1 for state identifier column
    reshaped_data = data[:, 1:].reshape(n_states, n_years, n_features)

    # Calculate sum of last three columns for the first year
    sum_last_three_cols = np.sum(reshaped_data[:, 0, -3:], axis=1)
    sum_last_three_cols = sum_last_three_cols[:, np.newaxis, np.newaxis]

    # Scale the data
    scaled_data = reshaped_data / sum_last_three_cols
    return scaled_data
def mu_to_v(mapping, svar_process, mask, names, intervention_effects):
    force_vector = np.zeros(len(names)).reshape(-1, 1)
    for col, value in intervention_effects.items():
        interventional_vector = np.zeros(len(names)).reshape(-1, 1)
        index = mapping[col]
        interventional_vector[index] = value
        force_vector += (svar_process.A_solve @ svar_process._char_mat @ interventional_vector) * mask
    return force_vector

def varprocess_additive_intervention(process, intervention_effects, names,
                                     intervention_steps, asymptotic=False):
    mask = np.zeros(len(names)).reshape(-1, 1)
    mapping = {names[i]: i for i in range(len(names))}

    for col, value in intervention_effects.items():
        index = mapping[col]
        mask[index] = 1

    interventional_vector = np.zeros(len(names)).reshape(-1, 1)
    if asymptotic:
        interventional_vector = mu_to_v(mapping, process, mask, names, intervention_effects)
    else:
        for col, value in intervention_effects.items():
            index = mapping[col]
            interventional_vector[index] = value

    size = process.coefs.shape[1]
    interventional_vector = np.array(interventional_vector).reshape(-1, 1)
    effects_vector = np.zeros(shape=(0, size))
    effects = np.zeros(size).reshape(-1, 1)
    response_matrices = process.ma_rep(maxn=intervention_steps)
    for i in range(intervention_steps):
        if i <= np.inf:
            effects += response_matrices[i] @ interventional_vector
        else:
            effects = np.sum(response_matrices[i - np.inf:i + 1], axis=0) @ interventional_vector
        effects_vector = np.vstack([effects_vector, effects.T])
    return effects_vector



def plot_counterfactual(country, mean_age_total, mean_age_int, intervention_year, years, states):
    plt.figure(figsize=FIGSIZE)
    country_index = np.where(states == country)[0][0]
    # Plot total data (past and future combined)
    plt.plot(years, mean_age_total[country_index], color='#0072B2', linestyle='-',
             linewidth=LINEWIDTH, label=country)
    # Plot counterfactual forecast
    future_years = years[years >= intervention_year]
    plt.plot(future_years, mean_age_int[country_index], linewidth=LINEWIDTH,
             linestyle='dotted', color='darkmagenta')
    # Add intervention line (darkgray)
    plt.axvline(x=intervention_year, color='darkgray', linestyle='-', linewidth=GRID_LINEWIDTH + 4)
    # Add 2024 line (black dashed)
    plt.axvline(x=2023, color='black', linestyle='--', linewidth=GRID_LINEWIDTH + 4)
    plt.xlim(min(years), 2025)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICKS)
    plt.tick_params(axis='x', which='major', pad=15)
    plt.tick_params(axis='x', which='major', pad=10)
    plt.xlabel("Year", fontsize=FONTSIZE_TITLE - 2)
    plt.ylabel('Avg. Age', fontsize=FONTSIZE_TITLE - 2)
    """if country == 'Singapore':
        plt.yticks([35, 37, 40], fontsize=FONTSIZE_TICKS)"""
    # Set up x-axis ticks and labels
    year_ticks = list(range(min(years), max(years), 10))
    year_ticks.append(2023)
    year_ticks.remove(2021)
    plt.xticks(year_ticks)
    if GRID_ENABLED:
        plt.grid(axis='x', color=GRID_COLOR, linestyle=GRID_STYLE, linewidth=GRID_LINEWIDTH)
    # Add legend with only the country name
    plt.legend(fontsize=34, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(f'/Users/martina/Desktop/ci_over_time/overleaf/get_plots/pdfs/Counterfactual_{country}.pdf', dpi=200)
    plt.show()


# Load and prepare data
tot_df = pd.read_csv('/Users/martina/Desktop/ci_over_time/real_world/census_shifted.csv')
tot_df = tot_df[~tot_df.Name.isin(['Qatar', 'Seychelles', 'Turks and Caicos Islands', 'Kuwait'])]
tot_df.reset_index(inplace=True, drop=True)

# Convert 'Year' to datetime and then extract the year
tot_df['Year'] = pd.to_datetime(tot_df['Year']).dt.year

years = tot_df['Year'].unique()
years = np.append(years, 2023)
intervention_year = 2011
past_steps = sum(years < intervention_year)
intervention_steps = sum(years >= intervention_year)

df = tot_df.drop(["Year"], axis=1)
n_states = df['Name'].nunique()
states = df['Name'].unique()
cols = ['Births', 'Deaths', 'Migrations', 'Population 0-14', 'Population 15-64', 'Population 65+']
df = df[['Name'] + cols]

# Reshape and scale data
df_array = df.values
df_array = scale_data_by_pop(df_array)

# Fit VAR model
res_states = {}
for state in range(n_states):
    model = VAR(df_array[state, :, :])
    res = estimate_bounded(model=model, lags=1, trend="c")
    res_states[states[state]] = res

coefs_array = np.array([res_states[state].coefs for state in states])
coefs_mean = np.mean(coefs_array, axis=0)

# Set up intervention
common_process = VARProcess(coefs=coefs_mean, sigma_u=np.eye(6), names=cols, coefs_exog=np.zeros((1, 6)))
intervention = {'Births': 0.004}

# Calculate effects
effects = varprocess_additive_intervention(common_process, intervention, names=cols, intervention_steps=intervention_steps)

# Generate forecasts and counterfactuals
forecasts = np.zeros(shape=(n_states, 1, len(cols)))
for i, state in enumerate(states):
    forecasts[i, :, :] = res_states[state].forecast(df_array[i, :, :], steps=1)

df_array = np.concatenate([df_array, forecasts], axis=1)

# Calculate mean age
mean_age_past = calculate_mean_age(df_array[:, :past_steps, :])
mean_age_future = calculate_mean_age(df_array[:, past_steps:, :])
int_forecasts = df_array[:, past_steps:, :] + effects
mean_age_int = calculate_mean_age(int_forecasts)

# Plot for selected countries
selected_countries = ['Singapore', 'Spain', 'Germany']
mean_age_total = np.concatenate((mean_age_past, mean_age_future), axis=1)

# Update the call to plot_counterfactual:
for country in selected_countries:
    plot_counterfactual(country, mean_age_total, mean_age_int, intervention_year, years, states)

"""germany_index = np.where(states == 'Germany')[0][0]
plt.plot(df_array[germany_index, :, 0])
plt.show()"""


print(f"Past steps: {past_steps}")
print(f"Intervention steps: {intervention_steps}")
print(f"Years range: {min(years)} to {max(years)}")