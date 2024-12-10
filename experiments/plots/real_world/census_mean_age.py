import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR, VARProcess
from src.CausalVAR.fitting_var_bounded import estimate_bounded
from utils import scale_data_by_pop, varprocess_additive_intervention


# RETRIEVE DATA AND SPLIT IT
tot_df = pd.read_csv('../../../data/census_shifted.csv')
tot_df = tot_df[~tot_df.Name.isin(['Qatar','Seychelles','Turks and Caicos Islands','Kuwait'])]
tot_df.reset_index(inplace=True, drop=True)
df_year = tot_df["Year"]
df = tot_df.drop(["Year"], axis=1)
n_states = df['Name'].nunique()
states = df['Name'].unique()
states_names = df['Name'].unique()

rows = df['Name'].value_counts().sort_index()
df.drop(['Name'], axis=1, inplace=True)
cols = ['Births', 'Deaths', 'Migrations', 'Population 0-14', 'Population 15-64', 'Population 65+']
df = df[cols]
n_columns = len(cols)

# FITTING PHASE
df_array = df.to_numpy().reshape(n_states, -1, 6)

res_states = {}

def calculate_mean_age(data):
    # Weights for the age groups: 0-14, 15-64, 65+
    age_weights = np.array([7, 40, 80])

    # Extract the population columns (last three columns)
    populations = data[:, :, -3:]

    # Calculate the weighted mean
    mean_age = np.sum(populations * age_weights, axis=2) / np.sum(populations, axis=2)

    return mean_age


df_array = scale_data_by_pop(df_array)

for state in range(n_states):
    model = VAR(df_array[state, :, :])
    lag_order = 1
    res = estimate_bounded(model=model, lags=1, trend="c")
    fitted_values = res.fittedvalues
    res_states[states[state]] = res

coefs_array = np.array([res_states[state].coefs for state in states])
coefs_mean = np.mean(coefs_array, axis=(0))

# INTERVENTION EFFECTS
common_process = VARProcess(coefs=coefs_mean,sigma_u=np.eye(6), names=list(df.columns),coefs_exog=np.zeros((1, 6)))
intervention = {'Births': 0.004}
intervention_steps = 30
past_steps = 31
effects = varprocess_additive_intervention(common_process,intervention,intervention_steps=intervention_steps,names=list(df.columns))

# ESTIMATING SINGLE MODELS FORECASTS
forecasts = np.zeros(shape=(n_states, intervention_steps, n_columns))
for i,state in enumerate(states):
    #res_states[state].coefs = coefs_mean
    forecasts[i,:,:] = res_states[state].forecast(df_array[i, :, :], steps=intervention_steps)
    forecast_cov = res_states[state].forecast_cov(steps=intervention_steps)

int_forecasts = forecasts + effects

mean_age_past = calculate_mean_age(df_array)
mean_age_future = calculate_mean_age(forecasts)
mean_age_int = calculate_mean_age(int_forecasts)

mean_traj_past = np.mean(mean_age_past, axis=0)
mean_traj_future = np.mean(mean_age_future, axis=0)
mean_traj_int = np.mean(mean_age_int, axis=0)

italy_index, italy_name = np.where(states == 'Japan')[0][0], 'Japan'
other_index, name_index = np.where(states == 'Chile')[0][0], 'Chile'

# BIRTHS -> MEAN AGE

# Calculate the total number of steps
total_steps = past_steps + intervention_steps

# Create an array of years corresponding to each time step
years = np.arange(2023 - past_steps, 2023 + intervention_steps + 1)

# Generate tick years: starting from 2023, every 5 years forward and backward
tick_years_forward = np.arange(2023, 2023 + (20 * ((intervention_steps // 20) + 1)), 20)
tick_years_backward = np.arange(2023, 2023 - (20 * ((past_steps // 20) + 1)), -20)

# Combine forward and backward tick years
tick_years = np.concatenate((tick_years_backward, tick_years_forward))

# Filter out tick_years that are not in the years array
tick_years = [year for year in tick_years if year in years]

# Find the corresponding indices for the valid tick years
tick_indices = [np.where(years == year)[0][0] for year in tick_years]

FONTSIZE_LABELS = 50
LINEWIDTH = 2
LINEWIDTH_VERTICAL = 6
FIGSIZE = (11, 7)
FONTSIZE_LEGEND = 40

plt.figure(figsize=FIGSIZE)
traj_alpha = 0.12

# Plotting each state's trajectory
for i in range(n_states):
    if i != other_index and i != italy_index:
        past_data = mean_age_past[i, :].reshape(1, -1)
        future_data = mean_age_future[i, :].reshape(1, -1)
        int_future_data = mean_age_int[i, :].reshape(1, -1)

        past_and_future = np.hstack([past_data, future_data[:, :1]]).flatten()
        future_extended = future_data.flatten()
        int_future_extended = int_future_data.flatten()

        plt.plot(past_and_future, alpha=traj_alpha, linewidth=LINEWIDTH)
        plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(future_extended)),
                 future_extended, alpha=traj_alpha, linestyle='--', linewidth=LINEWIDTH)
        plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(int_future_extended)),
                 int_future_extended, alpha=traj_alpha + 0.20, linestyle=':', linewidth=LINEWIDTH)

# Highlighting specific countries with distinct colors
for country_index, country_name, color in [(italy_index, italy_name, 'darkmagenta'), (other_index, name_index, '#DAA520')]:
    past_data = mean_age_past[country_index, :].reshape(1, -1)
    future_data = mean_age_future[country_index, :].reshape(1, -1)
    int_future_data = mean_age_int[country_index, :].reshape(1, -1)

    past_and_future = np.hstack([past_data, future_data[:, :1]]).flatten()
    future_extended = future_data.flatten()
    int_future_extended = int_future_data.flatten()

    plt.plot(past_and_future, color=color, linewidth=6, label=country_name)
    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(future_extended)),
             future_extended, color=color, linestyle='--', linewidth=6)
    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(int_future_extended)),
             int_future_extended, color=color, linestyle=':', linewidth=8)

# Plotting the mean trajectories
past_data = mean_traj_past.reshape(1, -1)
future_data = mean_traj_future.reshape(1, -1)
int_future_data = mean_traj_int.reshape(1, -1)

past_and_future = np.hstack([past_data, future_data[:, :1]]).flatten()
future_extended = future_data.flatten()
int_future_extended = int_future_data.flatten()

plt.plot(past_and_future, color='#0072B2', alpha=1, linewidth=LINEWIDTH+8, label='Mean')
plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(future_extended)),
         future_extended, color='#0072B2', alpha=1, linestyle='--', linewidth=LINEWIDTH+7.5)
plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(int_future_extended)),
         int_future_extended, color='#0072B2', alpha=1, linewidth=LINEWIDTH+10, label='Mean Intervention', linestyle=':')


plt.axvline(x=past_steps+1, color='darkgray', linestyle='-', linewidth=LINEWIDTH_VERTICAL, label='Intervention Time')

# Set the x-ticks and labels
plt.xticks(ticks=tick_indices, labels=tick_years)
plt.tick_params(axis='both', which='major', labelsize=FONTSIZE_LABELS - 9)
plt.tick_params(axis='x', which='major', pad=15)
plt.tick_params(axis='x', which='major', pad=10)
plt.xlim(0, total_steps-1)

# Add labels for the axes
plt.xlabel('Year', fontsize=FONTSIZE_LABELS)
plt.ylabel('Avg. Age', fontsize=FONTSIZE_LABELS)

# Optional: Add a grid for better readability
ax = plt.gca()
plt.grid(False)
plt.gca().xaxis.grid(True, linestyle='dotted', alpha=0.7, linewidth=1.5, zorder=200)
plt.gca().set_axisbelow(True)


handles, labels = plt.gca().get_legend_handles_labels()
country_handles = [handle for handle, label in zip(handles, labels) if label in [italy_name, name_index, 'Mean']]
country_labels = [label for label in labels if label in [italy_name, name_index, 'Mean']]


legend_linewidth = 15
legend_handles = []
for handle in country_handles:
    label = handle.get_label()
    if label == 'Mean':
        new_handle = plt.Line2D([], [], color=handle.get_color(), linewidth=legend_linewidth,
                                label=label)
    else:
        new_handle = plt.Line2D([], [], color=handle.get_color(), linewidth=legend_linewidth,
                                alpha=0.6, label=label)
    legend_handles.append(new_handle)
    country_labels.append(label)

# Riordiniamo gli handle e le etichette per mettere 'mean' tra Japan e Chile
japan_index = country_labels.index('Japan')
chile_index = country_labels.index('Chile')
mean_index = country_labels.index('Mean')

ordered_handles = [
    legend_handles[japan_index],
    legend_handles[mean_index],
    legend_handles[chile_index]
]
ordered_labels = ['Japan', 'Mean', 'Chile']

plt.subplots_adjust(top=0.9, left=0.1)
plt.legend(handles=ordered_handles, labels=ordered_labels, ncol=1, loc='upper left', frameon=False,
           fontsize=FONTSIZE_LEGEND-6, bbox_to_anchor=(0.001, 1.02))

plt.tight_layout()
plt.savefig('../pdfs/census_mean_age.pdf', dpi=200)
plt.show()