import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR, VARProcess
from src.CausalVAR.fitting_var_bounded import estimate_bounded
from utils import scale_data_by_pop, varprocess_additive_intervention, softplus

# Load and preprocess the census data
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

# Reshape the data and scale it by population
df_array = df.to_numpy().reshape(n_states, -1, 6)
df_array = scale_data_by_pop(df_array)

# Fit VAR models for each state
res_states = {}
for state in range(n_states):
    model = VAR(df_array[state, :, :])
    lag_order = 1
    res = estimate_bounded(model=model, lags=1, trend="c")
    fitted_values = res.fittedvalues
    res_states[states[state]] = res

# Calculate the mean coefficients across states
coefs_array = np.array([res_states[state].coefs for state in states])
coefs_mean = np.mean(coefs_array, axis=(0))

# Create a common VAR process with the mean coefficients
common_process = VARProcess(coefs=coefs_mean, sigma_u=np.eye(6), names=list(df.columns), coefs_exog=np.zeros((1, 6)))
intervention_steps = 30

# Generate forecasts for each state
forecasts = np.zeros(shape=(n_states, intervention_steps, n_columns))
for i, state in enumerate(states):
    forecasts[i, :, :] = res_states[state].forecast(df_array[i, :, :], steps=intervention_steps)
    forecast_cov = res_states[state].forecast_cov(steps=intervention_steps)

# Define intervention effects and apply them to the forecasts
intervention = {'Migrations': 0.04}
effects = varprocess_additive_intervention(common_process, intervention, intervention_steps=intervention_steps, names=list(df.columns))
int_forecasts = forecasts + effects
past_steps = 31

# Apply softplus function to the birth data
smooth = True
beta = 100
if smooth:
    births_past = softplus(df_array[:, :, 0], beta) * 100
    births_future = softplus(forecasts[:, :, 0], beta) * 100
    births_int = softplus(int_forecasts[:, :, 0], beta) * 100

# Calculate mean birth trajectories
mean_births_past = np.mean(births_past, axis=0)
mean_births_future = np.mean(births_future, axis=0)
mean_births_int = np.mean(births_int, axis=0)

# Find specific country indices
italy_index, italy_name = np.where(states == 'Italy')[0][0], 'Italy'
other_index, name_index = np.where(states == 'Panama')[0][0], 'Panama'

# Plot the birth trajectories
FONTSIZE_LABELS = 50
LINEWIDTH = 2
LINEWIDTH_VERTICAL = 6
FIGSIZE = (11, 7)
FONTSIZE_LEGEND = 40

plt.figure(figsize=FIGSIZE)
traj_alpha = 0.12

for i in range(n_states):
    if i != other_index and i != italy_index:
        plt.plot(range(past_steps + 1), births_past[i, :], alpha=traj_alpha, linewidth=LINEWIDTH)
        plt.plot(range(past_steps, past_steps + intervention_steps + 1),
                 np.concatenate(([births_past[i, -1]], births_future[i, :])), alpha=traj_alpha,
                 linestyle='--', linewidth=LINEWIDTH)
        plt.plot(range(past_steps, past_steps + intervention_steps + 1),
                 np.concatenate(([births_past[i, -1]], births_int[i, :])), alpha=traj_alpha + 0.20, linestyle=':', linewidth=LINEWIDTH)

# Highlight specific countries with distinct colors
for country_index, country_name, color in [(other_index, name_index, 'darkmagenta'), (italy_index, italy_name, '#DAA520')]:
    plt.plot(range(past_steps + 1), births_past[country_index, :], color=color, linewidth=6, label=country_name)
    plt.plot(range(past_steps, past_steps + intervention_steps + 1),
             np.concatenate(([births_past[country_index, -1]], births_future[country_index, :])), color=color,
             linestyle='--', linewidth=6)
    plt.plot(range(past_steps, past_steps + intervention_steps + 1),
             np.concatenate(([births_past[country_index, -1]], births_int[country_index, :])), color=color,
             linestyle=':', linewidth=8)

# Plot the mean trajectories
plt.plot(range(past_steps + 1), mean_births_past, color='#0072B2', alpha=1, linewidth=LINEWIDTH+8, label='Mean')
plt.plot(range(past_steps, past_steps + intervention_steps + 1),
         np.concatenate(([mean_births_past[-1]], mean_births_future)), color='#0072B2', alpha=1,
         linestyle='--', linewidth=LINEWIDTH+7.5)
plt.plot(range(past_steps, past_steps + intervention_steps + 1),
         np.concatenate(([mean_births_past[-1]], mean_births_int)), color='#0072B2', alpha=1, linewidth=LINEWIDTH+10,
         label='Mean Intervention', linestyle=':')

# Add vertical line at intervention time
plt.axvline(x=past_steps, color='darkgray', linestyle='-', linewidth=LINEWIDTH_VERTICAL, label='Intervention Time')

# Set x-ticks and labels
total_steps = past_steps + intervention_steps
years = np.arange(2023 - past_steps, 2023 + intervention_steps + 1)
tick_years_forward = np.arange(2023, 2023 + (20 * ((intervention_steps // 20) + 1)), 20)
tick_years_backward = np.arange(2023, 2023 - (20 * ((past_steps // 20) + 1)), -20)
tick_years = np.concatenate((tick_years_backward, tick_years_forward))
tick_years = [year for year in tick_years if year in years]
tick_indices = [np.where(years == year)[0][0] for year in tick_years]

plt.xticks(ticks=tick_indices, labels=tick_years)
plt.tick_params(axis='both', which='major', labelsize=FONTSIZE_LABELS - 9)
plt.tick_params(axis='x', which='major', pad=15)
plt.tick_params(axis='x', which='major', pad=10)
plt.xlim(0, total_steps-1)

# Add labels and grid
plt.xlabel('Year', fontsize=FONTSIZE_LABELS)
plt.ylabel('Births (%)', fontsize=FONTSIZE_LABELS)
plt.grid(False)
plt.gca().xaxis.grid(True, linestyle='dotted', alpha=0.7, linewidth=1.5, zorder=200)
plt.gca().set_axisbelow(True)

# Customize the legend
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

japan_index = country_labels.index('Panama')
chile_index = country_labels.index('Italy')
mean_index = country_labels.index('Mean')

ordered_handles = [
    legend_handles[japan_index],
    legend_handles[mean_index],
    legend_handles[chile_index]
]
ordered_labels = ['Panama', 'Mean', 'Italy']

plt.subplots_adjust(top=0.9, left=0.1)
plt.legend(handles=ordered_handles, labels=ordered_labels, ncol=1, loc='upper left', frameon=False,
           fontsize=FONTSIZE_LEGEND-6, bbox_to_anchor=(0.001, 1.02))

plt.tight_layout()
plt.savefig('../pdfs/census_births.pdf', dpi=200)
plt.show()