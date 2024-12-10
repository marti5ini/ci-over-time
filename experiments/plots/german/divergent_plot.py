import matplotlib.pyplot as plt

from src.CausalVAR.simulation import SVARSimulation
from src.generation.setup_simulation import *
from experiments.plots.additive_forecast import get_effects

dataset_name = 'german'
FONTSIZE_LABELS = 50
LINEWIDTH = 6
LINEWIDTH_VERTICAL = 6
FIGSIZE = (11, 7)

# Define parameters
column = 6
past_steps = 50
maximum_time = 100  # Changed to match the first plot
n_simulations = 100
threshold = 80  # Changed to match the first plot
intervention = {'Expertise': 0.28}  # Changed to match the first plot
future_steps = 70  # Changed to match the first plot
seed = 2

# Time steps to years conversion
time_per_year = 50
start_year = 2024
time_to_year = lambda t: start_year + (t - past_steps) / time_per_year

true_matrices, true_sigma_u, column_names, A0, n_lags, relations_python = setup_german_simulation()
true_simulation = SVARSimulation(coefs=true_matrices,sigma_u=true_sigma_u,names=column_names,A_0=A0)
# Generate data
X, _ = true_simulation.generate_data(n_simulations=n_simulations, steps=past_steps + future_steps,seed=seed)
effects = get_effects(true_simulation, intervention, column_names, future_steps, n_simulations)
effects = np.concatenate((np.zeros((n_simulations, past_steps, X.shape[2])), effects), axis=1)
X_int = X + effects

# Define the window size (h steps before and after the intervention)
h = 12  # Adjust this to control how close to the intervention time the comparison is made

# Time window for comparison around the intervention time
window_start = past_steps - h
window_end = past_steps + h

# Extract the values within the window around the intervention time
window_values = X_int[:, window_start:window_end, column]

# Calculate the mean values before and after the intervention within the window
pre_intervention_means = np.mean(X_int[:, past_steps-h:past_steps, column], axis=1)
post_intervention_means = np.mean(X_int[:, past_steps:past_steps+h, column], axis=1)

# Calculate the difference between pre and post means within the window
differences_within_window = post_intervention_means - pre_intervention_means

# Find the two trajectories that are most similar before the intervention but diverge after it
similarity_threshold = 0.5  # Adjust this value based on how similar you want the trajectories to be

# Mask for similar pre-intervention means
values_at_intervention = X_int[:, past_steps, column]
similar_indices = np.where(np.abs(values_at_intervention - values_at_intervention[:, None]) < similarity_threshold)

# Find pair of indices with the most divergent post-intervention behavior
max_divergence = 0
selected_indices = None

for i, j in zip(similar_indices[0], similar_indices[1]):
    if i != j:  # Ensure not comparing the same trajectory
        divergence = np.abs(differences_within_window[i] - differences_within_window[j])
        if divergence > max_divergence:
            max_divergence = divergence
            selected_indices = (i, j)

# Extract the two selected trajectories
traj_1 = X_int[selected_indices[0], :, column]
traj_2 = X_int[selected_indices[1], :, column]
traj_1_or = X[selected_indices[0], :, column]
traj_2_or = X[selected_indices[1], :, column]

# Prepare the plot focusing on these two trajectories
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=200)

# Plot the selected trajectories
ax.plot(range(past_steps + 1), traj_1[:past_steps + 1], color='#DAA520', linewidth=LINEWIDTH, label='Trajectory 1')
ax.plot(range(past_steps + 1), traj_2[:past_steps + 1], color='darkmagenta', linewidth=LINEWIDTH, label='Trajectory 2')

# Future (intervened)
ax.plot(range(past_steps, X_int.shape[1]), traj_1[past_steps:], color='#DAA520', linewidth=LINEWIDTH, linestyle=':')
ax.plot(range(past_steps, X_int.shape[1]), traj_2[past_steps:], color='darkmagenta', linewidth=LINEWIDTH, linestyle=':')

# Future (original)
ax.plot(range(past_steps, X_int.shape[1]), traj_1_or[past_steps:], color='#DAA520', linewidth=LINEWIDTH, linestyle='--')
ax.plot(range(past_steps, X_int.shape[1]), traj_2_or[past_steps:], color='darkmagenta', linewidth=LINEWIDTH, linestyle='--')

# Add a horizontal line at the threshold
ax.axhline(y=threshold, color='darkgreen', linestyle='--', linewidth=LINEWIDTH_VERTICAL-1, label='Acceptance Threshold', alpha=0.7)

# Add a vertical line at past_steps (intervention time)
ax.axvline(x=past_steps, color='darkgray', linestyle='-', linewidth=LINEWIDTH_VERTICAL, label='Intervention Time')

# Add a vertical line at maximum_time
ax.axvline(x=maximum_time + 0.05, color='black', linestyle='dotted', linewidth=LINEWIDTH_VERTICAL+2, label='Maximum Time', alpha=0.7)

# Highlight the divergence area around the intervention time
#ax.fill_between(range(window_start, window_end), traj_1[window_start:window_end], traj_2[window_start:window_end], color='#CC79A7', alpha=0.4, label='Divergence Area')

# Plot crossing points for both trajectories
for traj, label in zip([traj_1, traj_2], ['Trajectory 1', 'Trajectory 2']):
    crossing_indices = np.where(traj >= threshold)[0]
    if len(crossing_indices) > 0:
        crossing_time = crossing_indices[0]
        if crossing_time<=maximum_time:
            ax.scatter(crossing_time, threshold, marker='+', s=1000, color='darkgreen', linewidths=10, alpha=1, label=f'{label} crosses threshold', zorder=100)
        else:
            ax.scatter(maximum_time, traj[maximum_time], marker='x', s=700, linewidths=8, color='red', label=f'{label} does not cross', zorder=100)
    else:
        ax.scatter(maximum_time, traj[maximum_time], marker='x', s=400, markersize=7, color='red', label=f'{label} does not cross', zorder=100)

# Plot crossing points for both trajectories
for traj, color, label in zip([traj_1, traj_2], ['brown', 'green'], ['Trajectory 1', 'Trajectory 2']):
    crossing_indices = np.where(traj >= threshold)[0]
    if len(crossing_indices) > 0:
        crossing_time = crossing_indices[0]


# Convert x-axis from time steps to years
time_steps = np.arange(X_int.shape[1])
year_labels = [time_to_year(t) for t in time_steps]
ax.set_xticks(time_steps[::time_per_year])
ax.set_xticklabels([f'{int(year)}' for year in year_labels[::time_per_year]], fontsize=FONTSIZE_LABELS - 5)
import matplotlib.ticker as ticker
def y_formatter(x, pos):
    return f'{x/200:.0f}'
ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
ax.tick_params(axis='y', labelsize=FONTSIZE_LABELS - 5)

ax.set_xlabel('Year', fontsize=FONTSIZE_LABELS)
ax.set_ylabel(' ', fontsize=FONTSIZE_LABELS)

# Customize tick parameters
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_LABELS - 9, width=2, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=3)
ax.tick_params(axis='x', which='major', pad=15)

# Plot crossing points for both trajectories
for traj, color, label in zip([traj_1, traj_2], ['brown', 'green'], ['Trajectory 1', 'Trajectory 2']):
    crossing_indices = np.where(traj >= threshold)[0]
    if len(crossing_indices) > 0:
        crossing_time = crossing_indices[0]

handles, labels = plt.gca().get_legend_handles_labels()
country_handles = [handle for handle, label in zip(handles, labels) if label in ['Trajectory 1', 'Trajectory 2']]
country_labels = [label for label in labels if label in ['Trajectory 1', 'Trajectory 2']]

legend_linewidth = 15
legend_handles = []
for handle in country_handles:
    label = handle.get_label()
    new_handle = plt.Line2D([], [], color=handle.get_color(), linewidth=legend_linewidth,
                                label=label)
    legend_handles.append(new_handle)
    country_labels.append(label)


applicant1_index = country_labels.index('Trajectory 1')
applicant2_index = country_labels.index('Trajectory 2')

ordered_handles = [
    legend_handles[applicant1_index],
    legend_handles[applicant2_index]
]
ordered_labels = ['Applicant 1', 'Applicant 2']

FONTSIZE_LEGEND = 40
plt.subplots_adjust(top=0.9, left=0.1)
plt.legend(handles=ordered_handles, labels=ordered_labels, ncol=1, loc='upper left', frameon=True,
           fontsize=FONTSIZE_LEGEND-6, bbox_to_anchor=(0.001, 1.02), facecolor=(1, 1, 1, 1), edgecolor='none')
plt.tight_layout()
plt.savefig('../pdfs/german_divergent.pdf', dpi=200)
plt.show()