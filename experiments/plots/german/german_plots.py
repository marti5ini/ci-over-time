import matplotlib.pyplot as plt

from experiments.plots.additive_forecast import get_effects
from src.generation.setup_simulation import *
from src.CausalVAR.simulation import SVARSimulation
from experiments.plots.plotting_functions import *

dataset_name = 'german'
FONTSIZE_LABELS = 50
LINEWIDTH = 2
LINEWIDTH_VERTICAL = 6
FIGSIZE = (11, 7)
acceptance_color = 'darkgreen' #15b01a


column = 6
past_steps = 50
maximum_time = 140  # Maximum time to consider
n_simulations = 100
threshold = 200
intervention = {'Expertise': 0.38}
future_steps = 100
seed = 2

# Time steps to years conversion
time_per_year = 50
start_year = 2024
time_to_year = lambda t: start_year + (t - past_steps) / time_per_year

# Generate data
true_matrices, true_sigma_u, column_names, A0, n_lags, relations_python = setup_german_simulation()
true_simulation = SVARSimulation(coefs=true_matrices,sigma_u=true_sigma_u,names=column_names,A_0=A0)
X, _ = true_simulation.generate_data(n_simulations=n_simulations, steps=past_steps + future_steps, seed=seed)
effects = get_effects(true_simulation, intervention, column_names, future_steps, n_simulations)
effects = np.concatenate((np.zeros((n_simulations, past_steps, X.shape[2])), effects), axis=1)
X_int = X + effects

threshold = 2 * X[:, :, column].std()

# Calculate the mean and standard deviation across the trajectories
mean_trajectory = np.mean(X_int[:, :, column], axis=0)
std_trajectory = 2 * np.std(X_int[:, :, column], axis=0)

# Prepare the plot
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=200)
ax.set_axisbelow(True)
ax.grid(False)
ax.xaxis.grid(True, linestyle='dotted', alpha=0.7, linewidth=2.5, zorder=200)
# Number of gradient layers
n_shades = 40
alpha_multiplier = 0.05

# Plot the KDEs with gradient alpha effect decreasing as 1/n^2
for i in range(n_shades):
    alpha = alpha_multiplier / (i + 1) ** 1.1  # Gradient alpha effect
    lower_bound = mean_trajectory - (i + 1) / n_shades * std_trajectory
    upper_bound = mean_trajectory + (i + 1) / n_shades * std_trajectory
    ax.fill_between(
        range(X_int.shape[1]),
        lower_bound,
        upper_bound,
        color='orange',
        alpha=alpha
    )


# Plot each trajectory
for i in range(n_simulations):
    ax.plot(range(X_int.shape[1]), X_int[i, :, column], alpha=0.08, linewidth=LINEWIDTH)

ax.set_xlim(0, X_int.shape[1] - 1)

# Plot the mean trajectory
mean_trajectory_orig = np.mean(X[:, :, column], axis=0)
ax.plot(range(past_steps + 1), mean_trajectory[:past_steps + 1], label='Mean', color='#0072B2', linewidth=LINEWIDTH_VERTICAL+4, linestyle='-')
ax.plot(range(past_steps, X_int.shape[1]), mean_trajectory[past_steps:], color='#0072B2', linewidth=LINEWIDTH_VERTICAL+5, linestyle=':')
ax.plot(range(past_steps, X.shape[1]), mean_trajectory_orig[past_steps:], color='#0072B2', linewidth=LINEWIDTH_VERTICAL+2, linestyle='--')

# Add a vertical line at past_steps (intervention time)
ax.axvline(x=past_steps, color='darkgray', linestyle='-', linewidth=LINEWIDTH_VERTICAL, label='Intervention Time')

# Add a horizontal line at the threshold
ax.axhline(y=threshold, color='darkgreen', linestyle='--', linewidth=LINEWIDTH_VERTICAL-1, label='Acceptance Threshold', alpha=0.7)

# Add a vertical line at maximum_time
ax.axvline(x=maximum_time + 0.05, color='black', linestyle='dotted', linewidth=LINEWIDTH_VERTICAL+3, label='Maximum Time', alpha=0.7)

# Convert x-axis from time steps to years
time_steps = np.arange(X_int.shape[1])
year_labels = [time_to_year(t) for t in time_steps]

# Aumenta la dimensione del font per le etichette dell'asse x
ax.set_xticks(time_steps[::time_per_year])  # Set ticks every 100 steps (1 year)
ax.set_xticklabels([f'{int(year)}' for year in year_labels[::time_per_year]], fontsize=FONTSIZE_LABELS - 5)

# Aumenta la dimensione del font per le etichette dell'asse y
import matplotlib.ticker as ticker
def y_formatter(x, pos):
    return f'{x/500:.0f}'
ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
ax.tick_params(axis='y', labelsize=FONTSIZE_LABELS - 5)

ax.set_xlabel('Year', fontsize=FONTSIZE_LABELS)
ax.set_ylabel('Credit Score', fontsize=FONTSIZE_LABELS)

# Opzionale: aumenta la dimensione dei tick
ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_LABELS - 9, width=2, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=3)
ax.tick_params(axis='x', which='major', pad=15)

# Calculate the crossing times and plot the crossing points
crossing_times = []
crossing_points = []
non_crossing_points = []  # Store the non-crossing points

for i in range(n_simulations):
    trajectory = X_int[i, :, column]
    crossing_indices = np.where(trajectory >= threshold)[0]
    if len(crossing_indices) > 0 and crossing_indices[0] <= maximum_time:
        crossing_time = crossing_indices[0]
        crossing_times.append(crossing_time)
        crossing_points.append((crossing_time, threshold))
    else:
        non_crossing_points.append((maximum_time, trajectory[maximum_time]))

# Convert crossing and non-crossing points to numpy arrays for vectorized plotting
crossing_points = np.array(crossing_points)
non_crossing_points = np.array(non_crossing_points)

# Plot all crossing points at once
if len(crossing_points) > 0:
    ax.scatter(crossing_points[:, 0], crossing_points[:, 1], marker='+', color='darkgreen', s=400, linewidths=4, label='Accepted',
               alpha=0.45)

# Plot all non-crossing points at once with red "x"
if len(non_crossing_points) > 0:
    ax.scatter(non_crossing_points[:, 0], non_crossing_points[:, 1], color='red', marker='x', s=300, linewidths=2, label='Rejected',
               alpha=0.7, zorder=300)

kde_data, kde_bins = np.histogram(crossing_times, bins=15)
kde_max = kde_data.max()

# Get the current y-axis limits
y_min, y_max = ax.get_ylim()

kde_height = 0.24 * (y_max - y_min)

facecolors = []
# RGB values for the color darkgreen (#006400)
rgb_darkgreen = (0, 0.392, 0)

# Plot KDE bars from top
for count, bin_start, bin_end in zip(kde_data, kde_bins[:-1], kde_bins[1:]):
    bar_height = (count / kde_max) * kde_height
    # Use color intensity to represent the count
    color_intensity = 0.2 + (count / kde_max) * 0.8  # Ranges from 0.2 to 1.0
    facecolor = (*rgb_darkgreen, color_intensity)  # Use darkgreen shade with varying transparency
    facecolors.append(facecolor)
    ax.add_patch(plt.Rectangle((bin_start, y_max), bin_end - bin_start, -bar_height,
                               facecolor=facecolor, edgecolor='none'))

ax.set_ylim(y_min, y_max)

# Customize the plot
plt.tight_layout()
"""plt.subplots_adjust(top=0.70)  # Adjust top margin to make room for legend
import matplotlib.lines as mlines
# Move legend above the plot in a single row
handles, labels = ax.get_legend_handles_labels()


new_handles = []
new_labels = []

for handle, label in zip(handles, labels):
    if label == 'Accepted':
        new_handle = mlines.Line2D([], [], color='blue', marker='o', markersize=15, linestyle='None', label='Accepted')
        new_handles.append(new_handle)
        new_labels.append('Accepted')
    elif label == 'Rejected':
        new_handle = mlines.Line2D([], [], color='red', marker='x', markersize=15, markeredgewidth=5, linestyle='None', label='Rejected')
        new_handles.append(new_handle)
        new_labels.append('Rejected')
    else:
        if isinstance(handle, mlines.Line2D):
            new_handle = mlines.Line2D([], [], color=handle.get_color(), linestyle=handle.get_linestyle(), linewidth=6)
            new_handles.append(new_handle)
            new_labels.append(label)
desired_order = [

"Acceptance Threshold",
    "Intervention Time",
    "Maximum Time",
"Mean",
    "Accepted",
    "Rejected"
]

handle_dict = dict(zip(new_labels, new_handles))
handles = [handle_dict[label] for label in desired_order]
labels = desired_order
fig.legend(handles, labels, loc='upper center', ncol=len(handles) / 3, fontsize=FONTSIZE_LABELS - 15,
           bbox_to_anchor=(0.485, 1), frameon=False, columnspacing=0.3)
"""

handles, labels = plt.gca().get_legend_handles_labels()
country_handles = [handle for handle, label in zip(handles, labels) if label in ['Mean']]
country_labels = [label for label in labels if label in ['Mean']]


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

mean_index = country_labels.index('Mean')

ordered_handles = [
    legend_handles[mean_index],
]
ordered_labels = ['Mean']

FONTSIZE_LEGEND = 40
#plt.subplots_adjust(top=0.9, left=0.13)
plt.legend(handles=ordered_handles, labels=ordered_labels, ncol=1, loc='upper left', frameon=False,
           fontsize=FONTSIZE_LEGEND-6, bbox_to_anchor=(0.001, 1.02))
plt.savefig('../pdfs/german_crossed.pdf', dpi=200)
plt.show()