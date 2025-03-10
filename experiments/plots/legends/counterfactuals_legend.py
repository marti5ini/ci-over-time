import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import List, Tuple

# Set the linewidth for plot elements
LINEWIDTH = 10

# Function to create a figure with a centered legend
def create_legend_figure(
    handles: List[mlines.Line2D],
    labels: List[str],
    ncol: int
) -> Tuple[plt.Figure, plt.Legend]:
    """
    Create a figure with only a legend.

    Args:
        handles (List[mlines.Line2D]): A list of Line2D objects representing the legend items.
        labels (List[str]): A list of labels corresponding to the legend items.
        ncol (int): Number of columns in the legend.

    Returns:
        Tuple[plt.Figure, plt.Legend]: The figure and legend objects.
    """
    fig, ax = plt.subplots(figsize=(30, 1))  # Create a new figure with a specified size
    ax.axis('off')  # Hide the axes
    legend = ax.legend(handles, labels, loc='center', ncol=ncol,
                       fontsize=36, title_fontsize=16,
                       frameon=False, columnspacing=2.3)  # Create the legend
    return fig, legend

# Define the legend items (handles) with their corresponding styles and labels
handles: List[mlines.Line2D] = [
    mlines.Line2D([], [], color='#0072B2', linestyle='-', linewidth=LINEWIDTH-2, label='Past'),
    mlines.Line2D([], [], color='darkmagenta', linestyle='dotted', linewidth=LINEWIDTH, label='Counterfactual'),
    mlines.Line2D([], [], color='darkgray', linestyle='-', linewidth=LINEWIDTH, label='Hypothetical Intervention'),
    mlines.Line2D([], [], color='black', linestyle='--', linewidth=LINEWIDTH, label='Present')
]

# Extract labels from handles
labels: List[str] = [handle.get_label() for handle in handles]

# Create the figure with the combined legend
fig, legend = create_legend_figure(handles, labels, ncol=5)

# Save the figure as a PDF file
plt.savefig('../pdfs/legend_counterfactual.pdf', dpi=200)

# Display the figure
plt.show()
