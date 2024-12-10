import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from typing import List, Tuple

# Set the linewidth for plot elements
LINEWIDTH = 11

def create_legend_figure(
    handles: List[mlines.Line2D],
    labels: List[str],
    ncol: int = 4,
    fig_size: Tuple[int, int] = (30, 4),
    font_size: int = 34,
    title_size: int = 16
) -> Tuple[plt.Figure, plt.Legend]:
    """
    Create a figure with only a legend.

    Args:
        handles (List[mlines.Line2D]): A list of Line2D objects representing the legend items.
        labels (List[str]): A list of labels corresponding to the legend items.
        ncol (int): Number of columns in the legend.
        fig_size (Tuple[int, int]): Size of the figure.
        font_size (int): Font size for legend text.
        title_size (int): Font size for legend title.

    Returns:
        Tuple[plt.Figure, plt.Legend]: The figure and legend objects.
    """
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis('off')  # Hide the axes
    legend = ax.legend(handles, labels, loc='center', ncol=ncol,
                       fontsize=font_size, title_fontsize=title_size,
                       frameon=False, columnspacing=3.5)
    return fig, legend

# Define the legend items (handles) with their corresponding styles and labels
handles: List[mlines.Line2D] = [
    mlines.Line2D([], [], color='darkgreen', marker='+', markersize=LINEWIDTH + 15,
                  markeredgewidth=6, linestyle='None', label='Accepted'),
    mlines.Line2D([], [], color='red', marker='x', markersize=LINEWIDTH + 5,
                  markeredgewidth=6, linestyle='None', label='Rejected'),
    mlines.Line2D([], [], color='black', linestyle='dotted', linewidth=LINEWIDTH, label='Maximum Time'),
    mlines.Line2D([], [], color='darkgreen', linestyle='--', linewidth=LINEWIDTH-2, label='Acceptance Threshold'),
    mlines.Line2D([], [], color='#0072B2', linestyle='-', linewidth=LINEWIDTH-2, label='Past Observations'),
    mlines.Line2D([], [], color='#0072B2', linestyle='--', linewidth=LINEWIDTH-3, label='Observational Forecast'),
    mlines.Line2D([], [], color='#0072B2', linestyle=':', linewidth=LINEWIDTH, label='Interventional Forecast'),
    mlines.Line2D([], [], color='darkgray', linestyle='-', linewidth=LINEWIDTH, label='Intervention Time'),
]

# Extract labels from handles
labels: List[str] = [handle.get_label() for handle in handles]

# Create the figure with the combined legend
fig, legend = create_legend_figure(handles, labels)

# Save the figure as a PDF file
plt.savefig('../pdfs/legend_unique.pdf', dpi=200)

# Display the figure
plt.show()
