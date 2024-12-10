import os

# Class paramaters
FUTURE_STEPS = 20
PAST_STEPS = 20
YEAR_START = '2004'
YEAR_START_CF='1984'


# General parameters for the plot function
dir_path = os.getcwd()
LINEWIDTH = 7.5
FONTSIZE_TITLE = 45
FONTSIZE_TICKS = 35
FONTSIZE_LEGEND = 9
YEAR_RANGE_X_LABELS = 5
SAVE_DIR = os.path.join(dir_path, 'pdfs')
SAVE_DIR_CF = os.path.join(os.path.dirname(dir_path), 'pdfs')
FIGSIZE = (9, 6.2)

# Legend labels
PAST_FACTUAL_LABEL = "Past and Factual"
CF_TRUE_LABEL = "CF True"
CF_PREDICTED_LABEL = "CF Predicted"
PAST_FORECAST_LABEL = "Past and Forecast"
TRUE_LABEL = "Intervened Oracle"
PREDICTED_LABEL = "Intervened VAR"


# Grid settings
GRID_ENABLED = True  # Set to False for disabling the grid
GRID_COLOR = 'gray'
GRID_STYLE = 'dotted'
GRID_LINEWIDTH = 0.3
