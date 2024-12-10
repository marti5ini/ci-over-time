import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from experiments.plots.config import (LINEWIDTH, FONTSIZE_TITLE, FONTSIZE_TICKS, YEAR_RANGE_X_LABELS,
                    YEAR_START, FONTSIZE_LEGEND, SAVE_DIR, FIGSIZE,
                    TRUE_LABEL, PREDICTED_LABEL, GRID_LINEWIDTH, GRID_COLOR, GRID_STYLE,
                    GRID_ENABLED, YEAR_START_CF, SAVE_DIR_CF)

line_cycler = (cycler(color=["#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]) +
               cycler(linestyle=["-", "--", (0, (5, 1)), ":", "-", "--", "-."]))

plt.rc("axes", prop_cycle=line_cycler)


def plot_cf_forecast2(past, test_factual, true_cf, predicted_cf, column_index, column_name,
                      title=None, year_end=None):
    # Extract the relevant column
    factual_line = test_factual[:, column_index]
    true_cf = true_cf[:, column_index]
    past_data = past[:, column_index].reshape(1, -1)
    predicted_cf = predicted_cf[:, column_index]

    factual_data = factual_line.reshape(1, -1)

    past_extended = np.hstack([past_data, factual_data[:, :1]]).flatten()
    factual_extended = factual_data.flatten()

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.set_prop_cycle(line_cycler)

    # Plot past data
    plt.plot(past_extended, label="Past", linewidth=LINEWIDTH)

    # Plot factual forecast
    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(factual_extended)),
             factual_extended, color='#0072B2', linestyle='-', label='Observational Oracle', linewidth=LINEWIDTH)

    # Plot predicted counterfactual
    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(predicted_cf)),
             predicted_cf.flatten(), label=PREDICTED_LABEL, linewidth=LINEWIDTH, linestyle='dotted',
             color='darkmagenta')

    if GRID_ENABLED:
        special_x = 20
        DARKER_GRID_COLOR = 'darkgray'
        plt.axvline(x=special_x, color=DARKER_GRID_COLOR, linestyle='-', linewidth=GRID_LINEWIDTH + 4)

    if GRID_ENABLED:
        special_x = 39.3
        DARKER_GRID_COLOR = 'black'
        plt.axvline(x=special_x, color=DARKER_GRID_COLOR, linestyle='--', linewidth=GRID_LINEWIDTH + 6)

    plt.xlim(0, len(past_data.flatten()) + len(true_cf) + 1)
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICKS)
    plt.xlabel("Year", fontsize=FONTSIZE_TITLE - 2, fontweight='normal')
    plt.ylabel(f'{column_name}', fontsize=FONTSIZE_TITLE - 2, fontweight='normal')

    freq_years = str(YEAR_RANGE_X_LABELS) + 'YE'
    years = pd.date_range(start=YEAR_START_CF, end=year_end, freq=freq_years)
    plt.gca().xaxis.set_major_locator(
        plt.FixedLocator(np.arange(0, len(years) * YEAR_RANGE_X_LABELS, YEAR_RANGE_X_LABELS)))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: years[int(x / YEAR_RANGE_X_LABELS)].year))

    ticks = np.arange(0, len(years) * YEAR_RANGE_X_LABELS, YEAR_RANGE_X_LABELS).tolist()

    # Add the special x position to ticks
    ticks.append(special_x)
    ticks.sort()

    # Set the x-ticks and corresponding labels
    plt.xticks(ticks, labels=[str(year.year) if year_index < len(years) else '2024'
                              for year_index, year in
                              enumerate(years.append(pd.to_datetime(['2024']))[:len(ticks)])],
               rotation=45)

    if column_name == 'Expertise':
        plt.yticks([0, 2, 5])

    plt.tight_layout()


    plt.grid(axis='x', color=GRID_COLOR, linestyle=GRID_STYLE, linewidth=GRID_LINEWIDTH)
    plt.grid(axis='y', visible=False)

    if title:
        plt.savefig(os.path.join(f'{SAVE_DIR_CF}', f'{title}.pdf'), dpi=200)

    plt.show()


def plot_ai_fi_intervention_forecast(data, true_forecast, true_forecast_cov, true_intervention_forecast,
                                     predicted_intervention_forecast,
                                     true_forcing_cov,
                                     intervention_dict=None, additive=False,
                                     year_end=None,
                                     column_index=None, column_name=None, title=None):
    past_data = data[:, column_index].reshape(1, -1)
    true_forecast_line = true_forecast[:, column_index]
    true_forecast_data = true_forecast_line.reshape(1, -1)

    past_and_true_forecast = np.hstack([past_data, true_forecast_data[:, :1]]).flatten()
    true_forecast_extended = true_forecast_data.flatten()

    true_forcing_forecast_line = true_intervention_forecast[:, column_index]
    pred_forcing_forecast_line = predicted_intervention_forecast[:, column_index]

    plt.figure(figsize=FIGSIZE)
    ax = plt.gca()
    ax.set_prop_cycle(line_cycler)
    if column_name in intervention_dict and additive == False:
        plt.hlines(y=intervention_dict[column_name], xmin=len(past_data.flatten()), xmax=len(past_data.flatten()) +
                                                                                         len(true_forecast_extended),
                   color='red', linewidth=LINEWIDTH + 1, alpha=0.7,
                   label=f'Target Value={intervention_dict[column_name]}')
    if column_name == 'LoanDuration':
        column_name = 'Loan Duration'

    plt.plot(past_and_true_forecast, label="Past", linewidth=LINEWIDTH)

    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(true_forecast_extended)),
             true_forecast_extended, label='Observational Oracle', linewidth=LINEWIDTH)

    true_forecast_cov = np.concatenate((np.zeros((1, 7, 7)), true_forecast_cov[:-1, :, :]), axis=0)
    std_error = np.sqrt(true_forecast_cov[:, column_index, column_index])
    upper_bound = true_forecast_line + 1 * std_error
    lower_bound = true_forecast_line - 1 * std_error
    plt.fill_between(range(len(past_data.flatten()), len(past_data.flatten()) + len(true_forecast_extended)),
                     lower_bound.flatten(), upper_bound.flatten(), alpha=0.15)

    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(true_forcing_forecast_line.flatten())),
             true_forcing_forecast_line.flatten(), label=TRUE_LABEL,
             linewidth=LINEWIDTH)

    true_forcing_cov = np.concatenate((np.zeros((1, 7, 7)), true_forcing_cov[:-1, :, :]), axis=0)
    std_error = np.sqrt(true_forcing_cov[:, column_index, column_index])
    upper_bound = true_forcing_forecast_line + 1 * std_error
    lower_bound = true_forcing_forecast_line - 1 * std_error
    plt.fill_between(
        range(len(past_data.flatten()), len(past_data.flatten()) + len(true_forcing_forecast_line.flatten())),
        lower_bound.flatten(), upper_bound.flatten(), color='#E69F00', alpha=0.15)

    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(pred_forcing_forecast_line.flatten())),
             pred_forcing_forecast_line.flatten(), label=PREDICTED_LABEL,
             linewidth=LINEWIDTH)

    if GRID_ENABLED:
        special_x = 20
        DARKER_GRID_COLOR = 'darkgray'
        plt.axvline(x=special_x, color=DARKER_GRID_COLOR, linestyle='-', linewidth=GRID_LINEWIDTH + 4)

    plt.xlim(0, len(past_data.flatten()) + len(true_forecast_extended) - 1)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICKS)
    ax.set_xlabel("Year", fontsize=FONTSIZE_TITLE - 2, fontweight='normal')
    if min(plt.gca().get_yticks()) < 0:  # Check if there are negative y-values
        plt.ylabel(f'{column_name}', fontsize=FONTSIZE_TITLE - 2, fontweight='normal')  # Adjust the labelpad as needed
    else:
        plt.ylabel(f'{column_name}', fontsize=FONTSIZE_TITLE - 2, fontweight='normal')
    plt.xticks(rotation=45)
    freq_years = str(YEAR_RANGE_X_LABELS) + 'YE'
    years = pd.date_range(start=YEAR_START, end=year_end, freq=freq_years)
    plt.gca().xaxis.set_major_locator(
        plt.FixedLocator(np.arange(0, len(years) * YEAR_RANGE_X_LABELS, YEAR_RANGE_X_LABELS)))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: years[int(x / YEAR_RANGE_X_LABELS)].year))
    plt.grid(axis='x', color=GRID_COLOR, linestyle=GRID_STYLE, linewidth=GRID_LINEWIDTH)
    plt.grid(axis='y', visible=False)
    plt.tight_layout()
    plt.savefig(os.path.join(f'{SAVE_DIR}', f'{title}.pdf'), dpi=200)
    plt.show()


def plot_forecast_smape(past, actual, predicted, actual_cov, predicted_cov, column_index,
                        column_name, forecast_capability=True, forecast_capability_name='True',
                        if_capability=True, smape_title=None, year_end=None):
    forecast_check_names = 0
    plot_past = False
    plt.figure(figsize=FIGSIZE)
    for forecast_values, forecast_cov in zip([actual, predicted], [actual_cov, predicted_cov]):
        forecast_line = forecast_values[:, column_index]
        past_data = past[:, column_index].reshape(1, -1)
        forecast_data = forecast_line
        forecast_extended = np.insert(forecast_data.flatten(), 0, past_data.flatten())

        if plot_past:
            plt.plot(range(len(past_data.flatten())), past_data.flatten(), color='blue', linestyle='-',
                     linewidth=LINEWIDTH)
        total_forecast_color = plt.gca().lines[-1].get_color()

        if forecast_capability:
            if if_capability:
                plt.plot(range(len(forecast_extended)), forecast_extended, linestyle='-', color=total_forecast_color,
                         linewidth=LINEWIDTH)
            else:
                plt.plot(range(len(forecast_extended)), forecast_extended, linestyle='--', color=total_forecast_color,
                         linewidth=LINEWIDTH)

        std_error = np.sqrt(forecast_cov[:, column_index, column_index])
        upper_bound = forecast_line + 1 * std_error
        lower_bound = forecast_line - 1 * std_error

        plt.fill_between(range(len(past_data.flatten()), len(past_data.flatten()) + len(forecast_line)),
                         lower_bound.flatten(), upper_bound.flatten(), color=total_forecast_color, alpha=0.2)

        forecast_check_names += 1
        if forecast_check_names == 1:
            plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(forecast_line)),
                     forecast_line.flatten(), linestyle='--', label=f'{forecast_capability_name} {TRUE_LABEL}',
                     color='red', linewidth=LINEWIDTH)
        elif forecast_check_names == 2:
            plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(forecast_line)),
                     forecast_line.flatten(), linestyle='--', label=f'{forecast_capability_name} {PREDICTED_LABEL}',
                     color='green', linewidth=LINEWIDTH)

    plt.legend(loc='upper left', fontsize=FONTSIZE_LEGEND)
    plt.title(f'{smape_title}', fontsize=FONTSIZE_TITLE, fontweight='bold')

    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICKS)
    plt.xlabel("Year", fontsize=FONTSIZE_TITLE - 2)
    plt.ylabel(f'{column_name}', fontsize=FONTSIZE_TITLE - 2)
    freq_years = str(YEAR_RANGE_X_LABELS) + 'Y'
    years = pd.date_range(start=YEAR_START, end=year_end, freq=freq_years)
    plt.gca().xaxis.set_major_locator(
        plt.FixedLocator(np.arange(0, len(years) * YEAR_RANGE_X_LABELS, YEAR_RANGE_X_LABELS)))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: years[int(x / YEAR_RANGE_X_LABELS)].year))

    plt.tight_layout()
    if GRID_ENABLED:
        plt.grid(axis='x', color=GRID_COLOR, linestyle=GRID_STYLE, linewidth=GRID_LINEWIDTH)
    plt.savefig(os.path.join(f'{SAVE_DIR}', f'{smape_title}.pdf'), dpi=200)
    plt.show()


def plot_capability_forecast(past, future, predicted_var, predicted_lstm, predicted_oracle, column_index,
                             column_name, title=None, year_end=None):
    plt.figure(figsize=FIGSIZE)
    if GRID_ENABLED:
        special_x = 20
        DARKER_GRID_COLOR = 'darkgray'
        plt.axvline(x=special_x, color=DARKER_GRID_COLOR, linestyle='-', linewidth=GRID_LINEWIDTH + 1)

    past_data = past[:, column_index].reshape(1, -1)
    future_line = future[:, column_index]
    future_data = future_line.reshape(1, -1)

    past_and_future = np.hstack([past_data, future_data[:, :1]]).flatten()
    future_extended = future_data.flatten()

    plt.plot(past_and_future, label="Past", linewidth=LINEWIDTH)

    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(future_extended)),
             future_extended, label='Future', linewidth=LINEWIDTH)

    pred_oracle_forecast_line = predicted_oracle[:, column_index]
    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(pred_oracle_forecast_line.flatten())),
             pred_oracle_forecast_line.flatten(), label='Forecast Oracle',
             linewidth=LINEWIDTH)

    pred_var_forecast_line = predicted_var[:, column_index]
    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(pred_var_forecast_line.flatten())),
             pred_var_forecast_line.flatten(), label='Forecast VAR',
             linewidth=LINEWIDTH)
    pred_lstm_forecast_line = predicted_lstm[:, column_index]
    plt.plot(range(len(past_data.flatten()), len(past_data.flatten()) + len(pred_lstm_forecast_line.flatten())),
             pred_lstm_forecast_line.flatten(), label='Forecast LSTM',
             linewidth=LINEWIDTH)

    plt.legend(loc='best', fontsize=FONTSIZE_LEGEND)
    plt.title(f'{title}', fontsize=FONTSIZE_TITLE)

    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE_TICKS)
    plt.xlabel("Year", fontsize=FONTSIZE_TITLE - 2)
    plt.ylabel(f'{column_name}', fontsize=FONTSIZE_TITLE - 2)
    plt.xlim(0, len(past_data.flatten()) + len(future_extended) - 1)
    plt.xticks(rotation=45)
    freq_years = str(YEAR_RANGE_X_LABELS) + 'YE'
    years = pd.date_range(start=YEAR_START, end=year_end, freq=freq_years)
    plt.gca().xaxis.set_major_locator(
        plt.FixedLocator(np.arange(0, len(years) * YEAR_RANGE_X_LABELS, YEAR_RANGE_X_LABELS)))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: years[int(x / YEAR_RANGE_X_LABELS)].year))

    plt.tight_layout()
    if GRID_ENABLED:
        plt.grid(axis='x', color=GRID_COLOR, linestyle=GRID_STYLE, linewidth=GRID_LINEWIDTH)
    plt.savefig(os.path.join(f'{SAVE_DIR}', f'{title}.pdf'), dpi=200)
    plt.show()
