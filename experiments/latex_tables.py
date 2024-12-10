import pandas as pd


def generate_train_group(train_steps, forecast_groups):
    group_latex = ""
    total_rows = sum(len(group_rows) for _, group_rows in forecast_groups)

    for i, (forecast_steps, group_rows) in enumerate(forecast_groups):
        if i > 0:
            group_latex += r"\cmidrule(lr){3-8}" + "\n"

        for j, (dataset, formatted_values) in enumerate(group_rows):
            if i == 0 and j == 0:
                group_latex += f"{dataset} & \\multirow{{{total_rows}}}{{*}}{{\\centering{train_steps}}} & \\multirow{{{len(group_rows)}}}{{*}}{{\\centering{forecast_steps}}} & "
            elif j == 0:
                group_latex += f"{dataset} & & \\multirow{{{len(group_rows)}}}{{*}}{{\\centering{forecast_steps}}} & "
            else:
                group_latex += f"{dataset} & & & "

            group_latex += " & ".join(formatted_values)
            group_latex += r" \\"
            group_latex += "\n"

    return group_latex


def generate_one_column_latex_table(df, metric):
    def format_number_with_std(mean, std, is_oracle=False, is_var=False, is_dlinear=False, var_mean=None):
        def format_value(value, precision):
            formatted = f"{value:.{precision}f}"
            if formatted.startswith('0.'):
                formatted = formatted[1:]  # Rimuove lo zero iniziale
            elif formatted.startswith('1.') and len(formatted) > 4:
                formatted = formatted[:4]  # Mantiene solo tre cifre se inizia con 1.
            return formatted

        mean_formatted = format_value(mean, 3)
        std_formatted = format_value(std, 3)
        if is_oracle:
            return f"$\\mathtt{{{mean_formatted}}}$"

        result = f"${mean_formatted}_{{{std_formatted}}}$"
        if is_var:
            result = f"$\\mathbf{{{mean_formatted}}}_{{{std_formatted}}}$"
        elif is_dlinear and var_mean is not None and abs(mean - var_mean) < 1e-6:
            result = f"$\\mathbf{{{mean_formatted}}}_{{{std_formatted}}}$"

        return result

    df['dataset'] = df['dataset'].replace('inverted', 'pendulum')
    grouped = df.groupby(['dataset', 'train_steps', 'forecast_prediction_steps', 'model'])['score']
    stats_df = grouped.agg(['mean', 'std']).reset_index()

    pivot_df = stats_df.pivot(index=['dataset', 'train_steps', 'forecast_prediction_steps'],
                              columns='model',
                              values=['mean', 'std'])

    dataset_order = ['pendulum', 'german']
    pivot_df = pivot_df.sort_index(level=['train_steps', 'forecast_prediction_steps'])
    pivot_df = pivot_df.reindex(dataset_order, level='dataset')

    latex_table = r"""\begin{table}[t]
    \centering
    \caption{\textbf{Observational forecasting.} """ + metric + r""" scores (\emph{lower is better}) for \VAR{}, DLinear~\citep{zeng2023dlinear}, TiDE~\cite{das2023tide} and TSMixer~\cite{chen2023tsmixer}, benchmarked against the oracle forecaster. Results averaged over ten runs, with standard deviation in subscript.}\label{tab:observational_forecasting_""" + metric + r"""}
    \setlength{\tabcolsep}{1pt}
    \footnotesize
    \renewcommand{\arraystretch}{1.18}
    \begin{tabular}{lcccccccc}
    \toprule
    & & & \multicolumn{5}{c}{""" + metric.upper() + r"""} \\
    \cmidrule(lr){4-8}
    Dataset & Size & Horizon & Oracle & VAR & DLinear & TiDE & TSMixer \\
    \midrule
    """

    train_groups = list(pivot_df.groupby(level='train_steps'))
    for i, (train_steps, train_group) in enumerate(train_groups):
        forecast_groups = []
        for forecast_steps, forecast_group in train_group.groupby(level='forecast_prediction_steps'):
            group_rows = []
            for dataset, row in forecast_group.groupby(level='dataset'):
                formatted_values = []
                var_mean = row[('mean', 'VAR')].iloc[0] if pd.notna(row[('mean', 'VAR')].iloc[0]) else None
                for model in ['Oracle', 'VAR', 'DLinear', 'TiDE', 'TSMixer']:
                    if pd.notna(row[('mean', model)].iloc[0]) and pd.notna(row[('std', model)].iloc[0]):
                        formatted_values.append(format_number_with_std(
                            row[('mean', model)].iloc[0],
                            row[('std', model)].iloc[0],
                            model == 'Oracle',
                            model == 'VAR',
                            model == 'DLinear',
                            var_mean
                        ))
                    else:
                        formatted_values.append("-")
                group_rows.append((dataset.capitalize(), formatted_values))
            forecast_groups.append((forecast_steps, group_rows))

        latex_table += generate_train_group(train_steps, forecast_groups)
        if i < len(train_groups) - 1:
            latex_table += r"\midrule" + "\n"
    latex_table += r"""\bottomrule
    \end{tabular}
\end{table}
"""

    return latex_table


def interventions_with_future_steps_german(mean_data, std_data, dataset_name="German"):
    def format_number_with_std(mean, std):
        def format_value(value, precision):
            formatted = f"{value:.{precision}f}"
            if formatted.startswith('0.'):
                formatted = formatted[1:]  # Remove leading zero
            elif formatted.startswith('1.') or formatted.startswith('2.') and len(formatted) > 4:
                formatted = formatted[:4]  # Keep only three digits if it starts with 1.
            return formatted

        mean_formatted = format_value(mean * 1000, 3)  # Multiply by 1000
        std_formatted = format_value(std * 1000, 3)  # Multiply by 1000
        return f"${mean_formatted}_{{{std_formatted}}}$"

    # Combine mean and std data
    combined_data = pd.concat([mean_data, std_data], axis=1)
    combined_data.columns = ['mean', 'std']
    combined_data = combined_data.reset_index()

    # Get unique train steps and h-steps
    unique_train_steps = sorted(combined_data['train_steps'].unique())
    unique_h_steps = sorted(combined_data['future steps'].unique())

    # Generate LaTeX table
    latex_table = r"""\begin{table}[t]
    \centering
    \caption{Evaluation of interventional forecasting models on the """ + dataset_name + r""" dataset (\emph{lower scores are better}). Results averaged over multiple runs. Values are multiplied by $10^3$ for readability.}
    \label{tab:interventional_forecasting_""" + dataset_name.lower() + r"""}
    \setlength{\tabcolsep}{5pt}
    \footnotesize
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{ccccc}
    \toprule
    & & & \multicolumn{2}{c}{Interventional Forecasting} \\
    \cmidrule(lr){4-5}
    Dataset & Size & Horizon & Additive & Forcing \\
    \midrule
    """

    latex_table += f"\\multirow{{{len(unique_train_steps) * len(unique_h_steps)}}}{{*}}{{{dataset_name}}}"

    for i, train_steps in enumerate(unique_train_steps):
        for j, h_steps in enumerate(unique_h_steps):
            train_data = combined_data[
                (combined_data['train_steps'] == train_steps) & (combined_data['future steps'] == h_steps)]
            additive = train_data[train_data['Type'] == 'Additive']
            forcing = train_data[train_data['Type'] == 'Forcing']

            additive_value = format_number_with_std(additive['mean'].iloc[0], additive['std'].iloc[0])
            forcing_value = format_number_with_std(forcing['mean'].iloc[0], forcing['std'].iloc[0])

            if j == 0:
                latex_table += f" & \\multirow{{2}}{{*}}{{{train_steps}}} & {h_steps} & {additive_value} & {forcing_value} \\\\\n"
            else:
                latex_table += f"& & {h_steps} & {additive_value} & {forcing_value} \\\\\n"

        if i < len(unique_train_steps) - 1:
            latex_table += r"\cmidrule(lr){2-5}" + "\n"

    latex_table += r"""\bottomrule
    \end{tabular}
    \end{table}
    """

    return latex_table


def interventions_with_future_steps_appendix(mean_data, std_data, metric):
    def format_number_with_std(mean, std):
        def format_value(value, precision):
            formatted = f"{value:.{precision}f}"
            if formatted.startswith('0.'):
                formatted = formatted[1:]  # Remove leading zero
            elif formatted.startswith('1.') or formatted.startswith('2.') and len(formatted) > 4:
                formatted = formatted[:4]
            return formatted

        mean_formatted = format_value(mean, 3)  # Multiply by 100
        std_formatted = format_value(std, 3)  # Multiply by 100
        return f"${mean_formatted}_{{{std_formatted}}}$"

    # Combine mean and std data
    combined_data = pd.concat([mean_data, std_data], axis=1)
    combined_data.columns = ['mean', 'std']
    combined_data = combined_data.reset_index()

    # Get unique train steps, h-steps, and datasets
    unique_train_steps = sorted(combined_data['train_steps'].unique())
    unique_h_steps = sorted(combined_data['future steps'].unique())
    unique_datasets = sorted(combined_data['dataset'].unique())

    # Format metric name and arrow
    if metric.lower() == 'rmse':
        metric_formatted = "$RMSE \\downarrow$"
    elif metric.upper() == 'SMAPE':
        metric_formatted = "SMAPE $\\downarrow$"
    else:
        metric_formatted = metric.upper()

    # Generate LaTeX table
    latex_table = r"""\begin{table}[t]
    \centering
    \caption{\textbf{Interventional forecasting.} """ + metric + r""" scores (\emph{lower is better}) for Additive and Forcing Interventions. 
    Results averaged over ten runs. Scores are multiplied by $10^2$ for readability.}
    \label{tab:interventional_forecasting_""" + metric + r"""}
    \setlength{\tabcolsep}{1.5pt}
    \footnotesize
    \renewcommand{\arraystretch}{1.18}
    \begin{tabular}{lccccc}
    \toprule
    & & & \multicolumn{2}{c}{""" + metric_formatted + r"""} \\
    \cmidrule(lr){4-5}
    Dataset & Train Steps & Horizon & Additive & Forcing \\
    \midrule
    """

    for i, train_steps in enumerate(unique_train_steps):
        for j, h_steps in enumerate(unique_h_steps):
            for k, dataset in enumerate(unique_datasets):
                filtered_data = combined_data[(combined_data['dataset'] == dataset) &
                                              (combined_data['train_steps'] == train_steps) &
                                              (combined_data['future steps'] == h_steps)]

                additive = filtered_data[filtered_data['Type'] == 'Additive']
                forcing = filtered_data[filtered_data['Type'] == 'Forcing']

                additive_value = format_number_with_std(additive['mean'].iloc[0], additive['std'].iloc[0])
                forcing_value = format_number_with_std(forcing['mean'].iloc[0], forcing['std'].iloc[0])

                if k == 0 and j == 0:
                    latex_table += f"{dataset.capitalize()} & \multirow{{4}}{{*}}{{\centering{train_steps}}} & \multirow{{2}}{{*}}{{\centering{h_steps}}} & {additive_value} & {forcing_value} \\\\\n"
                elif k == 0:
                    latex_table += f"{dataset.capitalize()} & & \multirow{{2}}{{*}}{{\centering{h_steps}}} & {additive_value} & {forcing_value} \\\\\n"
                else:
                    latex_table += f"{dataset.capitalize()} & & & {additive_value} & {forcing_value} \\\\\n"

            if j < len(unique_h_steps) - 1:
                latex_table += r"\cmidrule(lr){3-5}" + "\n"

        if i < len(unique_train_steps) - 1:
            latex_table += r"\midrule" + "\n"

    latex_table += r"""\bottomrule
    \end{tabular}
    \end{table}
    """

    return latex_table
