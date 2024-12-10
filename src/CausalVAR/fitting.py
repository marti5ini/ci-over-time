import types
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import statsmodels.api as sm
import statsmodels.tsa.vector_ar.util as util
from numpy.typing import NDArray
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from statsmodels.tsa.vector_ar.svar_model import SVARResults


class SVARModel:
    def __init__(self, data: np.ndarray, relations: List[Tuple], names: List[str], n_lags: int = 3,
                 initialize_A_guess: bool = True):
        """
        Initialize the SVAR model.

        :param data: Input data for the model
        :param relations: List of relations between variables
        :param names: Names of the variables
        :param n_lags: Number of lags to consider
        :param initialize_A_guess: Whether to initialize A_0 guess
        """
        self.data = data
        self.n_columns = len(names)
        if initialize_A_guess:
            self.A_0_guess = self.initialize_A_0_g(relations)
        else:
            self.A_0_guess = np.eye(self.n_columns, dtype=object)
            self.A_0_guess[1, 0] = 'E'
        self.maxlags = n_lags
        self.A_avg: Optional[np.ndarray] = None
        self.coefs_avg: Optional[np.ndarray] = None
        self.sigma_u: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None

    def fitting(self, n_simulations: int, solver: str = 'nm', max_iter: int = 5000, trend: str = 'n',
                fixed_coefs: Optional[np.ndarray] = None):
        """
        Fit the SVAR model.

        :param n_simulations: Number of simulations to run
        :param solver: Solver to use for optimization
        :param max_iter: Maximum number of iterations
        :param trend: Trend type
        :param fixed_coefs: Fixed coefficients (if any)
        """
        A_tot = np.zeros((self.n_columns, self.n_columns))
        coefs_tot = np.zeros((self.maxlags, self.n_columns, self.n_columns))
        sigma_u_tot = np.zeros((self.n_columns, self.n_columns))

        for simulation in range(n_simulations):
            model = sm.tsa.SVAR(self.data[simulation], svar_type='A', A=self.A_0_guess)
            if fixed_coefs is not None:
                model._estimate_svar = types.MethodType(estimate_bounded, model)
                model.fixed_coefs = fixed_coefs
            res = model.fit(maxlags=self.maxlags, trend=trend, solver=solver, maxiter=max_iter)
            self.intercept = res.intercept
            A_tot += res.A_solve
            coefs_tot += res.coefs
            sigma_u_tot += res.sigma_u

        self.A_avg = A_tot / n_simulations
        self.coefs_avg = coefs_tot / n_simulations
        self.sigma_u = sigma_u_tot / n_simulations

    def initialize_A_0_g(self, relations: List[Tuple]) -> np.ndarray:
        """
        Initialize the A_0 guess matrix.

        :param relations: List of relations between variables
        :return: Initialized A_0 guess matrix
        """
        A_0_guess = np.zeros((self.n_columns, self.n_columns), dtype=object)
        np.fill_diagonal(A_0_guess, 1)
        filtered_relations = [rel for rel in relations if rel[0] == 0]
        for _, from_idx, to_idx, _ in filtered_relations:
            A_0_guess[to_idx, from_idx] = 'E'
        return A_0_guess


def estimate_bounded(self, start_params: np.ndarray, lags: int, maxiter: int, maxfun: int,
                     trend: str = 'c', solver: str = "nm", override: bool = False) -> SVARResults:
    """
    Estimate the SVAR model with bounded parameters.

    :param start_params: Initial parameters
    :param lags: Number of lags
    :param maxiter: Maximum number of iterations
    :param maxfun: Maximum number of function evaluations
    :param trend: Trend type
    :param solver: Solver to use for optimization
    :param override: Whether to override existing estimates
    :return: SVAR results
    """
    k_trend = util.get_trendorder(trend)
    y = self.endog

    z = util.get_var_endog(y, lags, trend=trend, has_constant='raise')
    y_sample = y[lags:]
    d_1 = z.shape[1] - y_sample.shape[1]
    d = y_sample.shape[1]

    var_params_temp = cp.Variable((d + d_1, d))
    objective = cp.Minimize(cp.norm(z @ var_params_temp - y_sample, 'fro'))

    constraints = []
    for i in range(d):
        for j in range(d):
            if self.fixed_coefs[i, j] == 0:
                constraints.append(var_params_temp[i + d_1, j] == 0)
            if self.fixed_coefs[i, j] != 0 and self.fixed_coefs[i, j] != 1:
                constraints.append(var_params_temp[i + d_1, j] == self.fixed_coefs[i, j])

    constraints.append(cp.sum(var_params_temp[5:8, 2]) == 1)

    for i in range(0, 2):
        for j in range(3, d):
            constraints.append(var_params_temp[i, j] == 0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    var_params = var_params_temp.value

    resid = y_sample - np.dot(z, var_params)

    avobs = len(y_sample)

    df_resid = avobs - (self.neqs * lags + k_trend)

    sse = np.dot(resid.T, resid)
    omega = sse / df_resid
    self.sigma_u = omega

    A, B = self._solve_AB(start_params, override=override,
                          solver=solver,
                          maxiter=maxiter)
    A_mask = self.A_mask
    B_mask = self.B_mask

    return SVARResults(y, z, var_params, omega, lags,
                       names=self.endog_names, trend=trend,
                       dates=self.data.dates, model=self,
                       A=A, B=B, A_mask=A_mask, B_mask=B_mask)


def numpy_to_panel_data(train_data: NDArray, column_names: List[str], train_steps: int) -> pd.DataFrame:
    """
    Convert numpy array to panel data format.

    :param train_data: 3D numpy array of shape (num_samples, train_steps, num_features)
    :param column_names: List of column names for features
    :param train_steps: Number of time steps in the training data
    :return: pandas DataFrame in panel data format
    """
    num_samples, _, num_features = train_data.shape

    time_array = np.tile(np.arange(train_steps), num_samples)
    id_array = np.repeat(np.arange(num_samples), train_steps)

    data_array = train_data.reshape(-1, num_features)
    full_array = np.column_stack((time_array, id_array, data_array))

    column_names_full = ['Time', 'ID'] + column_names

    return pd.DataFrame(full_array, columns=column_names_full)


def get_coef_matrices_r(coefs_df: pd.DataFrame, column_names: List[str], n_lags: int, threshold: float = 0) -> List[
    NDArray]:
    """
    Extract coefficient matrices from R model output.

    :param coefs_df: DataFrame containing model coefficients
    :param column_names: List of variable names
    :param n_lags: Number of lags in the model
    :param threshold: Threshold for coefficient values (set to 0 if below)
    :return: List of coefficient matrices for each lag
    """
    num_vars = len(column_names)
    coef_matrices_r = []

    for lag in range(1, n_lags + 1):
        matrix = np.zeros((num_vars, num_vars))

        for i, var in enumerate(column_names):
            for j, var2 in enumerate(column_names):
                col_name = f'demeaned_lag{lag}_{var2}'
                if col_name in coefs_df.columns:
                    coef_value = coefs_df.loc[f'demeaned_{var}', col_name]
                    matrix[i, j] = coef_value

        matrix[np.abs(matrix) < threshold] = 0
        coef_matrices_r.append(matrix)

    return coef_matrices_r


def var_fit_with_r(train_data: NDArray, n_lags: int, column_names: Optional[List[str]] = None, threshold: float = 0) -> \
        tuple[NDArray, NDArray]:
    """
    Fit a VAR model using R's panelvar package.

    :param train_data: 3D numpy array of shape (num_samples, train_steps, num_features)
    :param n_lags: Number of lags in the VAR model
    :param column_names: Optional list of column names (default: None, uses indices)
    :param threshold: Threshold for coefficient values (default: 0)
    :return: Tuple of coefficient matrices and fitted sigma_u
    """
    pandas2ri.activate()
    panelvars = importr('panelvar')
    base = importr('base')

    if column_names is None:
        column_names = [str(i) for i in range(train_data.shape[2])]

    train_data_df = numpy_to_panel_data(train_data, column_names, train_data.shape[1])

    r_dataframe = pandas2ri.py2rpy(train_data_df)
    ro.globalenv['df_r'] = r_dataframe
    ro.globalenv['column_names'] = ro.StrVector(column_names)
    ro.globalenv['n_lags'] = n_lags

    # Fit panelvar model in R and calculate sigma_u
    ro.r('''
        panelvar_model <- pvarfeols(dependent_vars = column_names, 
                            lags = n_lags, 
                            transformation = c("demean"),
                            data = df_r, 
                            panel_identifier = c("ID", "Time")
                            )

        residuals <- panelvar_model$residuals
        sse <- t(residuals) %*% residuals
        T <- nrow(df_r)
        K <- length(column_names)
        df_resid <- T - (K * n_lags + 1)
        sigma_u <- sse / df_resid
    ''')
    fitted_sigma_u = np.array(ro.r('sigma_u'))

    # Extract coefficients matrices from R model
    coefs = ro.r('as.data.frame(coef(panelvar_model))')
    coefs_df = pandas2ri.rpy2py(coefs)
    coef_matrices_r = np.asarray(get_coef_matrices_r(coefs_df, column_names, n_lags, threshold))

    return coef_matrices_r, fitted_sigma_u
