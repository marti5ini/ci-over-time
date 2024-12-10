import types

import numpy as np
from statsmodels.tools.validation import int_like
from statsmodels.tsa.vector_ar.svar_model import SVARProcess
from typing import List, Optional, Tuple


class SVARSimulation:
    def __init__(self,
                 coefs: np.ndarray,
                 sigma_u: np.ndarray,
                 names: List[str],
                 A_0: np.ndarray,
                 intercept: Optional[np.ndarray] = None):
        """
        Initialize the SVAR simulation with model parameters.

        :param coefs: VAR coefficients
        :param sigma_u: Covariance matrix of reduced-form errors
        :param names: Variable names
        :param A_0: Contemporaneous impact matrix
        :param intercept: Intercept terms (default is zero vector)
        """
        self.coefs: np.ndarray = coefs
        self.sigma_u: np.ndarray = sigma_u
        self.names: List[str] = names
        self.A_0: np.ndarray = A_0
        self.n_simulations: Optional[int] = None
        self.simulated_data: Optional[np.ndarray] = None
        if intercept is None:
            self.intercept = np.zeros(len(self.names))
        else:
            self.intercept = intercept
        self._initialize_svar_process()

    def _initialize_svar_process(self):
        # Set up the SVAR process with the given parameters
        self.svar_process = SVARProcess(
            coefs=self.coefs,
            intercept=self.intercept,
            sigma_u=self.sigma_u,
            names=self.names,
            A_solve=self.A_0,
            B_solve=np.eye(len(self.names))
        )
        # Set additional attributes for svar_process if needed
        self.svar_process.k_exog_user = 0
        self.svar_process.k_trend = 0
        self.svar_process.trend = 'n'
        self.svar_process.exog = None
        self.svar_process.coefs_exog = np.zeros(len(self.names))
        self.svar_process.n_totobs = self.n_simulations

    def update_parameters(self,
                          coefs: np.ndarray,
                          sigma_u: np.ndarray,
                          A_0: np.ndarray) -> None:
        """
        Update the SVAR model parameters and reinitialize the process.

        :param coefs: New VAR coefficients
        :param sigma_u: New covariance matrix of reduced-form errors
        :param A_0: New contemporaneous impact matrix
        """
        self.coefs = coefs
        self.sigma_u = sigma_u
        self.A_0 = A_0
        self._initialize_svar_process()

    @staticmethod
    def varsim(coefs: np.ndarray,
               intercept: Optional[np.ndarray],
               sig_u: Optional[np.ndarray],
               steps: int = 100,
               initial_values: Optional[np.ndarray] = None,
               seed: Optional[int] = None,
               nsimulations: Optional[int] = None,
               ugen: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a Vector Autoregression (VAR) process.

        :param coefs: VAR coefficients
        :param intercept: Intercept or constant term
        :param sig_u: Covariance matrix of errors
        :param steps: Number of time steps to simulate
        :param initial_values: Initial values for the time series
        :param seed: Random seed for reproducibility
        :param nsimulations: Number of simulations to run
        :param ugen: Pre-generated random shocks
        :return: Tuple of simulated data and generated shocks
        """
        rs = np.random.RandomState(seed=seed)
        rmvnorm = rs.multivariate_normal
        p, k, k = coefs.shape
        nsimulations = int_like(nsimulations, "nsimulations", optional=True)
        if isinstance(nsimulations, int) and nsimulations <= 0:
            raise ValueError("nsimulations must be a positive integer if provided")
        if nsimulations is None:
            result_shape = (steps, k)
            nsimulations = 1
        else:
            result_shape = (nsimulations, steps + p, k)
        if sig_u is None:
            sig_u = np.eye(k)
        if ugen is None:
            ugen = rmvnorm(np.zeros(len(sig_u)), sig_u, (steps + p) * nsimulations).reshape(nsimulations, steps + p, k)

        result = np.zeros((nsimulations, steps + p, k))
        if intercept is not None:
            # intercept can be 2-D like an offset variable
            if np.ndim(intercept) > 1:
                if not len(intercept) == ugen.shape[1]:
                    raise ValueError('2-D intercept needs to have length `steps`')
            # add intercept/offset also to intial values
            result += intercept
            result[:, p:] += ugen[:, p:]
        else:
            result[:, p:] = ugen[:, p:]

        if initial_values is not None:
            if initial_values.shape[1] > p:
                initial_values = initial_values[:, -p:, :]
            result[:, :p, :] = initial_values
        # add in AR terms
        for t in range(p, steps + p):
            ygen = result[:, t]
            for j in range(p):
                ygen += np.dot(coefs[j], result[:, t - j - 1].T).T

        partial_result = result.reshape(result_shape)
        result = partial_result[:, p:, :]

        return result, ugen

    def generate_data(self,
                      n_simulations: int,
                      steps: int = 100,
                      ugen: Optional[np.ndarray] = None,
                      seed: Optional[int] = None,
                      offset: Optional[np.ndarray] = None,
                      initialvalues: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate simulated data based on the SVAR model.

        :param n_simulations: Number of simulations to run
        :param steps: Number of time steps to simulate
        :param ugen: Pre-generated random shocks (if None, will be generated)
        :param seed: Random seed for reproducibility
        :param offset: Offset values for the simulation
        :param initialvalues: Initial values for the time series
        :return: Tuple of simulated data and generated shocks
        """
        def simulation_var(svar_process, initial_values, steps=None, offset=None, seed=None, nsimulations=None, ugen=None):
            steps_ = None
            if offset is None:
                if svar_process.k_exog_user > 0 or svar_process.k_trend > 1:
                    offset = svar_process.endog_lagged[:, : svar_process.k_exog].dot(
                        svar_process.coefs_exog.T
                    )
                    steps_ = svar_process.endog_lagged.shape[0]
                else:
                    offset = svar_process.intercept
            else:
                steps_ = offset.shape[0]
            if steps is None:
                if steps_ is None:
                    steps = 1000
                else:
                    steps = steps_
            else:
                if steps_ is not None and steps != steps_:
                    raise ValueError(
                        "if exog or offset are used, then steps must"
                        "be equal to their length or None"
                    )

            y, ugen = self.varsim(
                svar_process.coefs,
                offset,
                svar_process.sigma_u,
                steps=steps,
                seed=seed,
                initial_values=initial_values,
                nsimulations=nsimulations,
                ugen=ugen
            )
            return y, ugen

        self.n_simulations = n_simulations
        self.svar_process.n_totobs = n_simulations
        self.svar_process.simulate_var = types.MethodType(simulation_var, self.svar_process)
        self.simulated_data, ugen = self.svar_process.simulate_var(steps=steps, nsimulations=n_simulations, seed=seed, ugen=ugen, offset=offset,
                                                                   initial_values=initialvalues)

        return self.simulated_data, ugen