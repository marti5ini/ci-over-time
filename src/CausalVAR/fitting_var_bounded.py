import cvxpy as cp
import numpy as np
import statsmodels.tsa.vector_ar.util as util
from statsmodels.tsa.vector_ar.svar_model import VARResults


def estimate_bounded(model, lags: int, trend: str = 'c') -> VARResults:
    """
    Estimate a bounded Vector Autoregression (VAR) model with demographic constraints.

    :param model: The VAR model object
    :param lags: Number of lags in the model
    :param trend: Type of trend to include ('c' for constant, 'ct' for constant and trend, 'ctt' for constant, trend, and quadratic trend)
    :return: VARResults object
    """
    k_trend = util.get_trendorder(trend)
    y = model.endog

    z = util.get_var_endog(y, lags, trend=trend, has_constant='raise')
    y_sample = y[lags:]
    d_1 = z.shape[1] - y_sample.shape[1]
    d = y_sample.shape[1]

    var_params_temp = cp.Variable((d + d_1, d))
    objective = cp.Minimize(cp.norm(z @ var_params_temp - y_sample, 'fro'))
    constraints = []

    #legend: birth->0 death->1 migration->2 0-14->3 15-64->4 65+->5 , sum 3 on first dimension
    # trends legend [first dimension] constant  linear  quadratic
    if trend == 'ctt':
        constraints.append(var_params_temp[1:3, 3:6] == 0)  # no  t and tt for populations
        constraints.append(var_params_temp[6:8, 0:2] == 0)  # first three -> last three, not contrary (apart 65+ -> deaths)
        constraints.append(var_params_temp[8, 0] == 0)
        #constraints.append(var_params_temp[6:9, 3:6] == 0) # no interaction between populations diffs
        #NO INTERACTION BETWEEN BIRTH DEATH MIGRATION
        constraints.append(var_params_temp[3, 1:3] == 0) #NO BIRTH -> OTHERS
        constraints.append(var_params_temp[4, 0] == 0) #NO DEATH -> OTHERS
        #constraints.append(var_params_temp[4,2] == 0 )
        #constraints.append(var_params_temp[5, 2] == 0)
        constraints.append(var_params_temp[5, 0:2] == 0)
        #BIRTHS -> POPULATIONS
        constraints.append(var_params_temp[3, 3] == 1)  # births all in 0-14
        constraints.append(var_params_temp[3, 4] == 0)
        constraints.append(var_params_temp[3, 5] == 0)
        #DEATHS -> POPULATIONS
        constraints.append(var_params_temp[4, 3] <= 0)  # deaths negatively impact 0-14
        constraints.append(var_params_temp[4, 3] >= -1)
        constraints.append(var_params_temp[4, 4] <= 0)  # deaths negatively impact 15-64
        constraints.append(var_params_temp[4, 4] >= -1)
        constraints.append(var_params_temp[4, 5] <= -0.7)  # deaths negatively impact 65+
        constraints.append(var_params_temp[4, 5] == -1 - var_params_temp[4, 3] - var_params_temp[4, 4])
        #MIGRATIONS -> POPULATIONS
        constraints.append(var_params_temp[5, 3] >= 0)  # migration impact positively 0-14
        constraints.append(var_params_temp[5, 3] <= 1)
        constraints.append(var_params_temp[5, 4] <= 1)  # migration pos 15-64
        constraints.append(var_params_temp[5, 4] >= 0)
        constraints.append(var_params_temp[3, 5] >= 0)  # migration impact positively 65+
        constraints.append(var_params_temp[5, 5] <= 1)  # migration pos 65+
        constraints.append(var_params_temp[5, 5] == 1 - var_params_temp[5, 3] - var_params_temp[5, 4])
        #POPULATIONS --> POPULATIONS
        constraints.append(var_params_temp[6, 4] == 1/15) # 0-14 -> 15-64
        constraints.append(var_params_temp[6,3]== 14/15) # remaining 0-14
        constraints.append(var_params_temp[7, 5] == 1/50) # 15-64 -> 65+
        constraints.append(var_params_temp[7, 4] >= 49/50) # remaining 15-64
        constraints.append(var_params_temp[8, 5] == 1) #all 65+ remain or die
        # PoPULATIONS not -> POPULATIONS
        constraints.append(var_params_temp[6, 5] == 0)  # 0-14 not-> 65+
        constraints.append(var_params_temp[7, 3] == 0)  # 15-64 not-> 0-14
        constraints.append(var_params_temp[8, 3] == 0)  # 65+ not-> 0-14
        constraints.append(var_params_temp[8, 4] == 0)  # 65+ not-> 15-64

    if trend == 'ct':
        constraints.append(var_params_temp[1:2, 3:6] == 0)  # no  t and tt for populations
        #constraints.append(var_params_temp[5:8, 0:3] == 0)  # first three -> last three, not contrary
        constraints.append(var_params_temp[5:7, 0:2] == 0)  # first three -> last three, not contrary (apart 65+ -> deaths)
        constraints.append(var_params_temp[7, 0] == 0)
        # constraints.append(var_params_temp[6:9, 3:6] == 0) # no interaction between populations diffs
        # NO INTERACTION BETWEEN BIRTH DEATH MIGRATION
        constraints.append(var_params_temp[2, 1:3] == 0)  # NO BIRTH -> OTHERS
        constraints.append(var_params_temp[3, 0] == 0)  # NO DEATH -> OTHERS
        # constraints.append(var_params_temp[4,2] == 0 )
        # constraints.append(var_params_temp[5, 2] == 0)
        constraints.append(var_params_temp[4, 0:2] == 0)
        # BIRTHS -> POPULATIONS
        constraints.append(var_params_temp[2, 3] == 1)  # births all in 0-14
        constraints.append(var_params_temp[2, 4] == 0)
        constraints.append(var_params_temp[2, 5] == 0)
        # DEATHS -> POPULATIONS
        constraints.append(var_params_temp[3, 3] <= 0)  # deaths negatively impact 0-14
        constraints.append(var_params_temp[3, 3] >= -1)
        constraints.append(var_params_temp[3, 4] <= 0)  # deaths negatively impact 15-64
        constraints.append(var_params_temp[3, 4] >= -1)
        constraints.append(var_params_temp[3, 5] <= -0.5)  # deaths negatively impact 65+
        constraints.append(var_params_temp[3, 5] == -1 - var_params_temp[3, 3] - var_params_temp[3, 4])
        # MIGRATIONS -> POPULATIONS
        constraints.append(var_params_temp[4, 3] >= 0)  # migration impact positively 0-14
        constraints.append(var_params_temp[4, 3] <= 1)
        constraints.append(var_params_temp[4, 4] <= 1)  # migration pos 15-64
        constraints.append(var_params_temp[4, 4] >= 0)
        constraints.append(var_params_temp[4, 5] >= 0)  # migration impact positively 65+
        constraints.append(var_params_temp[4, 5] <= 1)  # migration pos 65+
        constraints.append(var_params_temp[4, 5] == 1 - var_params_temp[4, 3] - var_params_temp[4, 4])
        # POPULATIONS --> POPULATIONS
        constraints.append(var_params_temp[5, 4] == 1 / 15)  # 0-14 -> 15-64
        constraints.append(var_params_temp[5, 3] == 14 / 15)  # remaining 0-14
        constraints.append(var_params_temp[6, 5] == 1 / 50)  # 15-64 -> 65+
        constraints.append(var_params_temp[6, 4] >= 49 / 50)  # remaining 15-64
        constraints.append(var_params_temp[7, 5] == 1)  # all 65+ remain or die
        # PoPULATIONS not -> POPULATIONS
        constraints.append(var_params_temp[5, 5] == 0)  # 0-14 not-> 65+
        constraints.append(var_params_temp[6, 3] == 0)  # 15-64 not-> 0-14
        constraints.append(var_params_temp[7, 3] == 0)  # 65+ not-> 0-14
        constraints.append(var_params_temp[7, 4] == 0)  # 65+ not-> 15-64


    if trend == 'c':
        #constraints.append(var_params_temp[0:3, 3:6] == 0)  # no  t and tt for populations
        #constraints.append(var_params_temp[4:7, 0:3] == 0)  # first three -> last three, not contrary
        constraints.append(var_params_temp[4, 0:2] == 0)  # first three -> last three, not contrary (apart 65+ -> deaths and 15-64 ->births)
        constraints.append(var_params_temp[5, 0] >= 0) # 15-64 impact positively on births
        constraints.append(var_params_temp[5, 0] >= 0) #15-64 impact positively on deaths
        constraints.append(var_params_temp[6, 0] == 0) #65+ not-> births
        #constraints.append(var_params_temp[6:9, 3:6] == 0) # no interaction between populations diffs
        #NO INTERACTION BETWEEN BIRTH DEATH MIGRATION
        constraints.append(var_params_temp[1, 1:3] == 0) #NO BIRTH -> OTHERS
        constraints.append(var_params_temp[2, 0] == 0) #NO DEATH -> Births
        constraints.append(var_params_temp[2, 2] == 0)  # NO DEATH -> Migrations
        #constraints.append(var_params_temp[4,2] == 0 )
        #constraints.append(var_params_temp[5, 2] == 0)
        constraints.append(var_params_temp[3, 0:2] == 0)
        #BIRTHS -> POPULATIONS
        constraints.append(var_params_temp[1, 3] == 1)  # births all in 0-14
        constraints.append(var_params_temp[1, 4] == 0)
        constraints.append(var_params_temp[1, 5] == 0)
        #DEATHS -> POPULATIONS
        constraints.append(var_params_temp[2, 3] <= 0)  # deaths negatively impact 0-14
        constraints.append(var_params_temp[2, 3] >= -1)
        constraints.append(var_params_temp[2, 4] <= 0)  # deaths negatively impact 15-64
        constraints.append(var_params_temp[2, 4] >= -1)
        constraints.append(var_params_temp[2, 5] <= -0.5)  # deaths negatively impact 65+
        constraints.append(var_params_temp[2, 5] == -1 - var_params_temp[2, 3] - var_params_temp[2, 4])
        #MIGRATIONS -> POPULATIONS
        constraints.append(var_params_temp[3, 3] >= 0)  # migration impact positively 0-14
        constraints.append(var_params_temp[3, 3] <= 1)
        constraints.append(var_params_temp[3, 4] <= 1)  # migration pos 15-64
        constraints.append(var_params_temp[3, 4] >= 0)
        constraints.append(var_params_temp[3, 5] >= 0)  #migration impact positively 65+
        constraints.append(var_params_temp[3, 5] <= 1)  # migration pos 65+
        constraints.append(var_params_temp[3, 5] == 1 - var_params_temp[3, 3] - var_params_temp[3, 4])
        #POPULATIONS --> POPULATIONS
        constraints.append(var_params_temp[4, 4] == 1/15) # 0-14 -> 15-64
        constraints.append(var_params_temp[4,3]== 14/15) # remaining 0-14
        constraints.append(var_params_temp[5, 5] == 1/50) # 15-64 -> 65+
        constraints.append(var_params_temp[5, 4] >= 49/50) # remaining 15-64
        constraints.append(var_params_temp[6, 5] == 1) #all 65+ remain or die
        #PoPULATIONS not -> POPULATIONS
        constraints.append(var_params_temp[4, 5] == 0) #0-14 not-> 65+
        constraints.append(var_params_temp[5, 3] == 0) #15-64 not-> 0-14
        constraints.append(var_params_temp[6, 3] == 0)  # 65+ not-> 0-14
        constraints.append(var_params_temp[6, 4] == 0)  # 65+ not-> 15-64




    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    var_params = var_params_temp.value

    resid = y_sample - np.dot(z, var_params)

    avobs = len(y_sample)

    df_resid = avobs - (model.neqs * lags + k_trend)

    sse = np.dot(resid.T, resid)
    omega = sse / df_resid
    model.sigma_u = omega

    return VARResults(y, z, var_params, omega, lags,
                      names=model.endog_names, trend=trend,
                      dates=model.data.dates, model=model)
