import numpy as np
from pathlib import Path

import pickle
AllData = {}
with open('input/default.p', 'rb') as handle:
    AllData = pickle.load(handle)
DataList = AllData['observables'][0][1]

import src
src.Initialize()
from src import mcmc
chain = mcmc.Chain()
MCMCSamples = chain.load()

from src import lazydict, emulator
Emulator = emulator.Emulator.from_cache('HeavyIon')

tag = AllData['tag']

# Write posterior out for beautiful plots
Posterior = Emulator.predict(MCMCSamples)

for Item in DataList:
    np.savetxt(f'result/{tag}/txt/' + Item + '_Posterior.txt.gz', Posterior['R_AA'][Item])


# Write the samples out
np.savetxt(f'result/{tag}/txt/MCMCSamples.txt', MCMCSamples)

# Write QHat out

def running_alpha_s(mu_square: float, alphas: float) -> float:
    active_flavor = 3
    square_lambda_QCD_HTL = np.exp( -12.0 * np.pi/( (33 - 2 * active_flavor) * alphas) );
    ans = 12.0 * np.pi/( (33.0 - 2.0 * active_flavor) * np.log(mu_square/square_lambda_QCD_HTL) );
    if mu_square < 1.0:
        ans = alphas
    # print(f"Fixed-alphaS={alphas}, Lambda_QCD_HTL={np.sqrt(square_lambda_QCD_HTL)}, mu2={mu_square}, Running alpha_s={ans}")
    return ans;

def qhat(T=0, E=0, Q=0, parameters=None) -> float:
    model = "exponential"
    if model == "exponential":
        # Parameters
        # alpha_s_fix, Q0, C1, C2, tau_0, C3 = parameters
        alpha_s_fix = parameters[0]
        active_flavor = 3

        # Extracted from JetScapeConstants
        C_a = 3.0

        # From GeneralQhatFunction
        debye_mass_square = alpha_s_fix * 4 * np.pi * np.power(T, 2.0) * (6.0 + active_flavor) / 6.0
        scale_net = 2 * E * T
        if scale_net < 1.0:
            scale_net = 1.0

        # alpha_s should be taken as 2*E*T, per Abhijit
        # See: https://jetscapeworkspace.slack.com/archives/C025X5NE9SN/p1648404101376299
        # answer = (C_a * 50.4864 / np.pi) * running_alpha_s(mu_square=scale_net, alphas=scale_net) * alpha_s_fix * np.power(T, 3) * np.abs(np.log(scale_net / debye_mass_square))
        answer = (C_a * 50.4864 / np.pi) * running_alpha_s(mu_square=scale_net, alphas=scale_net) * alpha_s_fix * np.abs(np.log(scale_net / debye_mass_square))   # nb in this one I removed T^3

        # This is not to be evaluated when evaluating qhat(T, E), per Abhijit. See: https://jetscapeworkspace.slack.com/archives/C025X5NE9SN/p1648404101376299
        # qhat = qhat * _virtuality_qhat_function(
        #  qhat_parametrization_type=ParametrizationType.exponential, ener_loc=E,
        #  # mu_square is the virtuality.
        #  # see: https://github.com/JETSCAPE/JETSCAPE-COMP/blob/e83b8ac71f8d71b9ad8ed71935f85d8951a16cb9/src/jet/Matter.cc#L805
        #  mu_square=Q,
        #  Q0=Q0, C1=C1, C2=C2, C3=C3,
        #  # C4 is unused for this parametrization, so just set to 0
        #  C4=0,
        #)

        return answer * 0.19732698   # 1/GeV to fm

Y = [qhat(T = 0.2, E = 100, Q = 0, parameters = x) for x in MCMCSamples]
np.savetxt(f'result/{tag}/txt/QHat_02_100_0.txt', Y)


