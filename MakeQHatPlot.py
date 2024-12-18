import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pickle
AllData = {}
with open('input/default.p', 'rb') as handle:
    AllData = pickle.load(handle)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--Alternate", help = "whether to plot a second collection", type = str, default = '')
parser.add_argument("--NominalLabel", help = "label for the nominal", type = str, default = '')
parser.add_argument('--AlternateLabel', help = 'label for alternate collection', type = str, default = '')
parser.add_argument('--Prefix', help = 'prefix to add to file name', type = str, default = '')
args = parser.parse_args()

import src
src.Initialize()
from src import mcmc
chain = mcmc.Chain()
MCMCSamples = chain.load()

if args.Alternate != '':
    chain2 = mcmc.Chain(path = Path(f'result/{args.Alternate}/mcmc_chain.hdf'))
    MCMCSamples2 = chain2.load()

# These taken from Raymond code
# https://github.com/raymondEhlers/STAT/blob/1b0df83a9fd479f8110fd326ae26c0ce002a1109/run_analysis_base.py

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

def PlotQHat(T = 0.15, E = 100, Q = 100, Scan = 'T', P1 = [[1, 1, 1, 1, 1, 1]], P2 = [[]], Type = "", Prefix = "", Suffix = "", DoJet = False, DoGreen = False):

    if Suffix != "":
        Suffix = "_" + Suffix

    X = []
    Y = []
    XLabel = ""
    ExtraText = ""

    if Scan == 'T':
        X = np.arange(0.16, 0.52, 0.01)
        XLabel = "T"
        ExtraText = f"E = {E} GeV"
    elif Scan == 'E':
        X = np.arange(5, 200, 1)
        XLabel = "E"
        ExtraText = f"T = {T} GeV"
    elif Scan == 'Q':
        X = np.arange(5, 200, 1)
        XLabel = "Q"
        ExtraText = f"T = {T} GeV, E = {E} GeV"
    elif Scan == 'EQ':
        X = np.arange(5, 200, 1)
        XLabel = "E"
        ExtraText = f"T = {T} GeV, Q = E"
    else:
        print('Error!  Illegal scan variable')

    figure, axes = plt.subplots(figsize = (5, 5))

    AllY1 = []
    AllY2 = []

    NSample = P1.shape[0]
    for i in range(NSample):
        if Scan == 'T':
            Y = [qhat(T = x, E = E, Q = Q, parameters = P1[i]) for x in X]
            XLabel = "T (GeV)"
        elif Scan == 'E':
            Y = [qhat(T = T, E = x, Q = Q, parameters = P1[i]) for x in X]
            XLabel = "E (GeV)"
        elif Scan == 'Q':
            Y = [qhat(T = T, E = E, Q = x, parameters = P1[i]) for x in X]
            XLabel = "Q (GeV)"
        elif Scan == 'EQ':
            Y = [qhat(T = T, E = x, Q = x, parameters = P1[i]) for x in X]
            XLabel = "E (GeV)"
        else:
            print('Error!  Illegal scan variable')

        AllY1.append(Y)

        if DoGreen == False:
            axes.plot(X, Y, 'b', alpha = 40 / NSample)
        if DoGreen == True:
            axes.plot(X, Y, 'g', alpha = 40 / NSample)

    if P2 != [[]]:
        NSample = P2.shape[0]
        for i in range(NSample):
            if Scan == 'T':
                Y = [qhat(T = x, E = E, Q = Q, parameters = P2[i]) for x in X]
                XLabel = "T (GeV)"
            elif Scan == 'E':
                Y = [qhat(T = T, E = x, Q = Q, parameters = P2[i]) for x in X]
                XLabel = "E (GeV)"
            elif Scan == 'Q':
                Y = [qhat(T = T, E = E, Q = x, parameters = P2[i]) for x in X]
                XLabel = "Q (GeV)"
            elif Scan == 'EQ':
                Y = [qhat(T = T, E = x, Q = x, parameters = P2[i]) for x in X]
                XLabel = "E (GeV)"
            else:
                print('Error!  Illegal scan variable')

            AllY2.append(Y)

    if DoJet == True:
        JetBox1X = [0.170, 0.170, 0.386, 0.386, 0.170]
        JetBox1Y = [3.4, 5.8, 5.8, 3.4, 3.4]
        JetBox2X = [0.340, 0.340, 0.486, 0.486, 0.340]
        JetBox2Y = [2.3, 5.1, 5.1, 2.3, 2.3]
        JetPointX = [0.365, 0.460]
        JetPointY = [4.6, 3.7]
        JetErrorY = [1.2, 1.4]

        # axes.plot(JetBox1X, JetBox1Y, 'k-.')
        # axes.plot(JetBox2X, JetBox2Y, 'k-.')
        axes.errorbar(JetPointX, JetPointY, fmt = 'ko', yerr = JetErrorY, label = "JET Collaboration")
        axes.legend(loc = "lower left", fontsize = 20)

    axes.text(0.95, 0.95, ExtraText, transform = axes.transAxes, ha = 'right', va = 'top', fontsize = 20)
    axes.text(0.95, 0.88, Type, transform = axes.transAxes, ha = 'right', va = 'top', fontsize = 20)

    axes.set_ylim(bottom = 0)
    axes.set_xlabel(XLabel, fontsize = 20)
    axes.set_ylabel(r'$\hat{q}/T^3$', fontsize = 20)
    axes.tick_params(axis = 'x', labelsize = 20)
    axes.tick_params(axis = 'y', labelsize = 20)

    plt.tight_layout()
    tag = AllData['tag']
    figure.savefig(f'result/{tag}/plots/{Prefix}QHat{Suffix}.pdf', dpi = 192)
    figure.savefig(f'result/{tag}/plots/{Prefix}QHat{Suffix}.png', dpi = 192)

    AllY1 = np.array(AllY1)
    AllY2 = np.array(AllY2)

    Y105 = []
    Y150 = []
    Y195 = []
    Y205 = []
    Y250 = []
    Y295 = []

    for i in range(0, AllY1.shape[1]):
        Y1 = np.sort(AllY1[:,i])
        Y105.append(Y1[int(Y1.shape[0]*0.05)])
        Y150.append(Y1[int(Y1.shape[0]*0.50)])
        Y195.append(Y1[int(Y1.shape[0]*0.95)])
    for i in range(0, AllY2.shape[1]):
        Y2 = np.sort(AllY2[:,i])
        Y205.append(Y2[int(Y2.shape[0]*0.05)])
        Y250.append(Y2[int(Y2.shape[0]*0.50)])
        Y295.append(Y2[int(Y2.shape[0]*0.95)])

    figure, axes = plt.subplots(figsize = (5, 5))

    axes.plot(X, Y105, '-', color= "blue", label = args.NominalLabel)
    axes.plot(X, Y150, '.', color = "blue")
    axes.plot(X, Y195, '-', color = "blue")
    axes.plot(X, Y205, '-', color = "orange", label = args.AlternateLabel)
    axes.plot(X, Y250, '.', color = "orange")
    axes.plot(X, Y295, '-', color = "orange")

    axes.text(0.95, 0.95, ExtraText, transform = axes.transAxes, ha = 'right', va = 'top', fontsize = 20)
    axes.text(0.95, 0.88, Type, transform = axes.transAxes, ha = 'right', va = 'top', fontsize = 20)
    axes.text(0.95, 0.81, 'Median and 90% range', transform = axes.transAxes, ha = 'right', va = 'top', fontsize = 20)

    axes.set_xlabel(XLabel)
    axes.set_ylabel(r'$\hat{q}/T^3$')

    if DoJet == True:
        axes.errorbar(JetPointX, JetPointY, fmt = 'ko', yerr = JetErrorY, label = "JET Collaboration")

    if DoJet == True or P2 != [[]]:
        axes.legend(loc = "lower left", fontsize = 20)

    plt.tight_layout()
    tag = AllData['tag']
    figure.savefig(f'result/{tag}/plots/{Prefix}QHatRange{Suffix}.pdf', dpi = 192)


Posterior = MCMCSamples[ np.random.choice(range(len(MCMCSamples)), 5000), :]
Posterior2 = [[]] if args.Alternate == '' else MCMCSamples2[ np.random.choice(range(len(MCMCSamples2)), 5000), :]

PlotQHat(T = 0.2, E = 100, Q = 100, Scan = 'T', P1 = Posterior, P2 = Posterior2, Type = "Posterior", Prefix = args.Prefix, Suffix = "Posterior_T_E100")
PlotQHat(T = 0.2, E = 100, Q = 100, Scan = 'T', P1 = Posterior, P2 = Posterior2, Type = "Posterior", Prefix = args.Prefix, Suffix = "Posterior_T_E100_Jet", DoJet = True)
PlotQHat(T = 0.2, E = 100, Q = 100, Scan = 'T', P1 = Posterior, P2 = Posterior2, Type = "Posterior", Prefix = args.Prefix, Suffix = "PosteriorG_T_E100", DoGreen = True)
PlotQHat(T = 0.2, E = 100, Q = 100, Scan = 'T', P1 = Posterior, P2 = Posterior2, Type = "Posterior", Prefix = args.Prefix, Suffix = "PosteriorG_T_E100_Jet", DoJet = True, DoGreen = True)
PlotQHat(T = 0.3, E = 100, Q = 100, Scan = 'E', P1 = Posterior, P2 = Posterior2, Type = "Posterior", Prefix = args.Prefix, Suffix = "Posterior_E_T0.3")
PlotQHat(T = 0.2, E = 20,  Q = 20,  Scan = 'T', P1 = Posterior, P2 = Posterior2, Type = "Posterior", Prefix = args.Prefix, Suffix = "Posterior_T_E20")
PlotQHat(T = 0.2, E = 10,  Q = 10,  Scan = 'T', P1 = Posterior, P2 = Posterior2, Type = "Posterior", Prefix = args.Prefix, Suffix = "Posterior_T_E10")

Design = AllData["design"]

PlotQHat(T = 0.2, E = 100, Q = 100, Scan = 'T', P1 = Design, Type = "Design", Suffix = "Design_T_E100")
PlotQHat(T = 0.3, E = 100, Q = 100, Scan = 'E', P1 = Design, Type = "Design", Suffix = "Design_E_T0.3")



