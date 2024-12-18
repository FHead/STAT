import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import pickle
AllData = {}
with open('input/default.p', 'rb') as handle:
    AllData = pickle.load(handle)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Suffix', help = 'suffix to add to file name', type = str, default = '')
args = parser.parse_args()

if args.Suffix != '':
    args.Suffix = '_' + args.Suffix

import src
src.Initialize()

from src import lazydict, emulator
Emulator = emulator.Emulator.from_cache('HeavyIon')

from src import mcmc
chain = mcmc.Chain()
MCMCSamples = chain.load()

Examples = MCMCSamples[ np.random.choice(range(len(MCMCSamples)), 500), :]
TempPrediction = {"HeavyIon": Emulator.predict(Examples)}

SystemCount = len(AllData["systems"])
BinCount = len(AllData['observables'][0][1])

RC = int(np.sqrt(BinCount))
CC = int(np.ceil(BinCount / RC))

figure, axes = plt.subplots(figsize = (3 * CC, 3 * RC), nrows = RC, ncols = CC)

for s2 in range(0, RC * CC):
    ax = s2 % RC
    ay = int(np.floor(s2 / RC))
    axes[ax][ay].set_xlabel(r"$p_{T}$")
    axes[ax][ay].set_ylabel(r"$R_{AA}$")

    if s2 >= BinCount:
        axes[ax][ay].axis('off')
        continue

    S1 = AllData["systems"][0]
    O  = AllData["observables"][0][0]
    S2 = AllData["observables"][0][1][s2]

    DX = AllData["data"][S1][O][S2]['x']
    DY = AllData["data"][S1][O][S2]['y']
    DE = np.sqrt(AllData["data"][S1][O][S2]['yerr']['stat'][:,0]**2 + AllData["data"][S1][O][S2]['yerr']['sys'][:,0]**2)

    axes[ax][ay].set_title(AllData["observables"][0][1][s2])
    # axes[ax][ay].set_xscale('log')
    axes[ax][ay].set_ylim(0, 1.2)
    if len(DX) > 1:
        axes[ax][ay].plot([DX[0], DX[-1]], [1, 1], 'k-.')
    else:
        axes[ax][ay].plot([np.floor(DX[0] * 0.9), np.ceil(DX[0] * 1.1)], [1, 1], 'k-.')

    if DoAlternate == False:
        for i, y in enumerate(TempPrediction[S1][O][S2]):
            if len(DX) > 1:
                axes[ax][ay].plot(DX, y, 'b-', alpha=0.05, label="Nominal" if i==0 else '')
            else:
                axes[ax][ay].plot([np.floor(DX[0] * 0.9), np.ceil(DX[0] * 1.1)], [y[0], y[0]], 'b-', alpha=0.025, label = 'Nominal' if i == 0 else '')
    if DoAlternate == True:
        for i, y in enumerate(TempPrediction2[S1][O][S2]):
            if len(DX) > 1:
                axes[ax][ay].plot(DX, y, 'g-', alpha=0.05, label=args.AlternateLabel if i==0 else '')
            else:
                axes[ax][ay].plot([np.floor(DX[0] * 0.9), np.ceil(DX[0] * 1.1)], [y[0], y[0]], 'g-', alpha=0.025, label = 'Nominal' if i == 0 else '')
    axes[ax][ay].errorbar(DX, DY, yerr = DE, fmt='ro', label="Measurements")

plt.tight_layout()
tag = AllData['tag']
figure.savefig(f'result/{tag}/plots/ObservablePosterior{args.Suffix}.pdf', dpi = 192)
figure.savefig(f'result/{tag}/plots/ObservablePosterior{args.Suffix}.png', dpi = 192)
# figure.savefig(f'result/{tag}/plots/ObservablePosterior.png', dpi = 192)


