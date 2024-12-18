import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import yaml
from pathlib import Path

import pickle
AllData = {}
with open('input/default.p', 'rb') as handle:
    AllData = pickle.load(handle)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--Config", help = "plot configuration file", type = str, default = "yaml/PlotConfig.yaml")
parser.add_argument("--Alternate", help = "whether to plot a second collection", type = str, default = '')
parser.add_argument('--AlternateLabel', help = 'label for alternate collection', type = str, default = '')
parser.add_argument('--Suffix', help = 'suffix to add to file name', type = str, default = '')
args = parser.parse_args()

DoAlternate = False
if args.Alternate != '':
    DoAlternate = True

if args.Suffix != '':
    args.Suffix = '_' + args.Suffix

fontsize = 22

import src
src.Initialize()

from src import lazydict, emulator
Emulator = emulator.Emulator.from_cache('HeavyIon')

from src import mcmc
chain = mcmc.Chain()
MCMCSamples = chain.load()

Examples = MCMCSamples[ np.random.choice(range(len(MCMCSamples)), 500), :]
TempPrediction = {"HeavyIon": Emulator.predict(Examples)}

if DoAlternate == True:
    chain = mcmc.Chain(path = Path(f'result/{args.Alternate}/mcmc_chain.hdf'))
    MCMCSamples2 = chain.load()
    Examples2 = MCMCSamples2[ np.random.choice(range(len(MCMCSamples2)), 500), :]
    TempPrediction2 = {"HeavyIon": Emulator.predict(Examples2)}


# Function to plot error boxes
def MakeErrorBoxes(ax,xdata,ydata,xerror,yerror,fc='r',ec='None',alpha=0.5):
    # Create list for all the error patches
    errorboxes = []

    # print(xdata)
    # print(ydata)
    # print(xerror)
    # print(yerror)

    # Loop over data points; create box from errors at each point
    for xc,yc,xe,ye in zip(xdata,ydata,xerror.T,yerror.T):
        rect = Rectangle((xc-xe,yc-ye),xe*2,ye*2)
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes,facecolor=fc,alpha=alpha,edgecolor=ec)

    # Add collection to axes
    ax.add_collection(pc)

def MakePlot(Item):
    # Let's not remove things silently.  We should let fails fail for this
    # ToPlot = [x for x in ToPlot if x in AllData["observables"][0][1]]

    ToPlot     = Item['Key']
    Suffix     = Item['Suffix']
    Label      = Item['Label']
    Common     = Item['Common']
    Logx       = Item['Logx'] if 'Logx' in Item else []
    Tickx      = Item['Tickx'] if 'Tickx' in Item else [[]]
    XRange     = Item['XRange'] if 'XRange' in Item else []
    Guest      = Item['Guest'] if 'Guest' in Item else []
    GuestLabel = Item['GuestLabel'] if 'GuestLabel' in Item else []

    if Suffix != "":
        Suffix = "_" + Suffix

    if len(ToPlot) == 0:
        print("Nothing valid specified")
        return

    SystemCount = len(AllData["systems"])
    BinCount = len(ToPlot)

    figure = plt.figure(figsize = (3 * BinCount + 1, 3 + 1))
    gridspec = figure.add_gridspec(1, BinCount, hspace = 0, wspace = 0)
    axes = gridspec.subplots(sharex = 'col', sharey = 'row')

    if BinCount == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.set_xlabel(r"$p_{T}$ (GeV/c)", fontsize = fontsize)
        ax.set_ylabel(r"$R_{AA}$", fontsize = fontsize)

        S1 = AllData["systems"][0]
        O  = AllData["observables"][0][0]
        S2 = ToPlot[i]

        if i < len(Label):
            ax.text(0.05, 0.94, Label[i],
                verticalalignment='top', horizontalalignment='left',
                transform = ax.transAxes, fontsize = fontsize)
        if i < len(GuestLabel):
            ax.text(0.95, 0.04, GuestLabel[i],
                verticalalignment='bottom', horizontalalignment='right',
                transform = ax.transAxes, fontsize = fontsize, color = 'red')

        ax.set_ylim([0, 1.35])
        ax.label_outer()

        if S2 not in AllData["data"][S1][O]:
            ax.axis('off')
            continue

        DX = AllData["data"][S1][O][S2]['x']
        DY = AllData["data"][S1][O][S2]['y']
        DE = np.sqrt(AllData["data"][S1][O][S2]['yerr']['stat'][:,0]**2 + AllData["data"][S1][O][S2]['yerr']['sys'][:,0]**2)
        DEStat = AllData["data"][S1][O][S2]['yerr']['stat'][:,0]
        DESys = AllData["data"][S1][O][S2]['yerr']['sys'][:,0]

        if DoAlternate == True:
            for j, y in enumerate(TempPrediction2[S1][O][S2]):
                if len(DX) > 1:
                    ax.plot(DX, y, '-', alpha=0.05, color='goldenrod', label = args.AlternateLabel if j == 0 else '')
                else:
                    ax.plot([np.floor(DX[0] * 0.9), np.ceil(DX[0] * 1.1)], [y[0], y[0]], '-', color='goldenrod', alpha=0.05, label = args.AlternateLabel if j == 0 else '')
        else:
            for j, y in enumerate(TempPrediction[S1][O][S2]):
                if len(DX) > 1:
                    ax.plot(DX, y, '-', color='royalblue', alpha=0.025, label = 'Nominal' if j == 0 else '')
                else:
                    ax.plot([np.floor(DX[0] * 0.9), np.ceil(DX[0] * 1.1)], [y[0], y[0]], '-', color='royalblue', alpha=0.025, label = 'Nominal' if j == 0 else '')

        if i < len(Guest) and Guest[i] != '':
            ExtraData = np.loadtxt(Guest[i])

            MakeErrorBoxes(ax, ExtraData[:,0], ExtraData[:,2], ExtraData[:,0] * 0.05, ExtraData[:,4], 'red', 'None', 0.5)
            ax.errorbar(ExtraData[:,0], ExtraData[:,2], xerr = ExtraData[:,0] * 0.05, yerr = ExtraData[:,3], fmt = 'o', color = 'red')

        RangeX = DX[-1] - DX[0]
        if i < len(Logx) and Logx[i] == True:
            MakeErrorBoxes(ax, DX, DY, DX * 0.05, DESys, 'darkmagenta', 'None', 0.5)
            ax.errorbar(DX, DY, yerr = DEStat, xerr = DX * 0.05, fmt='o', color='darkmagenta')
        else:
            MakeErrorBoxes(ax, DX, DY, np.array([RangeX * 0.02] * len(DX)), DESys, 'darkmagenta', 'None', 0.5)
            ax.errorbar(DX, DY, yerr = DEStat, xerr = DX[-1] * 0.02, fmt='o', color='darkmagenta')
        ax.tick_params(axis = 'x', labelsize = fontsize)
        ax.tick_params(axis = 'y', labelsize = fontsize)

        if i < len(Logx) and Logx[i] == True:
            ax.set_xscale('log')
            ax.xaxis.set_major_formatter(ScalarFormatter())
        if 'Tickx' in Item:
            ax.get_xaxis().set_ticks(Tickx[i])
            ax.get_xaxis().set_minor_formatter(NullFormatter())
            ax.tick_params(axis='x', which='minor', bottom=False)

        if i < len(XRange) and XRange[i] != 'None':
            ax.set_xlim(XRange[i])

    axes[0].set_title(Common, loc = 'left', fontsize = fontsize)
    # axes[len(axes)-1].legend(loc = 'upper right')

    plt.tight_layout()
    tag = AllData['tag']
    figure.savefig(f'result/{tag}/plots/SmallObservablePosterior{args.Suffix}{Suffix}.pdf', dpi = 192)
    # figure.savefig(f'result/{tag}/plots/SmallObservablePosterior{args.Suffix}{Suffix}.png', dpi = 192)
    plt.close('all')


with open(args.Config, 'r') as stream:
    try:
        setup = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

for item in setup['ObservablePlot']:
    MakePlot(item)







