import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm
from scipy.optimize import curve_fit
import numpy as np

plt.style.use("ggplot")

radii_df = pd.read_csv("data/fiber_radii.csv")
fig, axs = plt.subplots(8, 2, sharex=True, sharey=True)
# fig.set_tight_layout(True)

# To figure out the sample names:
sampleKeys = [key for key in radii_df.keys() if key.endswith(".csv")]
sampleNames = [
    key.split("_")[0] for key in radii_df.keys() if key.endswith(".csv")
]
sampleNames = list(dict.fromkeys(sampleNames))
nameKeys = [
    [key for key in sampleKeys if sampleName in key]
    for sampleName in sampleNames
]

axis_dict = {
    "225AMA": [0, 0],
    "225AMB": [1, 0],
    "225ASA": [2, 0],  # no data was processed for this one
    "225ASB": [3, 0],
    "231AMA": [4, 0],
    "231AMB": [5, 0],
    "231ASA": [6, 0],
    "231ASB": [7, 0],
    "PV16-216MA": [0, 1],
    "PV16-216MB": [1, 1],
    "PV16-216SA": [2, 1],
    "PV16-216SB": [3, 1],
    "PV16-481MA": [4, 1],
    "PV16-481MB": [5, 1],
    "PV16-481SA": [6, 1],
    "PV16-481SB": [7, 1],
}

radii = radii_df.pop("Radius")
nameData = [radii_df[keys] for keys in nameKeys]
binCenters = np.array(radii)
maxHist = binCenters[-1]
binWidth = binCenters[1] - binCenters[0]
plotCutoff = 20
maxNormHist = 0
fitLognorm = True
fitGamma = True
colorGamma = "black"
colorLognorm = "blue"
linestyleGamma = "--"
linestyleLognorm = "-."
if fitGamma:
    gammaParamsDict = {}
    gammaRSquaredDict = {}
    p0 = [1.1, 0.2]
if fitLognorm:
    lognormParamsDict = {}
    lognormRSquaredDict = {}
    p0 = [1.1, 0.2]
for keys, cols in zip(nameKeys, nameData):
    key = keys[0]
    if "PV16-481SB" in key:
        print("here")
    sampleName, magnification, locName, _, _ = tuple(key.split("_"))
    # to choose in which axis to plot
    ax = axs[axis_dict[sampleName][0]][axis_dict[sampleName][1]]
    col = cols.sum(axis=1)
    if magnification == "1500x":
        isToPlot = True
        pixelSize = 10.0 / 150.0  # um/px
        scaleFactor = 9
        color = "red"
    elif magnification == "1000x":
        isToPlot = True
        pixelSize = 10.0 / 100.0  # um/px
        scaleFactor = 4
        color = "green"
    elif magnification == "500x":
        isToPlot = True
        pixelSize = 50.0 / 250.0  # um/px
        scaleFactor = 1
        color = "blue"
    else:
        pixelSize = 1.0
        color = "darkgray"
    linestyle = "-"
    hist = np.array(col)
    area = hist.sum() * binWidth
    normHist = hist / area
    if normHist.max() > maxNormHist:
        maxNormHist = normHist.max()

    loc = 0.0
    if fitGamma:
        f = lambda x, a, scale: gamma.pdf(x, a=a, scale=scale)
        a_gamma, scale_gamma = curve_fit(
            f, binCenters, normHist, maxfev=10000, p0=p0
        )[0]
        gammaParamsDict[sampleName] = [a_gamma, loc, scale_gamma]
        residuals = normHist - f(binCenters, a_gamma, scale_gamma)
        print(f"Gamma parmaeters are {a_gamma=}, {loc=}, {scale_gamma=}")
        ss_res = np.sum((residuals) ** 2)
        ss_tot = np.sum((normHist - np.mean(normHist)) ** 2)
        gammaRSquaredDict[sampleName] = 1 - (ss_res / ss_tot)
        print(gammaRSquaredDict[sampleName])
    if fitLognorm:
        f = lambda x, s, scale: lognorm.pdf(x, s=s, scale=scale)
        s_lognorm, scale_lognorm = curve_fit(
            f, binCenters, normHist, maxfev=10000, p0=p0
        )[0]
        lognormParamsDict[sampleName] = [s_lognorm, loc, scale_lognorm]
        residuals = normHist - f(binCenters, s_lognorm, scale_lognorm)
        print(f"Lognorm parmaeters are {s_lognorm=}, {loc=}, {scale_lognorm=}")
        ss_res = np.sum((residuals) ** 2)
        ss_tot = np.sum((normHist - np.mean(normHist)) ** 2)
        lognormRSquaredDict[sampleName] = 1 - (ss_res / ss_tot)
        print(lognormRSquaredDict[sampleName])

    if isToPlot:
        ax.plot(
            binCenters[:plotCutoff],
            normHist[:plotCutoff] * 100,
            color=color,
            alpha=0.2,
            linestyle=linestyle,
            marker=".",
            # label=magnification+" "+locName
        )

        D = np.linspace(0, binCenters[plotCutoff], 1000)
        if fitGamma:
            ax.plot(
                D,
                gamma.pdf(D, a=a_gamma, loc=loc, scale=scale_gamma) * 100,
                color=colorGamma,
                alpha=1.0,
                linestyle=linestyleGamma,
                # marker='.',
                label=magnification + " gamma " + "agg",
            )
        if fitLognorm:
            ax.plot(
                D,
                lognorm.pdf(D, s=s_lognorm, loc=loc, scale=scale_lognorm)
                * 100,
                color=colorLognorm,
                alpha=1.0,
                linestyle=linestyleLognorm,
                # marker='.',
                label=magnification + " lognorm " + "agg",
            )


for sampleName, index in axis_dict.items():
    ax = axs[index[0]][index[1]]  # to choose in which axis to plot
    ax.grid()
    ax.set_title(sampleName, fontsize=8, pad=-14)
    ax.set(ylim=(0, maxNormHist * 110))
    ax.legend(loc="upper right", fontsize=4)

axs[4, 0].set_ylabel("              Abs. Count ()", fontsize=8)
axs[4, 1].set_ylabel("              Abs. Count ()", fontsize=8)
axs[7, 0].set_xlabel("Approx Radius (" + "\u03bc" + "m)", fontsize=8)
axs[7, 1].set_xlabel("Approx Radius (" + "\u03bc" + "m)", fontsize=8)
dist = "gamma" if fitGamma else " "
dist += " and " if fitGamma and fitLognorm else " "
dist += "lognorm" if fitLognorm else " "
fig.suptitle(f"Fiber radii distribution with {dist} fit", fontsize=12)
plt.savefig(f"media/fiber_radii_{dist}_fit.pdf")
plt.show()

if fitGamma:
    gammaParamsDf = pd.DataFrame.from_dict(gammaParamsDict)
    gammaParamsDf.index = ["a", "loc", "scale"]
    gammaParamsDf.to_csv("data/fiber_radii_gamma_params.csv", index=True)
if fitLognorm:
    lognormParamsDf = pd.DataFrame.from_dict(lognormParamsDict)
    lognormParamsDf.index = ["s", "loc", "scale"]
    lognormParamsDf.to_csv("data/fiber_radii_lognorm_params.csv", index=True)
if fitGamma:
    gammaRSquaredDf = pd.DataFrame.from_dict(gammaRSquaredDict, orient="index")
    gammaRSquaredDf.to_csv("data/fiber_radii_gamma_r2.csv", index=True)
if fitLognorm:
    lognormRSquaredDf = pd.DataFrame.from_dict(
        lognormRSquaredDict, orient="index"
    )
    lognormRSquaredDf.to_csv("data/fiber_radii_lognorm_r2.csv", index=True)

rSquaredDf = pd.concat([lognormRSquaredDf, gammaRSquaredDf], axis=1)
rSquaredDf.columns = ["lognorm", "gamma"]

fig, ax = plt.subplots()
rSquaredDf.plot(kind="barh", ax=ax)
ax.set_title(f"R-squared values for {dist} fit")
rSquaredData = rSquaredDf.to_numpy()
ax.set_xlim(np.min(rSquaredData) - 0.1, np.max(rSquaredData) + 0.02)
plt.tight_layout()
plt.savefig(f"media/fiber_radii_{dist}_fit_r2.pdf")
plt.show()
