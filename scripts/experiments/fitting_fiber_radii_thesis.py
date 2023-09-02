import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm, chisquare
from scipy.optimize import curve_fit
import numpy as np

plt.style.use("ggplot")

radii_df = pd.read_csv("data/fiber_radii.csv")
fig, axs = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(3, 3)
# fig.set_tight_layout(True)

# To figure out the sample names:
sampleNames = ["225ASB", "PV16-216SB"]
sampleKeys = [
    key
    for key in radii_df.keys()
    if key.endswith(".csv")
    and any(sampleName in key for sampleName in sampleNames)
]
sampleNames = list(dict.fromkeys(sampleNames))
nameKeys = [
    [key for key in sampleKeys if sampleName in key]
    for sampleName in sampleNames
]

axis_dict = {
    "225ASB": 0,
    "PV16-216SB": 1,
}

radii = radii_df.pop("Radius")
nameData = [radii_df[keys] for keys in nameKeys]
binCenters = np.array(radii)
maxHist = binCenters[-1]
binWidth = binCenters[1] - binCenters[0]
plotCutoff = 20
maxNormHist = 0
fitLognorm = True
fitGamma = False
colorGamma = "black"
colorLognorm = "blue"
if fitGamma:
    gammaParamsDict = {}
    gammaChiSquaredDict = {}
    p0 = [1.1, 0.2]
if fitLognorm:
    lognormParamsDict = {}
    lognormChiSquaredDict = {}
    p0 = [1.1, 0.2]
for keys, cols in zip(nameKeys, nameData):
    key = keys[0]
    if "PV16-481SB" in key:
        print("here")
    sampleName, magnification, locName, _, _ = tuple(key.split("_"))
    # to choose in which axis to plot
    ax = axs[axis_dict[sampleName]]
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
        expected = f(binCenters, a_gamma, scale_gamma)
        expected *= np.sum(hist) / np.sum(expected)
        expected = expected[1:]
        index = np.min(np.where(expected < 5))
        expected[index] = np.sum(expected[index:])
        expected = expected[: index + 1]
        observed = hist[1:]
        observed[index] = np.sum(observed[index:])
        observed = observed[: index + 1]

        chiSquaredStat, pValue = chisquare(observed, expected)
        print(f"Gamma parmaeters are {a_gamma=}, {loc=}, {scale_gamma=}")
        gammaChiSquaredDict[sampleName] = pValue
        print(gammaChiSquaredDict[sampleName])
    if fitLognorm:
        f = lambda x, s, scale: lognorm.pdf(x, s=s, scale=scale)
        s_lognorm, scale_lognorm = curve_fit(
            f, binCenters, normHist, maxfev=10000, p0=p0
        )[0]
        lognormParamsDict[sampleName] = [s_lognorm, loc, scale_lognorm]
        expected = f(binCenters, s_lognorm, scale_lognorm)
        expected *= np.sum(hist) / np.sum(expected)
        expected = expected[1:]
        index = np.min(np.where(expected < 5))
        expected[index] = np.sum(expected[index:])
        expected = expected[: index + 1]
        observed = hist[1:]
        observed[index] = np.sum(observed[index:])
        observed = observed[: index + 1]
        chiSquaredStat, pValue = chisquare(observed, expected)
        print(f"Lognorm parmaeters are {s_lognorm=}, {loc=}, {scale_lognorm=}")
        lognormChiSquaredDict[sampleName] = pValue
        print(lognormChiSquaredDict[sampleName])

    if isToPlot:
        ax.plot(
            binCenters[:plotCutoff],
            normHist[:plotCutoff],
            color=color,
            alpha=0.2,
            linestyle=linestyle,
            marker=".",
            label=magnification,
        )

        D = np.linspace(0, binCenters[plotCutoff], 1000)
        if fitGamma:
            ax.plot(
                D,
                gamma.pdf(D, a=a_gamma, loc=loc, scale=scale_gamma),
                color=colorGamma,
                alpha=1.0,
                linestyle=linestyle,
                # marker='.',
                label=magnification + " gamma " + "agg",
            )
        if fitLognorm:
            ax.plot(
                D,
                lognorm.pdf(D, s=s_lognorm, loc=loc, scale=scale_lognorm),
                color=colorLognorm,
                alpha=1.0,
                linestyle=linestyle,
                # marker='.',
                label="Lognorm PDF",
            )


for sampleName, index in axis_dict.items():
    ax = axs[index]  # to choose in which axis to plot
    ax.grid()
    if "PV" in sampleName:
        title = "Standard"
    else:
        title = "Alternative"
    ax.set_title(title, fontsize=8, pad=-14)
    ax.set(ylim=(0, maxNormHist * 1.1))
    ax.legend(loc="upper right")

axs[0].set_ylabel("Density", fontsize=8)
axs[1].set_ylabel("Density", fontsize=8)
axs[1].set_xlabel("Radius (" + "$\mu$" + "m)", fontsize=8)
dist = "gamma" if fitGamma else " "
dist += " and " if fitGamma and fitLognorm else " "
dist += "lognorm" if fitLognorm else " "
plt.tight_layout()
plt.savefig(f"media/fiber_radii_{dist.strip()}_fit.pdf", bbox_inches="tight")
plt.show()

if fitGamma:
    gammaParamsDf = pd.DataFrame.from_dict(gammaParamsDict)
    gammaParamsDf.to_csv("data/fiber_radii_gamma_params.csv", index=False)
if fitLognorm:
    lognormParamsDf = pd.DataFrame.from_dict(lognormParamsDict)
    lognormParamsDf.to_csv("data/fiber_radii_lognorm_params.csv", index=False)
if fitGamma:
    gammaChiSquaredDf = pd.DataFrame.from_dict(
        gammaChiSquaredDict, orient="index"
    )
    gammaChiSquaredDf.to_csv("data/fiber_radii_gamma_r2.csv", index=True)
if fitLognorm:
    lognormChiSquaredDf = pd.DataFrame.from_dict(
        lognormChiSquaredDict, orient="index"
    )
    lognormChiSquaredDf.to_csv("data/fiber_radii_lognorm_r2.csv", index=True)

ChiSquaredDf = pd.concat([lognormChiSquaredDf, gammaChiSquaredDf], axis=1)
ChiSquaredDf.columns = ["lognorm", "gamma"]

fig, ax = plt.subplots()
ChiSquaredDf.plot(kind="barh", ax=ax)
ax.set_title(f"Chi-squared values for {dist} fit")
ChiSquaredData = ChiSquaredDf.to_numpy()
ax.set_xlim(np.min(ChiSquaredData) * 0.5, np.max(ChiSquaredData) * 1.1)
ax.set_xscale("log")
plt.tight_layout()
plt.savefig(f"media/fiber_radii_{dist}_fit_r2.pdf")
plt.show()
