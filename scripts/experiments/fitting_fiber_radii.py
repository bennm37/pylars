import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma
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
r_squared_dict = {}
# sampleNames = list(dict.fromkeys(sampleNames)) # workaround to drop duplicates
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
histogramData = radii_df[sampleKeys]
binCenters = np.array(radii)
maxHist = binCenters[-1]
binWidth = binCenters[1] - binCenters[0]
plotCutoff = 20
for key, col in histogramData.items():
    sampleName, magnification, locName, _, _ = tuple(key.split("_"))
    # to choose in which axis to plot
    ax = axs[axis_dict[sampleName][0]][axis_dict[sampleName][1]]

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

    if locName == "i1":
        linestyle = "--"
    elif locName == "i2":
        linestyle = "-."
    elif locName == "i3":
        linestyle = ":"
    else:
        linestyle = "-"

    hist = np.array(col)
    area = hist.sum() * binWidth
    normHist = hist / area

    # f = lambda x, s, loc, scale: lognorm.pdf(x, s=s, loc=loc, scale=scale)
    f = lambda x, a, loc, scale: gamma.pdf(x, a=a, loc=loc, scale=scale)
    a, loc, scale = curve_fit(f, binCenters, normHist, maxfev=5000)[0]
    residuals = normHist - f(binCenters, a, loc, scale)
    ss_res = np.sum((residuals) ** 2)
    ss_tot = np.sum((normHist - np.mean(normHist)) ** 2)
    r_squared_dict[sampleName] = 1 - (ss_res / ss_tot)
    print(r_squared_dict[sampleName])
    print(a, loc, scale)

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
        ax.plot(
            D,
            gamma.pdf(D, a=a, loc=loc, scale=scale) * 100,
            color=color,
            alpha=1.0,
            linestyle=linestyle,
            # marker='.',
            label=magnification + " " + locName,
        )


for sampleName, index in axis_dict.items():
    ax = axs[index[0]][index[1]]  # to choose in which axis to plot
    ax.grid()
    ax.set_ylabel("Abs. Count ()", fontsize=8)
    ax.set_title(sampleName, fontsize=8, pad=-14)
    ax.legend(loc="upper right", fontsize=4)

axs[7, 0].set_xlabel("Approx Radius (" + "\u03bc" + "m)", fontsize=8)
axs[7, 1].set_xlabel("Approx Radius (" + "\u03bc" + "m)", fontsize=8)
plt.savefig("media/fiber_radii_gamma_fit.pdf")
plt.show()

fig, ax = plt.subplots()
r2Data = [r_squared_dict[sampleName] for sampleName in sampleNames]
ax.barh(sampleNames, r2Data)
ax.set_title("R-squared values for gamma fit")
ax.set_xlim(np.min(r2Data) - 0.1, np.max(r2Data) + 0.02)
plt.tight_layout()
plt.savefig("media/fiber_radii_gamma_fit_r2.pdf")
plt.show()
