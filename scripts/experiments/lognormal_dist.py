from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np

scale, s, loc = np.exp(10), 1, 10
x = np.linspace(lognorm.ppf(0.01, s, loc, scale), lognorm.ppf(0.99, s, loc, scale), 100)
y = lognorm.pdf(x, s=s, loc=loc, scale=scale)
data = lognorm.rvs(s=s, loc=loc, scale=scale, size=10000)
fig, ax = plt.subplots(2, 1)
ax[0].plot(x, y, "r-", lw=5, alpha=0.6, label="logno rmal pdf")
ax[0].hist(
    data,
    density=True,
    histtype="stepfilled",
    alpha=0.2,
    # bins=np.linspace(loc, loc + 10 * scale, 100),
)
ax[1].hist(np.log(data - loc), density=True, histtype="stepfilled", alpha=0.2)
plt.show()
