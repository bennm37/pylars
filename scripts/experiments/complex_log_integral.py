import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


# def integrate(func, curve, deriv):
#     def integrand(s):
#         result = func(curve(s)) * deriv(s)
#         return result

#     return quad(integrand, 0, 1, complex_func=True)[0]


# k = 100
# curve = lambda t: k * np.exp(2j * np.pi * t)
# deriv = lambda t: 2j * k * np.pi * np.exp(2j * np.pi * t)
# func = lambda z: 1 / (z + 1)
# result = integrate(func, curve, deriv)
# print(result)

x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
plt.imshow(np.imag(np.log(Z)))
plt.show()
