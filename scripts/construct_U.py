from pylars.numerics import va_orthogonalise, va_evaluate
import numpy as np
from scipy.sparse import csr_matrix

Z = np.exp(1j * np.linspace(0, 2 * np.pi, 100)).reshape(100, 1)
# NOTE paper notation and MATLAB code are different.
# In paper x is stacked (f_r,f_i, g_r, g_i) but in MATLAB
# stacked (f_r, g_r, f_i, g_i). This is convenient as then
# splitting the coefficient vector into real and imaginary
# parts is slightly easier. This uses paper notation.
m = len(Z)
hessenbergs, Q = va_orthogonalise(Z, 10)
basis, basis_deriv = va_evaluate(Z, hessenbergs)
z_conj = csr_matrix((Z.conj().reshape(m), (range(m), range(m))))

u_1 = np.real(z_conj @ basis_deriv - basis)
u_2 = -np.imag(z_conj @ basis_deriv - basis)
u_3 = np.real(basis_deriv)
u_4 = -np.imag(basis_deriv)
U = np.hstack((u_1, u_2, u_3, u_4))

v_1 = -np.imag(z_conj @ basis_deriv + basis)
v_2 = -np.real(z_conj @ basis_deriv + basis)
v_3 = -np.imag(basis_deriv)
v_4 = -np.real(basis_deriv)
V = np.hstack((v_1, v_2, v_3, v_4))

p_1 = np.real(4 * basis_deriv)
p_2 = -np.imag(4 * basis_deriv)
p_3 = np.zeros_likes(p_1)
pressure = np.hstack((p_1, p_2, p_3, p_3))

s_1 = np.imag(z_conj @ basis)
s_2 = np.real(z_conj @ basis)
s_3 = np.imag(basis)
s_4 = np.real(basis)
stream_fuction = np.hstack((s_1, s_2, s_3, s_4))
