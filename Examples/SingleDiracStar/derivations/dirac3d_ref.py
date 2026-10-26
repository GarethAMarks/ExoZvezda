"""
Reference (Python) implementation of the 3+1 Dirac equations used by Source/Matter/DiracField.hpp in ExoZvezda (full
3D, no symmetry assumed). Conventions as in Spherical_BS_Evolver's DiracStar, signature (-,+,+,+):

  * (gamma^a nabla_a + mu) psi = 0, {gamma^a, gamma^b} = 2 eta^{ab}, gamma^a = -i gamma_D^a (standard Dirac
    representation, Pauli matrices sigma_x, sigma_y, sigma_z as usual), psibar = psi^dag gamma_D^0,
  * Gamma_mu = (1/8) omega_{mu ab} [gamma^a, gamma^b],  omega_mu^a_b = e^a_nu nabla_mu e_b^nu,
  * T_ab = (1/4)(X_ab + X_ba),  X_ab = psibar gamma_a nabla_b psi - (nabla_a psi)^dag gamma_D^0 gamma_b psi,
  * tetrad: e_0 = n (time gauge) and the Cholesky (Gram-Schmidt) triad: gamma = L L^T with L lower triangular, the
    co-triad theta^a_i = L_ia and the triad e_a^i = (L^{-T})_ia, so e_1 is along x, e_2 in the xy plane. It is a
    smooth algebraic function of gamma_ij (no extra variables), covariant under the coordinate reflections x -> -x,
    y -> -y, z -> -z, and equal to the symmetric triad of the 1D/2D codes when gamma_ij is conformally flat (so the
    initial spinor components are the same).
  * K_ij = -nabla_i n_j (d_t gamma_ij = -2 alpha K_ij + L_beta gamma_ij), as in GRChombo.

3+1 form:
  d_t psi = beta^k d_k psi - alpha Gamma_n psi + alpha gamma^0 gamma^b e_b^k (d_k psi + Gamma_k psi) + alpha mu gamma^0 psi,
  Gamma_k = (1/2) omega_{k0b} gamma^0 gamma^b + (1/4) omega_{kab} gamma^a gamma^b,   omega_{k0b} = K_kj e_b^j,
  Gamma_n = (1/2) n.omega_{0b} gamma^0 gamma^b + (1/4) n.omega_{ab} gamma^a gamma^b,  n.omega_{0b} = -e_b^j d_j alpha/alpha,
  n.omega_{ab} = [ e_a.(d_t e_b - L_beta e_b) - (a <-> b) ] / (2 alpha),
  omega_{kab} = gamma_ij e_a^i (d_k e_b^j + Gamma^j_kl e_b^l).
Derivatives of the triad: d L = L Phi(L^{-1} d gamma L^{-T}), Phi = lower triangle with half the diagonal, and
d E = -E (d L)^T E.

Matter: two spinors psi_A, psi_B (the m_j = +1/2 and -1/2 states of the spherical star), stored as the 8 complex
combinations D (DIRAC_BASIS below), each of which has a definite parity under x -> -x, y -> -y and z -> -z for a
reflection-symmetric configuration (so GRChombo's reflective boundaries can be used in any direction).
"""
import numpy as np

I = 1j
SIG = [np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], dtype=complex)]
Z2, I2 = np.zeros((2, 2), dtype=complex), np.eye(2, dtype=complex)


def block(a, b, c, d):
    return np.block([[a, b], [c, d]])


gD = [block(I2, Z2, Z2, -I2)] + [block(Z2, s, -s, Z2) for s in SIG]
GAM = [-I * g for g in gD]
ETA = np.diag([-1., 1., 1., 1.])
GAM_LOW = [ETA[a, a] * GAM[a] for a in range(4)]
BETA = gD[0]
G5 = I * gD[0] @ gD[1] @ gD[2] @ gD[3]
MREF = [G5 @ gD[a + 1] for a in range(3)]        # spinor matrices of the reflections x, y, z -> minus themselves
TAU = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1., -1.])]
# reflections acting on (psi_A, psi_B): R_a = T_a (x) M_a, with the flavour matrices that leave the star invariant
RREF = [np.kron(-TAU[0], MREF[0]), np.kron(TAU[1], MREF[1]), np.kron(-TAU[2], MREF[2])]


def gg(a, b):
    return GAM[a] @ GAM[b]


# ---------------------------------------------------------------- basis of the 16 real grid variables
# D = V^T (psi_A, psi_B) with psi_A = (A0..A3), psi_B = (B0..B3); names give the directions in which D is odd
DIRAC_NAMES = ['e', 'x', 'y', 'z', 'xy', 'yz', 'xz', 'xyz']
_r = 1 / np.sqrt(2)
_comb = {'e': [(0, 1), (5, 1)], 'xy': [(0, 1), (5, -1)], 'yz': [(1, 1), (4, 1)], 'xz': [(1, 1), (4, -1)],
         'xyz': [(2, 1), (7, 1)], 'z': [(2, 1), (7, -1)], 'x': [(3, 1), (6, 1)], 'y': [(3, 1), (6, -1)]}
DIRAC_BASIS = np.zeros((8, 8))      # columns: D_k in terms of (A0..A3, B0..B3)
for k, name in enumerate(DIRAC_NAMES):
    for comp, sgn in _comb[name]:
        DIRAC_BASIS[comp, k] = sgn * _r


def to_spinors(D):
    """(psi_A, psi_B) from the 8 complex grid combinations"""
    P = DIRAC_BASIS @ D
    return P[:4], P[4:]


def from_spinors(pA, pB):
    return DIRAC_BASIS.T @ np.concatenate([pA, pB])


# ---------------------------------------------------------------- triad
def cholesky_triad(gamma, dgammas):
    """E (triad e_a^i as columns), co-triad theta = L^T, and dE for each d gamma in dgammas"""
    L = np.linalg.cholesky(gamma)
    Li = np.linalg.inv(L)
    E = Li.T
    out = []
    for dg in dgammas:
        S = Li @ dg @ Li.T
        Phi = np.tril(S, -1) + 0.5 * np.diag(np.diag(S))
        dL = L @ Phi
        out.append(-E @ dL.T @ E)
    return E, L.T, out


def christoffel(gamma, dgamma):
    gi = np.linalg.inv(gamma)
    return 0.5 * (np.einsum('il,jlk->ijk', gi, dgamma) + np.einsum('il,klj->ijk', gi, dgamma) -
                  np.einsum('il,ljk->ijk', gi, dgamma))


def gamma_dot(alpha, beta, dbeta, gamma, dgamma, K):
    return (-2 * alpha * K + np.einsum('k,kij->ij', beta, dgamma) + np.einsum('kj,ik->ij', gamma, dbeta) +
            np.einsum('ik,jk->ij', gamma, dbeta))


def connections(alpha, dalpha, beta, dbeta, gamma, dgamma, K, E, dE, Edot):
    G3 = christoffel(gamma, dgamma)
    om = np.zeros((3, 3, 3))
    for k in range(3):
        nab_e = dE[k] + np.einsum('jl,lb->jb', G3[:, k, :], E)
        om[k] = E.T @ gamma @ nab_e
    om0 = K @ E
    an0 = -(E.T @ dalpha) / alpha
    Lbe = np.einsum('l,lib->ib', beta, dE) - np.einsum('lb,li->ib', E, dbeta)
    W = E.T @ gamma @ (Edot - Lbe)
    anab = (W - W.T) / (2 * alpha)
    Gk = []
    for k in range(3):
        M = np.zeros((4, 4), dtype=complex)
        for b in range(3):
            M += 0.5 * om0[k, b] * gg(0, b + 1)
            for a in range(3):
                if a != b:
                    M += 0.25 * om[k, a, b] * gg(a + 1, b + 1)
        Gk.append(M)
    Gn = np.zeros((4, 4), dtype=complex)
    for b in range(3):
        Gn += 0.5 * an0[b] * gg(0, b + 1)
        for a in range(3):
            if a != b:
                Gn += 0.25 * anab[a, b] * gg(a + 1, b + 1)
    return Gk, Gn


def geometry(alpha, dalpha, beta, dbeta, gamma, dgamma, K):
    gd = gamma_dot(alpha, beta, dbeta, gamma, dgamma, K)
    E, th, d = cholesky_triad(gamma, [dgamma[0], dgamma[1], dgamma[2], gd])
    dE, Edot = np.array(d[:3]), d[3]
    Gk, Gn = connections(alpha, dalpha, beta, dbeta, gamma, dgamma, K, E, dE, Edot)
    return dict(E=E, th=th, Gk=Gk, Gn=Gn, beta=beta, alpha=alpha)


def spinor_rhs(mu, geo, psi, dpsi):
    """d_t psi for one spinor (dpsi[k] = d_k psi, k = x, y, z)"""
    a, E = geo['alpha'], geo['E']
    rhs = sum(geo['beta'][k] * dpsi[k] for k in range(3)) - a * geo['Gn'] @ psi + a * mu * GAM[0] @ psi
    for b in range(3):
        rhs += a * gg(0, b + 1) @ sum(E[k, b] * (dpsi[k] + geo['Gk'][k] @ psi) for k in range(3))
    return rhs


def spinor_T(geo, psi, dpsi, psi_t):
    a, E, beta = geo['alpha'], geo['E'], geo['beta']
    nab = [(psi_t - sum(beta[k] * dpsi[k] for k in range(3))) / a + geo['Gn'] @ psi]
    for b in range(3):
        nab.append(sum(E[k, b] * (dpsi[k] + geo['Gk'][k] @ psi) for k in range(3)))
    psibar = psi.conj() @ BETA
    X = np.array([[psibar @ GAM_LOW[i] @ nab[j] - nab[i].conj() @ BETA @ GAM_LOW[j] @ psi for j in range(4)]
                  for i in range(4)])
    return 0.25 * (X + X.T)


def dirac_rhs_and_sources(mu, alpha, dalpha, beta, dbeta, gamma, dgamma, K, D, dD):
    """RHS of the 8 complex grid variables and the sources of both spinors: rho, S_i, S_ij, n (and max |Im T|)"""
    geo = geometry(alpha, dalpha, beta, dbeta, gamma, dgamma, K)
    pA, pB = to_spinors(D)
    dA = [to_spinors(dD[k])[0] for k in range(3)]
    dB = [to_spinors(dD[k])[1] for k in range(3)]
    rA, rB = spinor_rhs(mu, geo, pA, dA), spinor_rhs(mu, geo, pB, dB)
    T = spinor_T(geo, pA, dA, rA) + spinor_T(geo, pB, dB, rB)
    th = geo['th']
    rho = T[0, 0].real
    Si = -(th.T @ T[1:, 0]).real
    Sij = (th.T @ T[1:, 1:] @ th).real
    n = np.real(pA.conj() @ pA + pB.conj() @ pB)
    return from_spinors(rA, rB), rho, Si, Sij, n, np.max(np.abs(T.imag)), geo


def star_D(g, f, nvec):
    """grid variables for the spherical star at a point with unit radial vector n (f, g physical, t = 0)"""
    A = np.concatenate([g * np.array([1, 0]), -I * f * (sum(nvec[i] * SIG[i] for i in range(3)) @ np.array([1, 0]))])
    B = np.concatenate([g * np.array([0, 1]), -I * f * (sum(nvec[i] * SIG[i] for i in range(3)) @ np.array([0, 1]))])
    return from_spinors(A, B) / np.sqrt(4 * np.pi)
