"""
Reference implementation (numpy) of the 3+1 equations of a complex Proca field in full 3D, as implemented in
Source/Matter/ProcaField.hpp. Signature (-,+,+,+), K_ij = -nabla_i n_j (GRChombo convention).

  L = -1/4 F_ab Fbar^ab - 1/2 mu^2 A_a Abar^a          (the free Proca field of Spherical_BS_Evolver's ProcaStar)

Variables (Zilhao, Witek & Cardoso 2015), each complex:
  X_phi = -n^a A_a,   X_i = gamma_i^a A_a,   E^i = gamma^{ij} F_{ja} n^a,   Z (damps the Gauss constraint)
Equations:
  d_t X_i   = L_beta X_i - alpha (E_i + D_i X_phi) - X_phi D_i alpha
  d_t E^i   = L_beta E^i + alpha K E^i + D_j(alpha F^{ij}) + alpha mu^2 X^i + alpha D^i Z,   F_ij = d_i X_j - d_j X_i
  d_t X_phi = L_beta X_phi + alpha K X_phi - alpha D_i X^i - X^i D_i alpha - alpha Z
  d_t Z     = L_beta Z + alpha (D_i E^i + mu^2 X_phi - kappa Z)
(D_j(alpha F^{ij}) = (1/sqrt(gamma)) d_j(sqrt(gamma) alpha F^{ij}) = eps^{ijk} D_j(alpha B_k).)
Sources (sums over the real and imaginary parts, i.e. Re(a conj(b)) for products), B^2 = F_ij Fbar^ij / 2:
  rho  = (E_i Ebar^i + B^2)/2 + mu^2 (|X_phi|^2 + X_i Xbar^i)/2
  S_i  = Re(F_ij Ebar^j) + mu^2 Re(X_phi Xbar_i)
  S_ij = Re(-E_i Ebar_j + F_ik Fbar_jl gamma^kl) + mu^2 Re(X_i Xbar_j)
         + gamma_ij [E^2/2 - B^2/2 - mu^2 (X_k Xbar^k - |X_phi|^2)/2]
Gauss constraint C = D_i E^i + mu^2 X_phi. Number density (normal observers) n = -n_a j^a = Im(E^i Xbar_i) with j^a = Im(F^{ab} Abar_b), positive for
the stars of the 1D code; the particle number is int n sqrt(gamma) d^3x (the 1D noether_charge).
"""
import numpy as np


def proca_rhs_and_sources(mu, kappa, alpha, dalpha, beta, dbeta, gamma, dgamma, K, V, dV, ddX):
    """alpha, dalpha[k]; beta[i], dbeta[k, i] = d_k beta^i; gamma[i, j], dgamma[k, i, j] = d_k gamma_ij; K[i, j].
    V = dict(X (3,), E (3,) upper index, Xphi, Z), complex; dV = dict(X[k, i] = d_k X_i, E[k, i] = d_k E^i, Xphi[k],
    Z[k]); ddX[k, l, i] = d_k d_l X_i. Returns (rhs dict, sources dict)."""
    gi = np.linalg.inv(gamma)
    dgi = np.array([-gi @ dgamma[k] @ gi for k in range(3)])
    dlnsg = np.array([0.5 * np.trace(gi @ dgamma[k]) for k in range(3)])  # d_k ln sqrt(gamma)
    trK = np.trace(gi @ K)
    X, E, P, Z = V['X'], V['E'], V['Xphi'], V['Z']
    dX, dE, dP, dZ = dV['X'], dV['E'], dV['Xphi'], dV['Z']

    X_up = gi @ X
    dX_up = np.array([dgi[k] @ X + gi @ dX[k] for k in range(3)])          # [k, i] = d_k X^i
    divX = np.trace(dX_up) + X_up @ dlnsg
    divE = np.trace(dE) + E @ dlnsg
    E_dn = gamma @ E

    F = dX - dX.T                                                         # F[a, b] = d_a X_b - d_b X_a
    F_up = gi @ F @ gi
    dF = ddX - np.transpose(ddX, (0, 2, 1))                               # [k, a, b]
    dF_up = np.array([dgi[k] @ F @ gi + gi @ dF[k] @ gi + gi @ F @ dgi[k] for k in range(3)])
    # D_j(alpha F^{ij}) = alpha (d_j F^{ij} + F^{ij} d_j ln sqrt(gamma)) + F^{ij} d_j alpha
    divF = np.array([sum(dF_up[j, i, j] for j in range(3)) for i in range(3)]) + F_up @ dlnsg
    curl = alpha * divF + F_up @ dalpha

    rhs = {}
    rhs['X'] = beta @ dX + dbeta @ X - alpha * (E_dn + dP) - P * dalpha   # (dbeta @ X)_i = d_i beta^j X_j
    rhs['E'] = (beta @ dE - E @ dbeta + alpha * trK * E + alpha * mu**2 * X_up + alpha * gi @ dZ + curl)
    rhs['Xphi'] = beta @ dP + alpha * trK * P - alpha * divX - X_up @ dalpha - alpha * Z
    rhs['Z'] = beta @ dZ + alpha * (divE + mu**2 * P - kappa * Z)

    re = lambda a: np.real(a)
    E2 = re(E_dn @ np.conj(E))
    B2 = 0.5 * re(np.sum(F * np.conj(F_up)))
    X2 = re(X @ np.conj(X_up))
    P2 = abs(P)**2
    src = {}
    src['rho'] = 0.5 * (E2 + B2) + 0.5 * mu**2 * (P2 + X2)
    src['Si'] = re(F @ np.conj(E)) + mu**2 * re(P * np.conj(X))
    FF = re(F @ gi @ np.conj(F).T)                                        # F_ik Fbar_jl gamma^kl
    src['Sij'] = (re(-np.outer(E_dn, np.conj(E_dn))) + FF + mu**2 * re(np.outer(X, np.conj(X)))
                  + gamma * (0.5 * E2 - 0.5 * B2 - 0.5 * mu**2 * (X2 - P2)))
    src['Sij'] = 0.5 * (src['Sij'] + src['Sij'].T)
    src['S'] = np.trace(gi @ src['Sij'])
    src['Gauss'] = divE + mu**2 * P
    src['n'] = np.imag(E @ np.conj(X))
    return rhs, src
