"""
Numerical verification of the 3D Dirac equations in dirac3d_ref.py (run: python3 check_dirac3d.py, a few seconds).

A. The 3+1 right-hand side (Cholesky triad, triad derivatives in closed form) against the covariant 4D Dirac operator
   built directly from arbitrary alpha(t,x), beta^i(t,x), gamma_ij(t,x), psi(t,x) with finite differences:
       gamma^0 D psi + (d_t psi - RHS) / alpha = 0.
B. Reflections. For each of x -> -x, y -> -y, z -> -z the mirrored configuration (metric, gauge, K and
   (psi_A, psi_B) -> R (psi_A, psi_B)(P x)) evolves as the mirror image: RHS'(x) = R RHS(P x), and its sources are the
   mirrored ones. With the parities of the grid combinations D this is what GRChombo's reflective boundaries need.
   Also: the spherical star is mirror-symmetric, and each D_k has the parity its name says.
C. Spherical data (conformally flat, real f, g): each spinor obeys the 1D (u, v) equations of Spherical_BS_Evolver and
   the summed sources equal the 1D ones.
"""
import numpy as np
from dirac3d_ref import *

rng = np.random.default_rng(5)
mu = 0.9


def D_(f, X, k, h=2e-3):
    e = np.zeros(len(X)); e[k] = h
    return sum(w * (f(X + j * e) - f(X - j * e)) for j, w in [(1, 3 / 4), (2, -3 / 20), (3, 1 / 60)]) / h


def report(name, err, tol=1e-8):
    print('  %-74s %.1e  %s' % (name, err, 'OK' if err < tol else 'FAILED'))
    return err < tol


ok = True
c = rng.normal(size=80)
alpha_f = lambda X: 1.1 + 0.2 * np.sin(0.7 * X[0] + 0.3 * X[1] - 0.5 * X[2] + 0.2 * X[3] + c[0])
beta_f = lambda X: 0.15 * np.array([np.sin(c[1] + 0.4 * X[0] + 0.5 * X[1] + 0.3 * X[2]),
                                    np.cos(c[2] + 0.3 * X[0] - 0.6 * X[3] + 0.2 * X[1]),
                                    np.sin(c[3] - 0.5 * X[0] + 0.4 * X[2] + 0.7 * X[3])])


def gamma_f(X):
    M = np.zeros((3, 3)); n = 4
    for i in range(3):
        for j in range(i, 3):
            M[i, j] = M[j, i] = 0.25 * np.sin(c[n] + c[n + 1] * X[0] + c[n + 2] * X[1] + c[n + 3] * X[2] + c[n + 4] * X[3])
            n += 5
    return np.eye(3) + M


def psi_f(X):
    return np.array([np.exp(I * (c[40 + k] + 0.5 * X[0] + c[44 + k] * X[1])) * (0.5 + 0.2 * np.sin(c[48 + k] * X[2] + X[3]))
                     + 0.1 * I * np.cos(c[52 + k] * X[3] - X[1] * X[2]) for k in range(4)])


# ------------------------------------------------------------------------------------------------- A
print('A. 3+1 RHS (Cholesky triad) vs the covariant 4D Dirac operator, no symmetry:')
X0 = np.array([0.3, 0.2, -0.4, 0.5])


def metric4(X):
    a, b, g = alpha_f(X), beta_f(X), gamma_f(X)
    G = np.zeros((4, 4)); bl = g @ b
    G[0, 0] = -a * a + b @ bl; G[0, 1:] = G[1:, 0] = bl; G[1:, 1:] = g
    return G


def tetrad4(X):
    a, b, g = alpha_f(X), beta_f(X), gamma_f(X)
    E = np.zeros((4, 4))
    E[0, 0] = 1 / a; E[0, 1:] = -b / a
    E[1:, 1:] = np.linalg.inv(np.linalg.cholesky(g)).T.T   # rows = frame vectors e_a (E_3[i, a] -> E4[a, i])
    E[1:, 1:] = np.linalg.inv(np.linalg.cholesky(g))        # (L^{-T})^T = L^{-1}
    return E


g4 = metric4(X0); g4i = np.linalg.inv(g4)
dg4 = [D_(metric4, X0, m) for m in range(4)]
Chr = np.zeros((4, 4, 4))
for r in range(4):
    for m in range(4):
        for n in range(4):
            Chr[r, m, n] = 0.5 * sum(g4i[r, s] * (dg4[m][s, n] + dg4[n][s, m] - dg4[s][m, n]) for s in range(4))
E4 = tetrad4(X0); dE4 = [D_(tetrad4, X0, m) for m in range(4)]
Ecov = np.einsum('ab,mn,bn->am', ETA, g4, E4)
omega = np.array([ETA @ (Ecov @ (dE4[m] + np.einsum('nl,bl->bn', Chr[:, m, :], E4)).T) for m in range(4)])
Gs = [sum(omega[m, a, b] / 8 * (GAM[a] @ GAM[b] - GAM[b] @ GAM[a]) for a in range(4) for b in range(4)) for m in range(4)]
p0 = psi_f(X0); dp = [D_(psi_f, X0, m) for m in range(4)]
D4 = mu * p0 + sum(GAM[a] @ sum(E4[a, m] * (dp[m] + Gs[m] @ p0) for m in range(4)) for a in range(4))
a0, b0, g0 = alpha_f(X0), beta_f(X0), gamma_f(X0)
dal = np.array([D_(alpha_f, X0, k) for k in (1, 2, 3)])
dbe = np.array([D_(beta_f, X0, k) for k in (1, 2, 3)])
dga = np.array([D_(gamma_f, X0, k) for k in (1, 2, 3)])
gt = D_(gamma_f, X0, 0)
K0 = -(gt - (np.einsum('k,kij->ij', b0, dga) + np.einsum('kj,ik->ij', g0, dbe) + np.einsum('ik,jk->ij', g0, dbe))) / (2 * a0)
geo = geometry(a0, dal, b0, dbe, g0, dga, K0)
rhs = spinor_rhs(mu, geo, p0, dp[1:])
ok &= report('gamma^0 D4 psi + (d_t psi - RHS_3+1)/alpha', np.max(np.abs(GAM[0] @ D4 + (dp[0] - rhs) / a0)))

# ------------------------------------------------------------------------------------------------- B
print('B. reflections (generic data, both spinors):')
cB = rng.normal(size=120) * 0.5
sp = lambda X, n, amp=0.2: amp * np.sin(cB[n] + cB[n + 1] * X[0] + cB[n + 2] * X[1] + cB[n + 3] * X[2])
al3 = lambda X: 0.9 + sp(X, 0)
be3 = lambda X: np.array([sp(X, 4), sp(X, 8), sp(X, 12)])


def ga3(X):
    M = np.array([[sp(X, 16), sp(X, 20), sp(X, 24)], [0, sp(X, 28), sp(X, 32)], [0, 0, sp(X, 36)]])
    return np.eye(3) + M + np.triu(M, 1).T


def K3(X):
    M = np.array([[sp(X, 40), sp(X, 44), sp(X, 48)], [0, sp(X, 52), sp(X, 56)], [0, 0, sp(X, 60)]])
    return M + np.triu(M, 1).T


Psi3 = lambda X: np.array([sp(X, 64 + 4 * k, 0.5) + 1j * sp(X, 66 + 4 * k, 0.5) + 0.3 for k in range(8)])


def evaluate(al, be, ga, K, Psi, X):
    d = lambda f: np.array([D_(f, X, k, 1e-3) for k in range(3)])
    Dv = from_spinors(*np.split(Psi(X), 2))
    dD = [from_spinors(*np.split(x, 2)) for x in d(Psi)]
    return dirac_rhs_and_sources(mu, al(X), d(al), be(X), d(be), ga(X), d(ga), K(X), Dv, dD)


X = np.array([0.31, -0.42, 0.57])
r0 = evaluate(al3, be3, ga3, K3, Psi3, X)
errs, serr = [], []
for a in range(3):
    P = np.ones(3); P[a] = -1; Pm = np.diag(P)
    al_m = lambda Y: al3(P * Y)
    be_m = lambda Y: Pm @ be3(P * Y)
    ga_m = lambda Y: Pm @ ga3(P * Y) @ Pm
    K_m = lambda Y: Pm @ K3(P * Y) @ Pm
    Psi_m = lambda Y: RREF[a] @ Psi3(P * Y)
    rm = evaluate(al_m, be_m, ga_m, K_m, Psi_m, P * X)
    # RHS'(P X) = R RHS(X)   (in spinor form)
    lhs = np.concatenate(to_spinors(rm[0])); rhs_ = RREF[a] @ np.concatenate(to_spinors(r0[0]))
    errs.append(np.max(np.abs(lhs - rhs_)))
    serr.append(max(abs(rm[1] - r0[1]), np.max(np.abs(rm[2] - Pm @ r0[2])), np.max(np.abs(rm[3] - Pm @ r0[3] @ Pm)),
                    abs(rm[4] - r0[4])))
ok &= report('mirrored data evolve as the mirror image, RHS\'(Px) = R RHS(x), for x, y, z', max(errs), 1e-7)
ok &= report('mirrored sources: rho\'(Px) = rho(x), S\'_i = P S_i, S\'_ij = P S_ij P, n', max(serr), 1e-7)
ok &= report('T_ab real', r0[5], 1e-12)
ok &= report('reflections commute and square to 1', max(np.max(np.abs(RREF[i] @ RREF[j] - RREF[j] @ RREF[i])) for i in range(3) for j in range(3))
             + max(np.max(np.abs(R @ R - np.eye(8))) for R in RREF), 1e-14)

gfun = lambda r: 0.2 * np.exp(-r * r / 4) * (1 + 0.1 * r * r)
ffun = lambda r: 0.05 * r * np.exp(-r * r / 3)


def star_field(Y):
    r = np.linalg.norm(Y)
    return np.concatenate(to_spinors(star_D(gfun(r), ffun(r), Y / r)))


perr, derr = [], []
for a in range(3):
    P = np.ones(3); P[a] = -1
    for Y in rng.normal(size=(5, 3)):
        perr.append(np.max(np.abs(star_field(Y) - RREF[a] @ star_field(P * Y))))
        Dk, Dk_m = from_spinors(*np.split(star_field(Y), 2)), from_spinors(*np.split(star_field(P * Y), 2))
        eps = np.array([-1 if 'xyz'[a] in name else 1 for name in DIRAC_NAMES])
        Dk_t = from_spinors(*np.split(Psi3(Y), 2))
        derr.append(np.max(np.abs(Dk_m - eps * Dk)))
ok &= report('spherical star is mirror-symmetric in x, y and z', max(perr), 1e-14)
ok &= report('each grid variable D_name is odd exactly in the directions in its name', max(derr), 1e-14)
# D-basis parities for any mirror-symmetric field: R in the D basis is diagonal with those signs
derr2 = 0.
for a in range(3):
    RD = DIRAC_BASIS.T @ RREF[a] @ DIRAC_BASIS
    eps = np.array([-1 if 'xyz'[a] in name else 1 for name in DIRAC_NAMES])
    derr2 = max(derr2, np.max(np.abs(RD - np.diag(eps))))
ok &= report('reflections are diagonal (+-1) in the D basis', derr2, 1e-14)

# ------------------------------------------------------------------------------------------------- C
print('C. spherical (conformally flat) data reduce to the 1D equations and sources:')
psif = lambda r: 1 + 0.3 * np.exp(-r * r / 5)
alf = lambda r: 0.8 + 0.15 * (1 - np.exp(-r * r / 6))


def sph_eval(X):
    r = np.linalg.norm(X)
    rr = lambda Y: np.linalg.norm(Y)
    d = lambda f: np.array([D_(f, X, k, 1e-3) for k in range(3)])
    ga = lambda Y: psif(rr(Y))**4 * np.eye(3)
    al = lambda Y: alf(rr(Y))
    be = lambda Y: np.zeros(3)
    Dv = lambda Y: star_D(gfun(rr(Y)), ffun(rr(Y)), Y / rr(Y))
    return dirac_rhs_and_sources(mu, al(X), d(al), be(X), d(be), ga(X), d(ga), np.zeros((3, 3)), Dv(X), list(d(Dv)))


def d1(f, r):
    return D_(lambda Y: np.array([f(Y[0])]), np.array([r]), 0, 1e-3)[0]


eR, eS = [], []
for X in [np.array([0.4, 0.9, -0.3]), np.array([-1.1, 0.3, 0.8])]:
    r = np.linalg.norm(X); n = X / r
    res = sph_eval(X)
    chi = psif(r)**-4; cc = np.sqrt(chi)
    L = 0.5 * (-d1(lambda s: psif(s)**-4, r) / chi + d1(alf, r) / alf(r))
    u, v = gfun(r), ffun(r)
    ut = -I * alf(r) * mu * u + I * alf(r) * (cc * (d1(ffun, r) + L * v) + 2 * cc * v / r)
    vt = I * alf(r) * mu * v - I * alf(r) * cc * (d1(gfun, r) + L * u)
    expect = star_D(ut, vt, n)
    eR.append(np.max(np.abs(res[0] - expect)))
    Srr = cc * (v * d1(gfun, r) - u * d1(ffun, r)) / (2 * np.pi)
    Stt = -cc * u * v / r / (2 * np.pi)
    rho1 = mu * (u * u - v * v) / (2 * np.pi) + Srr + 2 * Stt
    g = psif(r)**4
    Sij1 = g * (Srr * np.outer(n, n) + Stt * (np.eye(3) - np.outer(n, n)))
    eS.append(max(abs(res[1] - rho1), np.max(np.abs(res[3] - Sij1)), np.max(np.abs(res[2])),
                  abs(res[4] - (u * u + v * v) / (2 * np.pi))))
ok &= report('RHS = 1D (u, v) equations for both spinors', max(eR), 1e-8)
ok &= report('rho, S_i = 0, S_ij and n = 1D sources (sum of both spinors)', max(eS), 1e-8)
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
