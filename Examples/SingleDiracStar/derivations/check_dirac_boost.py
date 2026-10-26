"""
Boosted Dirac stars: derivation and numerical check of the spinor transformation (python3 check_dirac_boost.py).

Setting (as for the boosted boson stars of ExoZvezda's BosonStarHelperFunction and 2DChombo's BosonStar). The star is
static in its rest frame,
    ds^2 = -alpha(r~)^2 dt~^2 + psi(r~)^4 (dx~^2 + dy~^2 + dz~^2),
and the lab coordinates are related by t~ = cosh(xi) t + sinh(xi) x, x~ = sinh(xi) t + cosh(xi) x (rapidity xi; at t = 0,
t~ = x sinh xi and x~ = x cosh xi, with x measured from the star). In the lab slices the 3-metric is
diag(psi^4 cosh^2 - alpha^2 sinh^2, psi^4, psi^4), lapse alpha psi^2 / sqrt(psi^4 cosh^2 - alpha^2 sinh^2) and shift
along x, exactly as for boson stars (with psi_BS = psi^2).

Spinor. The spinor itself is a coordinate scalar, but its components refer to the tetrad. The rest-frame tetrad is
e~_0 = alpha^-1 d_t~, e~_i = psi^-2 d_i~. The lab tetrad of the evolution code is e_0 = n, the unit normal of the lab
slices, and the spatial triad of the lab 3-metric. That 3-metric is diagonal, so the Cholesky (3D) and symmetric (2D)
triads coincide: e_x along d_x|_t, e_y = e~_y, e_z = e~_z. Writing n and e_x in the rest frame,
    n = cosh(eta) e~_0 + sinh(eta) e~_x,   e_x = sinh(eta) e~_0 + cosh(eta) e~_x,   tanh(eta) = tanh(xi) alpha / psi^2,
so the two frames differ by a local boost in the 0-x plane with rapidity eta(x) (the velocity of the lab normal
observers relative to the star, measured in the star's orthonormal frame). The lab components are then
    psi_lab(t, x) = U(eta) psi_rest(t~, x~),   U(eta) = exp(+(eta/2) gamma^0 gamma^1) = cosh(eta/2) - sinh(eta/2) alpha_x,
with alpha_x = gamma_D^0 gamma_D^1 (block off-diagonal sigma_x; sigma_3 in the 2D code, whose quantisation axis is x).
U is fixed by U^-1 gamma^a U Lambda^b_a = gamma^b for the frame change e_a = e~_b Lambda^b_a, which leaves
gamma^a e_a^mu d_mu invariant.

For the star, psi_rest = e^{-i omega t~} psi_0(x~), so the initial data at lab t = 0 are
    psi_lab(x) = U(eta(r~)) e^{-i omega x sinh xi} psi_0(x cosh xi, y, z),   r~ = sqrt(x^2 cosh^2 xi + y^2 + z^2),
applied to both spinors (U acts on the spinor index only, commutes with the y and z reflections, and maps the
x reflection of velocity v to -v, so the parities of the grid variables are unchanged and head-on binaries can use an
x = 0 reflection).

Checks:
  1. U^-1 gamma^a U = (Lambda^-1)^a_b gamma^b (the representation property; the other sign fails).
  2. Covariance on an arbitrary static background and an arbitrary spinor field (not a solution): the 4D Dirac
     operator evaluated in lab coordinates with the lab tetrad on psi_lab equals U times the operator evaluated in
     rest coordinates with the rest tetrad on psi_rest: D_lab psi_lab = U D_rest psi_rest. The same check with the other
     sign of eta, or without U, fails.
  3. The lab 3+1 RHS (dirac3d_ref, with lapse, shift, gamma_ij and K_ij of the lab slicing) applied to psi_lab equals
     the exact lab time derivative of psi_lab = U psi_rest whenever psi_rest solves the rest-frame equation; here,
     with a field that is not a solution, the residuals transform as d_t psi_lab - RHS_lab = (alpha_lab gamma^0) U
     (alpha_rest gamma^0)^-1 (d_t~ psi_rest - RHS_rest) (times the appropriate factor), which is what check 2 says in
     3+1 language; we check it directly.
"""
import numpy as np
from dirac3d_ref import *

rng = np.random.default_rng(23)
mu, omega = 0.9, 0.83
xi = 0.4                       # boost rapidity
ch, sh = np.cosh(xi), np.sinh(xi)
alf = lambda r: 0.75 + 0.2 * (1 - np.exp(-r * r / 4))          # toy static background
psif = lambda r: 1 + 0.25 * np.exp(-r * r / 5)
cS = rng.normal(size=40)


def chi0(Xt):
    """arbitrary smooth rest-frame spinor (spatial part)"""
    x, y, z = Xt
    return np.array([np.exp(I * cS[k]) * (0.4 + 0.2 * np.sin(cS[4 + k] * x + cS[8 + k] * y + cS[12 + k] * z))
                     + 0.1 * I * np.cos(cS[16 + k] * x * y - z) for k in range(4)])


def psi_rest(Xt4):
    return np.exp(-I * omega * Xt4[0]) * chi0(Xt4[1:])


def rest_coords(X):
    t, x, y, z = X
    return np.array([ch * t + sh * x, sh * t + ch * x, y, z])


def eta_of(X):
    Xt = rest_coords(X)
    r = np.linalg.norm(Xt[1:])
    return np.arctanh(np.tanh(xi) * alf(r) / psif(r)**2)


def U(eta, sign=1):
    return expm_boost(sign * eta)


def expm_boost(e):
    G01 = GAM[0] @ GAM[1]
    return np.cosh(e / 2) * np.eye(4) + np.sinh(e / 2) * G01    # (gamma^0 gamma^1)^2 = 1


def report(name, err, tol=1e-8):
    print('  %-80s %.1e  %s' % (name, err, 'OK' if err < tol else 'FAILED'))
    return err < tol


def Dd(f, X, k, h=2e-3):
    e = np.zeros(len(X)); e[k] = h
    return sum(w * (f(X + j * e) - f(X - j * e)) for j, w in [(1, 3 / 4), (2, -3 / 20), (3, 1 / 60)]) / h


def dirac4(metric, tetrad, field, X):
    """4D Dirac operator (gamma^a e_a^mu nabla_mu + mu) field at X, from metric(X), tetrad(X) (rows e_a^mu)"""
    g = metric(X); gi = np.linalg.inv(g)
    dg = [Dd(metric, X, m) for m in range(4)]
    Chr = 0.5 * (np.einsum('rs,msn->rmn', gi, np.array(dg)) + np.einsum('rs,nsm->rmn', gi, np.array(dg))
                 - np.einsum('rs,smn->rmn', gi, np.array(dg)))
    E = tetrad(X); dE = [Dd(tetrad, X, m) for m in range(4)]
    Ecov = np.einsum('ab,mn,bn->am', ETA, g, E)
    om = np.array([ETA @ (Ecov @ (dE[m] + np.einsum('nl,bl->bn', Chr[:, m, :], E)).T) for m in range(4)])
    Gs = [sum(om[m, a, b] / 8 * (GAM[a] @ GAM[b] - GAM[b] @ GAM[a]) for a in range(4) for b in range(4)) for m in range(4)]
    p = field(X); dp = [Dd(field, X, m) for m in range(4)]
    return mu * p + sum(GAM[a] @ sum(E[a, m] * (dp[m] + Gs[m] @ p) for m in range(4)) for a in range(4))


# rest frame
def metric_rest(Xt):
    r = np.linalg.norm(Xt[1:])
    return np.diag([-alf(r)**2, psif(r)**4, psif(r)**4, psif(r)**4])


def tetrad_rest(Xt):
    r = np.linalg.norm(Xt[1:])
    return np.diag([1 / alf(r), psif(r)**-2, psif(r)**-2, psif(r)**-2])


# lab frame: g_lab = J^T g_rest J with J = d X~ / d X
J = np.array([[ch, sh, 0, 0], [sh, ch, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def metric_lab(X):
    return J.T @ metric_rest(rest_coords(X)) @ J


def lab_3p1(X):
    g = metric_lab(X)
    gam = g[1:, 1:]
    beta_low = g[0, 1:]
    beta = np.linalg.solve(gam, beta_low)
    alpha = np.sqrt(beta @ beta_low - g[0, 0])
    return alpha, beta, gam


def tetrad_lab(X):
    alpha, beta, gam = lab_3p1(X)
    E = np.zeros((4, 4))
    E[0, 0] = 1 / alpha; E[0, 1:] = -beta / alpha
    E[1:, 1:] = np.linalg.inv(np.linalg.cholesky(gam))   # rows e_a^i = (L^-T)^T
    return E


ok = True
print('1. representation property of U:')
eta = 0.37
Lam = np.eye(4); Lam[0, 0] = Lam[1, 1] = np.cosh(eta); Lam[0, 1] = Lam[1, 0] = np.sinh(eta)
for sgn, want in ((1, True), (-1, False)):
    S = U(eta, sgn)
    good = all(np.allclose(np.linalg.inv(S) @ GAM[a] @ S, sum(np.linalg.inv(Lam)[a, b] * GAM[b] for b in range(4)))
               for a in range(4))
    print('  sign %+d: U^-1 gamma^a U = (Lambda^-1)^a_b gamma^b : %s %s' % (sgn, good, 'OK' if good == want else 'FAILED'))
    ok &= (good == want)

print('2. covariance: D_lab (U psi_rest) = U D_rest psi_rest on an arbitrary background and field:')
# the lab tetrad really is the rest tetrad boosted by eta: e_0 = n = ch(eta) e~_0 + sh(eta) e~_x
errs = []
for X in [np.array([0.0, 0.7, -0.4, 0.3]), np.array([0.2, -1.1, 0.5, 0.9]), np.array([-0.3, 0.4, 1.2, -0.6])]:
    Xt = rest_coords(X)
    e_lab = tetrad_lab(X)                                 # rows: e_a in lab coordinate components
    e_rest_in_lab = tetrad_rest(Xt) @ np.linalg.inv(J).T   # rest tetrad vectors in lab components
    et = eta_of(X)
    Lm = np.eye(4); Lm[0, 0] = Lm[1, 1] = np.cosh(et); Lm[0, 1] = Lm[1, 0] = np.sinh(et)
    errs.append(np.max(np.abs(e_lab - Lm @ e_rest_in_lab)))
ok &= report('lab tetrad = rest tetrad boosted with tanh(eta) = tanh(xi) alpha/psi^2', max(errs), 1e-12)

for label, sgn in (('U(eta) [derived]', 1), ('U(-eta) [wrong sign]', -1), ('no U [scalar-like]', 0)):
    errs = []
    field_lab = lambda X: (U(eta_of(X), sgn) if sgn else np.eye(4)) @ psi_rest(rest_coords(X))
    for X in [np.array([0.0, 0.7, -0.4, 0.3]), np.array([0.2, -1.1, 0.5, 0.9])]:
        lhs = dirac4(metric_lab, tetrad_lab, field_lab, X)
        rhs = (U(eta_of(X), sgn) if sgn else np.eye(4)) @ dirac4(metric_rest, tetrad_rest, psi_rest, rest_coords(X))
        errs.append(np.max(np.abs(lhs - rhs)))
    if sgn == 1:
        ok &= report('D_lab psi_lab = U D_rest psi_rest with ' + label, max(errs), 1e-8)
    else:
        bad = max(errs) > 1e-3
        print('  %-80s %.1e  %s' % ('the same with ' + label + ' (must fail)', max(errs), 'OK' if bad else 'FAILED'))
        ok &= bad

print('3. lab 3+1 RHS (dirac3d_ref) on the boosted data:')
# residual relation: gamma^0 D4 + (d_t psi - RHS)/alpha = 0 in each frame (check A of check_dirac3d.py), so
# d_t psi_lab - RHS_lab = -alpha_lab gamma^0 D_lab psi_lab = -alpha_lab gamma^0 U D_rest psi_rest
field_lab = lambda X: U(eta_of(X)) @ psi_rest(rest_coords(X))
errs = []
for X in [np.array([0.0, 0.7, -0.4, 0.3]), np.array([0.0, -1.1, 0.5, 0.9])]:
    alpha, beta, gam = lab_3p1(X)
    d3 = lambda f: np.array([Dd(f, X, k) for k in (1, 2, 3)])
    dalpha = d3(lambda Y: np.array(lab_3p1(Y)[0]))
    dbeta = d3(lambda Y: lab_3p1(Y)[1])
    dgam = d3(lambda Y: lab_3p1(Y)[2])
    gt = Dd(lambda Y: lab_3p1(Y)[2], X, 0)
    K = -(gt - (np.einsum('k,kij->ij', beta, dgam) + np.einsum('kj,ik->ij', gam, dbeta) +
                np.einsum('ik,jk->ij', gam, dbeta))) / (2 * alpha)
    geo = geometry(alpha, dalpha, beta, dbeta, gam, dgam, K)
    dpsi = [Dd(field_lab, X, k) for k in (1, 2, 3)]
    rhs = spinor_rhs(mu, geo, field_lab(X), dpsi)
    dt = Dd(field_lab, X, 0)
    expect = -alpha * GAM[0] @ U(eta_of(X)) @ dirac4(metric_rest, tetrad_rest, psi_rest, rest_coords(X))
    errs.append(np.max(np.abs((dt - rhs) - expect)))
ok &= report('d_t psi_lab - RHS_lab = -alpha_lab gamma^0 U D_rest psi_rest', max(errs), 1e-8)
print('4. reflections (grid-variable parities and head-on binaries):')
Ub = np.kron(np.eye(2), U(0.3))                       # acts on both spinors
e4 = max(np.max(np.abs(Ub @ RREF[a] - RREF[a] @ Ub)) for a in (1, 2))
e4x = np.max(np.abs(RREF[0] @ Ub @ RREF[0] - np.kron(np.eye(2), U(-0.3))))
ok &= report('U commutes with the y and z reflections', e4, 1e-14)
ok &= report('x reflection maps U(eta) to U(-eta) (a star moving with -v is the mirror image)', e4x, 1e-14)
print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
