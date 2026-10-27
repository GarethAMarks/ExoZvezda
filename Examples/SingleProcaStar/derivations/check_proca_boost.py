"""
Boosted Proca stars: derivation of the lab-frame data and numerical checks (run: python3 check_proca_boost.py).

Coordinates (as for the boson and Dirac stars):  t~ = cosh(xi) t + sinh(xi) x,  x~ = sinh(xi) t + cosh(xi) x,  y~ = y,
z~ = z, so a star of rapidity xi > 0 moves towards -x. In the rest frame the star is (isotropic coordinates, P = psi^2)
    ds^2 = -alpha(r~)^2 dt~^2 + P(r~)^2 dx~^2,   A = e^{-i omega t~} (f dt~ + i g n~_i dx~^i),   F_{t~ i~} = -E n~_i e^{-i omega t~},
with f, g, E = f_iso, g_iso, E_iso of the 1D solver (F_ij = 0 in the rest frame, since A_i~ is a gradient direction).
A_mu is a covector and F_mu nu a 2-form, so the lab components follow from the Jacobian L = d x~ / d x alone (unlike the
spinor, no local Lorentz transformation of a frame is involved):
    A_t = cosh A_t~ + sinh A_x~,  A_x = sinh A_t~ + cosh A_x~,  A_y, A_z unchanged;
    with e_i = F_{t~ i~}:  F_tx = e_x,  F_ty = cosh e_y,  F_tz = cosh e_z,  F_xy = sinh e_y,  F_xz = sinh e_z,  F_yz = 0.
The 3+1 variables then follow from the lab slicing (lapse, beta^x, gamma_ij as for the boson stars):
    X_i = A_i,   X_phi = -(A_t - beta^x A_x) / lapse,
    E_x = -e_x / lapse,   E_y = -(cosh - beta^x sinh) e_y / lapse,   E_z = -(cosh - beta^x sinh) e_z / lapse,
    E^i = gamma^{ij} E_j   (gamma is diagonal: gamma_xx = P^2 cosh^2 - alpha^2 sinh^2, gamma_yy = gamma_zz = P^2).

Checks, for arbitrary profiles f, g, alpha, psi (not a solution; E = f' - omega g so that F = dA) and t != 0:
A. The closed forms above (and the lapse, shift, gamma_ij and K_ij of BosonStarHelperFunction) agree with the generic
   tensor transformation of the rest-frame metric, A and F, with K_ij from the lab metric's time derivative.
B. Evolution: the lab data are a coordinate transform of a rest-frame configuration, so the identities of
   check_proca3d.py hold with the residuals R^b and nabla_a A^a computed in the rest frame:
     d_t X_i - RHS = 0,  d_t E^i - RHS = alpha gamma^i_b R^b,  d_t X_phi - RHS = alpha nabla_a A^a,  C = n_b R^b,
   where the lab RHS uses the closed-form data and the time derivatives are finite differences in lab time.
   For the actual star R = 0 and nabla.A = 0 (box test, tests/box_test mode 1).
C. Mirror: under x -> -x the data of rapidity xi map to those of -xi with X_x, E^x odd, so an equal-mass head-on binary
   (star 2 = mirror of star 1) is mirror symmetric with the grid-variable parities.
"""
import numpy as np
from proca3d_ref import proca_rhs_and_sources

I = 1j
mu, kappa, omega, phase0 = 0.9, 0.0, 0.83, 0.3
h = 1e-2
prof = dict(f=lambda r: 0.3 * np.exp(-0.5 * r * r) * (1 - 0.4 * r * r), g=lambda r: -0.2 * r * np.exp(-0.4 * r * r),
            alpha=lambda r: 0.8 + 0.15 * (1 - np.exp(-r * r)),
            psi=lambda r: 1.05 - 0.04 * np.exp(-r * r))
dprof = lambda key, r: (prof[key](r + 1e-5) - prof[key](r - 1e-5)) / 2e-5
# F = dA requires E = f' - omega g (in isotropic coordinates, as the 1D E_iso = (f' - omega g) dr/dR)
prof['E'] = lambda r: -0.3 * r * np.exp(-0.5 * r * r) * (1 - 0.4 * r * r) - 0.24 * r * np.exp(-0.5 * r * r) - omega * prof['g'](r)


def D_(f, X, k, hh=h):
    e = np.zeros(len(X)); e[k] = hh
    return sum(w * (f(X + j * e) - f(X - j * e)) for j, w in [(1, 3 / 4), (2, -3 / 20), (3, 1 / 60)]) / hh


def report(name, err, tol=1e-8):
    print('  %-80s %.1e  %s' % (name, err, 'OK' if err < tol else 'FAILED'))
    return err < tol


def closed_form(X, xi):
    """lab 3+1 data at the 4D point X = (t, x, y, z), centre at the origin: the formulas of ProcaStarBoost.hpp"""
    t, x, y, z = X
    c_, s_, v_ = np.cosh(xi), np.sinh(xi), np.tanh(xi)
    tt, xx = c_ * t + s_ * x, s_ * t + c_ * x
    r = np.sqrt(xx * xx + y * y + z * z); n = np.array([xx, y, z]) / r
    alpha, psi = prof['alpha'](r), prof['psi'](r)
    dalpha, dpsi = dprof('alpha', r), dprof('psi', r)
    P, dP = psi * psi, 2 * psi * dpsi
    gxx = P * P * c_ * c_ - alpha * alpha * s_ * s_
    lapse = alpha * P / np.sqrt(gxx)
    bx = s_ * c_ * (P * P - alpha * alpha) / gxx
    gamma = np.diag([gxx, P * P, P * P])
    K = np.zeros((3, 3))
    K[2, 2] = K[1, 1] = -lapse * s_ * xx * dP / (r * P)
    K[0, 1] = K[1, 0] = lapse * c_ * s_ * (y / r) * (dP / P - dalpha / alpha)
    K[0, 2] = K[2, 0] = lapse * c_ * s_ * (z / r) * (dP / P - dalpha / alpha)
    K[0, 0] = lapse * (xx / r) * s_ * c_ * c_ * (dP / P - 2 * dalpha / alpha + v_ * v_ * alpha * dalpha / (P * P))
    ph = np.exp(I * (-omega * tt + np.pi * phase0))
    At_r, Ai_r, e = prof['f'](r) * ph, I * prof['g'](r) * n * ph, -prof['E'](r) * n * ph
    At, Ax = c_ * At_r + s_ * Ai_r[0], s_ * At_r + c_ * Ai_r[0]
    Xi = np.array([Ax, Ai_r[1], Ai_r[2]])
    Xphi = -(At - bx * Ax) / lapse
    E_dn = np.array([-e[0], -(c_ - bx * s_) * e[1], -(c_ - bx * s_) * e[2]]) / lapse
    return dict(lapse=lapse, beta=np.array([bx, 0, 0]), gamma=gamma, K=K, X=Xi, E=np.linalg.solve(gamma, E_dn), Xphi=Xphi)


def rest_fields(Xt):
    """rest-frame 4-metric, A_mu, F_mu nu at rest coordinates Xt = (t~, x~, y~, z~)"""
    r = np.linalg.norm(Xt[1:]); n = Xt[1:] / r
    g4 = np.diag([-prof['alpha'](r)**2] + [prof['psi'](r)**4] * 3)
    ph = np.exp(I * (-omega * Xt[0] + np.pi * phase0))
    A = np.concatenate([[prof['f'](r) * ph], I * prof['g'](r) * n * ph])
    F = np.zeros((4, 4), complex); F[0, 1:] = -prof['E'](r) * n * ph; F[1:, 0] = -F[0, 1:]
    return g4, A, F


def Lmat(xi):   # L[a, mu] = d x~^a / d x^mu
    L = np.eye(4); c_, s_ = np.cosh(xi), np.sinh(xi)
    L[0, 0] = L[1, 1] = c_; L[0, 1] = L[1, 0] = s_
    return L


def generic(X, xi):
    L = Lmat(xi)
    g4t, At, Ft = rest_fields(L @ X)
    g4, A, F = L.T @ g4t @ L, L.T @ At, L.T @ Ft @ L
    gi4 = np.linalg.inv(g4)
    lapse = 1 / np.sqrt(-gi4[0, 0]); beta = gi4[0, 1:] * lapse**2
    n = np.concatenate([[1 / lapse], -beta / lapse])
    gamma = g4[1:, 1:]
    return dict(lapse=lapse, beta=beta, gamma=gamma, X=A[1:], Xphi=-(n @ A), E=np.linalg.solve(gamma, F[1:, :] @ n))


ok = True
xi = 0.4
X0 = np.array([0.35, 0.3, -0.5, 0.45])
print('A. Closed forms vs the generic transformation (rapidity %.1f, t = %.2f):' % (xi, X0[0]))
cf, gn = closed_form(X0, xi), generic(X0, xi)
ok &= report('lapse, shift, gamma_ij', max(abs(cf['lapse'] - gn['lapse']), np.max(np.abs(cf['beta'] - gn['beta'])),
                                           np.max(np.abs(cf['gamma'] - gn['gamma']))), 1e-12)
ok &= report('X_i, X_phi, E^i', max(np.max(np.abs(cf['X'] - gn['X'])), abs(cf['Xphi'] - gn['Xphi']),
                                    np.max(np.abs(cf['E'] - gn['E']))), 1e-12)
# K_ij = -(d_t gamma_ij - L_beta gamma_ij) / (2 alpha) from the generic lab metric
gam = lambda Y: generic(Y, xi)['gamma']; bet = lambda Y: generic(Y, xi)['beta']
dgam = np.array([D_(gam, X0, k + 1) for k in range(3)]); dbet = np.array([D_(bet, X0, k + 1) for k in range(3)])
b0, g0 = gn['beta'], gn['gamma']
Kg = -(D_(gam, X0, 0) - (np.einsum('k,kij->ij', b0, dgam) + dbet @ g0 + (dbet @ g0).T)) / (2 * gn['lapse'])
ok &= report('K_ij', np.max(np.abs(cf['K'] - Kg)))

print('B. Lab 3+1 RHS (closed-form data incl. K_ij) vs exact lab time derivatives, with rest-frame residuals:')
key3 = lambda key: (lambda Y: closed_form(Y, xi)[key])
d = lambda key, k: D_(key3(key), X0, k + 1)
dalpha = np.array([d('lapse', k) for k in range(3)]); dbeta = np.array([d('beta', k) for k in range(3)])
dgamma = np.array([d('gamma', k) for k in range(3)])
V = dict(X=cf['X'], E=cf['E'], Xphi=cf['Xphi'], Z=0j)
dV = dict(X=np.array([d('X', k) for k in range(3)]), E=np.array([d('E', k) for k in range(3)]),
          Xphi=np.array([d('Xphi', k) for k in range(3)]), Z=np.zeros(3, complex))
ddX = np.array([[D_(lambda Y: D_(key3('X'), Y, l + 1), X0, k + 1) for l in range(3)] for k in range(3)])
rhs, src = proca_rhs_and_sources(mu, kappa, cf['lapse'], dalpha, cf['beta'], dbeta, cf['gamma'], dgamma, cf['K'], V, dV,
                                 ddX)
dtX, dtE, dtP = D_(key3('X'), X0, 0), D_(key3('E'), X0, 0), D_(key3('Xphi'), X0, 0)
# rest-frame residuals R~^b = nabla_a F^{ab} - mu^2 A^b and nabla_a A^a at x~ = L X0, then R^b = (L^{-1})^b_a R~^a
L = Lmat(xi); Xt0 = L @ X0


def rest_up(Y, which):
    g4, A, F = rest_fields(Y); gi = np.linalg.inv(g4); sg = np.sqrt(-np.linalg.det(g4))
    return sg * (gi @ F @ gi) if which == 'F' else sg * (gi @ A)


g4t, At0, Ft0 = rest_fields(Xt0); sg0 = np.sqrt(-np.linalg.det(g4t))
Rt = sum(D_(lambda Y: rest_up(Y, 'F')[m, :], Xt0, m) for m in range(4)) / sg0 - mu**2 * np.linalg.inv(g4t) @ At0
divA = sum(D_(lambda Y: rest_up(Y, 'A')[m], Xt0, m) for m in range(4)) / sg0
R = np.linalg.solve(L, Rt)
lapse = cf['lapse']; n_up = np.concatenate([[1 / lapse], -cf['beta'] / lapse]); n_dn = np.array([-lapse, 0, 0, 0])
proj = R[1:] + n_up[1:] * (n_dn @ R)
ok &= report('d_t X_i - RHS', np.max(np.abs(dtX - rhs['X'])))
ok &= report('d_t E^i - RHS - alpha gamma^i_b R^b', np.max(np.abs(dtE - rhs['E'] - lapse * proj)))
ok &= report('d_t X_phi - RHS - alpha nabla_a A^a', abs(dtP - rhs['Xphi'] - lapse * divA))
ok &= report('Gauss constraint - n_b R^b', abs(src['Gauss'] - n_dn @ R))
print('    (for scale: |R^b| = %.2f, |nabla_a A^a| = %.2f)' % (np.max(np.abs(R)), abs(divA)))

print('C. Mirror symmetry x -> -x:')
Pm = np.diag([-1., 1, 1])
Xm = X0 * np.array([1, -1, 1, 1])
a, b = closed_form(X0, xi), closed_form(Xm, -xi)
err = max(abs(a['lapse'] - b['lapse']), np.max(np.abs(Pm @ a['beta'] - b['beta'])), np.max(np.abs(Pm @ a['gamma'] @ Pm - b['gamma'])),
          np.max(np.abs(Pm @ a['K'] @ Pm - b['K'])), np.max(np.abs(Pm @ a['X'] - b['X'])), np.max(np.abs(Pm @ a['E'] - b['E'])),
          abs(a['Xphi'] - b['Xphi']))
ok &= report('data(-x; -xi) = mirror of data(x; xi)', err, 1e-13)
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
