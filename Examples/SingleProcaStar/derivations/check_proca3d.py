"""
Numerical verification of the 3D Proca equations in proca3d_ref.py (run: python3 check_proca3d.py, a few seconds).

A. Against the covariant 4D theory, for arbitrary alpha(t,x), beta^i(t,x), gamma_ij(t,x) and an arbitrary complex
   A_a(t,x) (no field equations imposed), with Z = 0 and finite differences. Writing
   R^b = nabla_a F^{ab} - mu^2 A^b (the field equation) and L = nabla_a A^a (the Lorenz condition it implies),
     d_t X_i   - RHS = 0                              (definition of E^i)
     d_t E^i   - RHS = alpha gamma^i_b R^b            (spatial projection of the field equation)
     d_t X_phi - RHS = alpha L
     C = D_i E^i + mu^2 X_phi = n_b R^b               (the Gauss law is the normal projection)
   so the 3+1 system is the field equation, with the Lorenz condition and the Gauss law as constraints.
   The sources against T_ab = Re(F_ac Fbar_b^c + mu^2 A_a Abar_b) + g_ab L_matter projected with n.
   The number density against -n_a j^a with j^a = Im(F^{ab} Abar_b) (the Noether current of A -> e^{i theta} A).
B. Reflections x_k -> -x_k: with the parities X_k, E^k odd in x_k (everything else even) the RHS is mirror covariant.
C. Spherical data: the 3D RHS and sources reduce to those of Spherical_BS_Evolver's ComplexVectorField (D = 4).
"""
import numpy as np
from proca3d_ref import proca_rhs_and_sources

rng = np.random.default_rng(7)
mu, kappa = 0.8, 0.0
I = 1j
h = 1e-2


def D_(f, X, k, hh=h):
    e = np.zeros(len(X)); e[k] = hh
    return sum(w * (f(X + j * e) - f(X - j * e)) for j, w in [(1, 3 / 4), (2, -3 / 20), (3, 1 / 60)]) / hh


def report(name, err, tol=1e-8):
    print('  %-80s %.1e  %s' % (name, err, 'OK' if err < tol else 'FAILED'))
    return err < tol


ok = True
c = rng.normal(size=120)
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


def A_f(X):   # covariant A_mu, complex
    return np.array([(0.5 + 0.3 * np.sin(c[40 + m] + c[44 + m] * X[0] + c[48 + m] * X[1] - c[52 + m] * X[2] + 0.4 * X[3]))
                     * np.exp(I * (c[56 + m] + 0.6 * X[0] - c[60 + m] * X[1] + 0.3 * c[64 + m] * X[3]))
                     for m in range(4)])


def metric4(X):
    a, b, g = alpha_f(X), beta_f(X), gamma_f(X)
    G = np.zeros((4, 4)); bl = g @ b
    G[0, 0] = -a * a + b @ bl; G[0, 1:] = G[1:, 0] = bl; G[1:, 1:] = g
    return G


def F_f(X):
    dA = np.array([D_(A_f, X, m) for m in range(4)])   # dA[m, n] = d_m A_n
    return dA - dA.T


def n_up(X):
    a, b = alpha_f(X), beta_f(X)
    return np.concatenate([[1 / a], -b / a])


def vars31(X):
    """(X_i, E^i, X_phi) at the 4D point X"""
    A, F, n, g = A_f(X), F_f(X), n_up(X), gamma_f(X)
    E_dn = F[1:, :] @ n                                # E_i = F_{i a} n^a
    return A[1:], np.linalg.inv(g) @ E_dn, -(n @ A)


X0 = np.array([0.3, 0.2, -0.4, 0.5])
print('A. 3+1 RHS vs the covariant 4D Proca equations (arbitrary metric, gauge and A_a; Z = 0):')
alpha, beta, gamma = alpha_f(X0), beta_f(X0), gamma_f(X0)
dalpha = np.array([D_(alpha_f, X0, k + 1) for k in range(3)])
dbeta = np.array([D_(beta_f, X0, k + 1) for k in range(3)])
dgamma = np.array([D_(gamma_f, X0, k + 1) for k in range(3)])
dtgamma = D_(gamma_f, X0, 0)
Lbg = sum(beta[k] * dgamma[k] for k in range(3)) + dbeta @ gamma + (dbeta @ gamma).T
K = -(dtgamma - Lbg) / (2 * alpha)

Xv, Ev, Pv = vars31(X0)
dvar = lambda k: [D_(lambda Y: vars31(Y)[q], X0, k + 1) for q in range(3)]
dd = [dvar(k) for k in range(3)]
V = dict(X=Xv, E=Ev, Xphi=Pv, Z=0j)
dV = dict(X=np.array([dd[k][0] for k in range(3)]), E=np.array([dd[k][1] for k in range(3)]),
          Xphi=np.array([dd[k][2] for k in range(3)]), Z=np.zeros(3, complex))
ddX = np.array([[D_(lambda Y: D_(lambda W: A_f(W)[1:], Y, l + 1), X0, k + 1) for l in range(3)] for k in range(3)])
rhs, src = proca_rhs_and_sources(mu, kappa, alpha, dalpha, beta, dbeta, gamma, dgamma, K, V, dV, ddX)
dt = [D_(lambda Y: vars31(Y)[q], X0, 0) for q in range(3)]

# 4D quantities
g4 = metric4(X0); g4i = np.linalg.inv(g4)
sqrtg4 = lambda Y: np.sqrt(-np.linalg.det(metric4(Y)))
Fup = lambda Y: np.linalg.inv(metric4(Y)) @ F_f(Y) @ np.linalg.inv(metric4(Y))
divF = sum(D_(lambda Y: sqrtg4(Y) * Fup(Y)[m, :], X0, m) for m in range(4)) / sqrtg4(X0)
A0 = A_f(X0); Aup = g4i @ A0
R = divF - mu**2 * Aup                                  # R^b = nabla_a F^{ab} - mu^2 A^b
Lor = sum(D_(lambda Y: sqrtg4(Y) * (np.linalg.inv(metric4(Y)) @ A_f(Y))[m], X0, m) for m in range(4)) / sqrtg4(X0)
n4 = n_up(X0); n4_dn = g4 @ n4
proj = R[1:] + n4[1:] * (n4_dn @ R)                     # gamma^i_b R^b
sc = np.max(np.abs(R)) + 1
ok &= report('d_t X_i - RHS', np.max(np.abs(dt[0] - rhs['X'])) / sc)
ok &= report('d_t E^i - RHS - alpha gamma^i_b R^b', np.max(np.abs(dt[1] - rhs['E'] - alpha * proj)) / sc)
ok &= report('d_t X_phi - RHS - alpha nabla_a A^a', abs(dt[2] - rhs['Xphi'] - alpha * Lor) / sc)
ok &= report('Gauss constraint - n_b R^b', abs(src['Gauss'] - n4_dn @ R) / sc)
print('    (for scale: |R^b| = %.2f, |nabla_a A^a| = %.2f)' % (np.max(np.abs(R)), abs(Lor)))

# stress tensor
F4 = F_f(X0); F4mix = F4 @ g4i                           # F_a^c
Lm = -0.25 * np.real(np.sum(F4 * np.conj(g4i @ F4 @ g4i))) - 0.5 * mu**2 * np.real(A0 @ np.conj(Aup))
T = (0.5 * np.real(F4 @ g4i @ np.conj(F4).T + np.conj(F4) @ g4i @ F4.T)
     + 0.5 * mu**2 * np.real(np.outer(A0, np.conj(A0)) + np.outer(np.conj(A0), A0)) + g4 * Lm)
proj4 = np.eye(4) + np.outer(n4, n4_dn)                  # gamma^a_b
rho4 = n4 @ T @ n4
S4 = -(proj4.T @ T @ n4)[1:]
Sij4 = (proj4.T @ T @ proj4)[1:, 1:]
ok &= report('rho vs n^a n^b T_ab (rho = %.3f)' % rho4, abs(src['rho'] - rho4))
ok &= report('S_i vs -gamma_i^a n^b T_ab', np.max(np.abs(src['Si'] - S4)))
ok &= report('S_ij vs gamma_i^a gamma_j^b T_ab', np.max(np.abs(src['Sij'] - Sij4)))
jup = np.imag((g4i @ F4 @ g4i) @ np.conj(A0))               # j^a = Im(F^{ab} Abar_b)
ok &= report('n vs -n_a j^a', abs(src['n'] - (-(n4_dn @ jup))))

# ------------------------------------------------------------------------------------------------- B
print('B. Reflections (parities: X_k, E^k odd in x_k, X_phi and Z even):')
Vr = dict(X=Xv, E=Ev, Xphi=Pv, Z=0.3 + 0.1j)
dVr = dict(dV); dVr['Z'] = np.array([0.1, -0.2, 0.05]) * (1 + 0.5j)
rhs0, src0 = proca_rhs_and_sources(mu, 0.7, alpha, dalpha, beta, dbeta, gamma, dgamma, K, Vr, dVr, ddX)
for ax in range(3):
    P = np.eye(3); P[ax, ax] = -1                       # x' = P x; tensors pick up one P per index
    T1 = lambda v: P @ v
    T2 = lambda m: P @ m @ P
    rhsm, srcm = proca_rhs_and_sources(mu, 0.7, alpha, P @ dalpha, P @ beta, P @ dbeta @ P, T2(gamma),
                                       np.einsum('k,kij->kij', np.diag(P), np.array([T2(dgamma[k]) for k in range(3)])),
                                       T2(K), dict(X=P @ Xv, E=P @ Ev, Xphi=Pv, Z=Vr['Z']),
                                       dict(X=P @ dV['X'] @ P, E=P @ dV['E'] @ P, Xphi=P @ dV['Xphi'], Z=P @ dVr['Z']),
                                       np.einsum('a,b,c,abc->abc', np.diag(P), np.diag(P), np.diag(P), ddX))
    err = max(np.max(np.abs(rhsm['X'] - P @ rhs0['X'])), np.max(np.abs(rhsm['E'] - P @ rhs0['E'])),
              abs(rhsm['Xphi'] - rhs0['Xphi']), abs(rhsm['Z'] - rhs0['Z']), abs(srcm['rho'] - src0['rho']),
              np.max(np.abs(srcm['Si'] - P @ src0['Si'])), np.max(np.abs(srcm['Sij'] - T2(src0['Sij']))),
              abs(srcm['n'] - src0['n']))
    ok &= report('mirror image in %s' % 'xyz'[ax], err, 1e-13)

# ------------------------------------------------------------------------------------------------- C
print('C. Spherical data vs the 1D equations of Spherical_BS_Evolver (ComplexVectorField, D = 4):')
# smooth radial profiles: chi(r), h_zz(r) (radial), h_ww(r) (tangential), alpha, beta (radial), K_rr, K_TT, fields
prof = dict(chi=lambda r: 0.8 + 0.1 * np.exp(-r**2), hzz=lambda r: 1 + 0.05 * r**2 * np.exp(-r**2),
            hww=lambda r: 1 - 0.03 * r**2 * np.exp(-r**2), alpha=lambda r: 0.9 - 0.1 * np.exp(-r**2),
            beta=lambda r: 0.05 * r * np.exp(-r**2), Krr=lambda r: 0.02 + 0.01 * r**2, KTT=lambda r: -0.03 + 0.02 * r**2,
            X=lambda r: (0.3 + 0.2j) * r * np.exp(-r**2), E=lambda r: (0.1 - 0.4j) * r * np.exp(-0.5 * r**2),
            P=lambda r: (0.5 + 0.1j) * np.exp(-r**2), Z=lambda r: (0.02 - 0.01j) * np.exp(-r**2))


def fields3(x):
    r = np.linalg.norm(x); nn = x / r
    gzz, gww = prof['hzz'](r) / prof['chi'](r), prof['hww'](r) / prof['chi'](r)
    gamma = gww * np.eye(3) + (gzz - gww) * np.outer(nn, nn)
    K = prof['KTT'](r) * np.eye(3) + (prof['Krr'](r) - prof['KTT'](r)) * np.outer(nn, nn)   # K_rr, K_TT ~ K_thth/r^2
    # radial covector X_z dr -> X_i = X_z n_i;  E^z d_r -> E^i = E^z n^i
    return dict(alpha=prof['alpha'](r), beta=prof['beta'](r) * nn, gamma=gamma, K=K, X=prof['X'](r) * nn,
                E=prof['E'](r) * nn, Xphi=prof['P'](r), Z=prof['Z'](r))


x0 = np.array([0.4, -0.3, 0.5]); r0 = np.linalg.norm(x0)
f0 = fields3(x0)
d3 = lambda key, k: D_(lambda y: fields3(y)[key], x0, k)
dalpha3 = np.array([d3('alpha', k) for k in range(3)]); dbeta3 = np.array([d3('beta', k) for k in range(3)])
dgamma3 = np.array([d3('gamma', k) for k in range(3)])
dV3 = dict(X=np.array([d3('X', k) for k in range(3)]), E=np.array([d3('E', k) for k in range(3)]),
           Xphi=np.array([d3('Xphi', k) for k in range(3)]), Z=np.array([d3('Z', k) for k in range(3)]))
ddX3 = np.array([[D_(lambda y: D_(lambda w: fields3(w)['X'], y, l), x0, k) for l in range(3)] for k in range(3)])
rhs3, src3 = proca_rhs_and_sources(mu, 0.7, f0['alpha'], dalpha3, f0['beta'], dbeta3, f0['gamma'], dgamma3, f0['K'],
                                   dict(X=f0['X'], E=f0['E'], Xphi=f0['Xphi'], Z=f0['Z']), dV3, ddX3)


def rhs1d(r):
    """ComplexVectorField::rhs_CVF transcribed for one complex field (complex arithmetic), D = 4."""
    d = lambda key: D_(lambda y: prof[key](y[0]), np.array([r]), 0)
    chi, hzz, hww = prof['chi'](r), prof['hzz'](r), prof['hww'](r)
    g_zz, g_ZZ = hzz / chi, chi / hzz
    dchi, dhzz, dhww = d('chi'), d('hzz'), d('hww')
    d_g_ZZ = dchi / hzz - chi * dhzz / hzz**2
    alpha, beta = prof['alpha'](r), prof['beta'](r)
    K = prof['Krr'](r) / g_zz + 2 * prof['KTT'](r) / (hww / chi)
    dln = 0.5 * dhzz / hzz + dhww / hww - 1.5 * dchi / chi
    div = lambda V, dV: dV + 2 * V / r + V * dln
    X, E, P, Z = prof['X'](r), prof['E'](r), prof['P'](r), prof['Z'](r)
    dX, dE, dP, dZ = d('X'), d('E'), d('P'), d('Z')
    X_up, dX_up = g_ZZ * X, d_g_ZZ * X + g_ZZ * dX
    rX = beta * dX + X * d('beta') - alpha * (g_zz * E + dP) - P * d('alpha')
    rE = beta * dE - E * d('beta') + alpha * K * E + alpha * mu**2 * X_up + alpha * g_ZZ * dZ
    rP = beta * dP + alpha * K * P - alpha * div(X_up, dX_up) - X_up * d('alpha') - alpha * Z
    rZ = beta * dZ + alpha * (div(E, dE) + mu**2 * P - 0.7 * Z)
    rho = 0.5 * g_zz * abs(E)**2 + 0.5 * mu**2 * (abs(P)**2 + g_ZZ * abs(X)**2)
    jz = mu**2 * np.real(P * np.conj(X))
    Szz = -0.5 * g_zz**2 * abs(E)**2 + mu**2 * (abs(X)**2 - 0.5 * g_zz * (g_ZZ * abs(X)**2 - abs(P)**2))
    Sww = 0.5 * (hww / chi) * g_zz * abs(E)**2 - 0.5 * mu**2 * (hww / chi) * (g_ZZ * abs(X)**2 - abs(P)**2)
    return dict(X=rX, E=rE, Xphi=rP, Z=rZ, rho=rho, jz=jz, Szz=Szz, Sww=Sww, n=np.imag(E * np.conj(X)),
                Gauss=div(E, dE) + mu**2 * P)


o = rhs1d(r0); nn = x0 / r0
tang = np.cross(nn, [0, 0, 1.]); tang /= np.linalg.norm(tang)
gww = prof['hww'](r0) / prof['chi'](r0)
errs = [np.max(np.abs(rhs3['X'] - o['X'] * nn)), np.max(np.abs(rhs3['E'] - o['E'] * nn)),
        abs(rhs3['Xphi'] - o['Xphi']), abs(rhs3['Z'] - o['Z'])]
ok &= report('RHS of X_z, E^z, X_phi, Z', max(errs))
ok &= report('rho, j_z, S_zz, S_ww, Gauss, n',
             max(abs(src3['rho'] - o['rho']), abs(src3['Si'] @ nn - o['jz']), abs(nn @ src3['Sij'] @ nn - o['Szz']),
                 abs(tang @ src3['Sij'] @ tang - o['Sww']), abs(src3['Gauss'] - o['Gauss']), abs(src3['n'] - o['n'])))
print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
