"""
Compares Source/Matter/ProcaField.hpp with the Python reference (proca3d_ref.proca_rhs_and_sources) at random points.
    g++ -O2 -std=c++14 -DCH_SPACEDIM=3 -Istub -I../../../Source/Matter -I$GRCHOMBO_SOURCE/utils \
        -I$GRCHOMBO_SOURCE/CCZ4 test_proca_cpp.cpp -o test_proca_cpp
    python3 test_proca_cpp.py ./test_proca_cpp
"""
import os, sys, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'derivations'))
from proca3d_ref import proca_rhs_and_sources

exe = sys.argv[1] if len(sys.argv) > 1 else './test_proca_cpp'
rng = np.random.default_rng(19)
sym6 = lambda M: [M[0, 0], M[0, 1], M[0, 2], M[1, 1], M[1, 2], M[2, 2]]
cv = lambda n: rng.normal(size=n) + 1j * rng.normal(size=n)
lines, refs = [], []
for trial in range(200):
    sym = lambda s: (lambda m: (m + m.T) / 2)(rng.normal(size=(3, 3)) * s)
    chi = rng.uniform(0.3, 1.2); h = np.eye(3) + sym(0.15); Kt = rng.normal() * 0.3
    A = sym(0.2); A -= h * np.trace(np.linalg.inv(h) @ A) / 3      # trace free, as in CCZ4
    lapse = rng.uniform(0.3, 1.0); shift = rng.normal(size=3) * 0.2
    dchi = rng.normal(size=3) * 0.3; dh = [sym(0.3) for k in range(3)]; dlapse = rng.normal(size=3) * 0.3
    dshift = rng.normal(size=(3, 3)) * 0.2                    # [k, i] = d_k beta^i
    X, E, P, Z = cv(3), cv(3), cv(1)[0], cv(1)[0]
    dX, dE, dP, dZ = cv(9).reshape(3, 3), cv(9).reshape(3, 3), cv(3), cv(3)   # [k, i] = d_k
    ddX = cv(27).reshape(3, 3, 3); ddX = 0.5 * (ddX + np.transpose(ddX, (1, 0, 2)))   # [j, k, i] = d_j d_k X_i
    mu, kappa = rng.uniform(0.5, 1.5), rng.uniform(0, 2)
    gamma = h / chi
    dgamma = np.array([dh[k] / chi - h * dchi[k] / chi**2 for k in range(3)])
    K = (A + h * Kt / 3) / chi
    rhs, src = proca_rhs_and_sources(mu, kappa, lapse, dlapse, shift, dshift, gamma, dgamma, K,
                                     dict(X=X, E=E, Xphi=P, Z=Z), dict(X=dX, E=dE, Xphi=dP, Z=dZ), ddX)
    nadv = lambda r, d: r - shift @ d                          # the C++ omits beta^k d_k
    rX, rE = nadv(rhs['X'], dX), nadv(rhs['E'], dE)
    rP, rZ = rhs['Xphi'] - shift @ dP, rhs['Z'] - shift @ dZ
    refs.append(np.concatenate([rX.real, rX.imag, rE.real, rE.imag, [rP.real, rP.imag, rZ.real, rZ.imag],
                                [src['rho']], src['Si'], sym6(src['Sij']), [src['S'], src['n'], src['Gauss'].real,
                                                                           src['Gauss'].imag]]))
    vals = [chi] + sym6(h) + [Kt] + sym6(A) + [lapse] + list(shift)
    for k in range(3):
        vals += [dchi[k]] + sym6(dh[k]) + [dlapse[k]] + list(dshift[k])
    vals += list(X.real) + list(X.imag) + list(E.real) + list(E.imag) + [P.real, P.imag, Z.real, Z.imag]
    for k in range(3):
        vals += list(dX[k].real) + list(dX[k].imag) + list(dE[k].real) + list(dE[k].imag)
        vals += [dP[k].real, dP[k].imag, dZ[k].real, dZ[k].imag]
    for j in range(3):
        for k in range(3):
            vals += list(ddX[j, k].real) + list(ddX[j, k].imag)
    vals += [mu, kappa]
    lines.append(' '.join('%.17e' % x for x in vals))
out = subprocess.run([exe], input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True).stdout
cpp = np.array([[float(x) for x in l.split()] for l in out.strip().split('\n')])
ref = np.array(refs)
err = np.max(np.abs(cpp - ref) / (1 + np.abs(ref)))
print('C++ vs Python reference at %d random points: max relative error %.1e  %s' % (len(ref), err, 'OK' if err < 1e-11 else 'FAILED'))
