"""
Compares Source/Matter/DiracField.hpp with the Python reference (dirac3d_ref.dirac_rhs_and_sources) at random points.
    g++ -O2 -std=c++14 -DCH_SPACEDIM=3 -Istub -I../../../Source/Matter -I$GRCHOMBO_SOURCE/utils \
        -I$GRCHOMBO_SOURCE/CCZ4 test_dirac_cpp.cpp -o test_dirac_cpp
    python3 test_dirac_cpp.py ./test_dirac_cpp
"""
import os, sys, subprocess
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'derivations'))
from dirac3d_ref import dirac_rhs_and_sources

exe = sys.argv[1] if len(sys.argv) > 1 else './test_dirac_cpp'
rng = np.random.default_rng(17)
sym6 = lambda M: [M[0, 0], M[0, 1], M[0, 2], M[1, 1], M[1, 2], M[2, 2]]
lines, refs = [], []
for trial in range(200):
    sym = lambda s: (lambda m: (m + m.T) / 2)(rng.normal(size=(3, 3)) * s)
    chi = rng.uniform(0.3, 1.2); h = np.eye(3) + sym(0.15); Kt = rng.normal() * 0.3; A = sym(0.2)
    lapse = rng.uniform(0.3, 1.0); shift = rng.normal(size=3) * 0.2
    dchi = rng.normal(size=3) * 0.3; dh = [sym(0.3) for k in range(3)]; dlapse = rng.normal(size=3) * 0.3
    dshift = rng.normal(size=(3, 3)) * 0.2                    # [k, i] = d_k beta^i
    D = rng.normal(size=8) + 1j * rng.normal(size=8)
    dD = [rng.normal(size=8) + 1j * rng.normal(size=8) for k in range(3)]
    mu = rng.uniform(0.5, 1.5)
    # physical quantities for the reference
    gamma = h / chi
    dgamma = np.array([dh[k] / chi - h * dchi[k] / chi**2 for k in range(3)])
    K = (A + h * Kt / 3) / chi
    rD, rho, Si, Sij, n, imT, geo = dirac_rhs_and_sources(mu, lapse, dlapse, shift, dshift, gamma, dgamma, K, D, dD)
    adv = sum(shift[k] * dD[k] for k in range(3))
    rr = rD - adv
    S = np.trace(np.linalg.inv(gamma) @ Sij)
    refs.append(np.concatenate([np.ravel(np.column_stack([rr.real, rr.imag])), [rho], Si, sym6(Sij), [S, n]]))
    vals = [chi] + sym6(h) + [Kt] + sym6(A) + [lapse] + list(shift)
    for k in range(3):
        vals += [dchi[k]] + sym6(dh[k]) + [dlapse[k]] + list(dshift[k])
    vals += list(D.real) + list(D.imag)
    for k in range(3):
        vals += list(dD[k].real) + list(dD[k].imag)
    vals += [mu]
    lines.append(' '.join('%.17e' % x for x in vals))
out = subprocess.run([exe], input='\n'.join(lines) + '\n', capture_output=True, text=True, check=True).stdout
cpp = np.array([[float(x) for x in l.split()] for l in out.strip().split('\n')])
ref = np.array(refs)
err = np.max(np.abs(cpp - ref) / (1 + np.abs(ref)))
print('C++ vs Python reference at %d random points: max relative error %.1e  %s' % (len(ref), err, 'OK' if err < 1e-11 else 'FAILED'))
