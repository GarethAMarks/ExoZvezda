# Proca stars in 3D (SingleProcaStar, BinaryProcaStar)

These examples evolve the Proca stars of Spherical_BS_Evolver (`proca-stars` branch) and 2DChombo (`proca-stars`
branch) in full 3D: a complex Proca field of mass mu coupled to CCZ4. The conventions are the same, with signature
(-,+,+,+) and K_ij = -nabla_i n_j:

    L = -1/4 F_ab Fbar^ab - 1/2 mu^2 A_a Abar^a,     A = e^{-i omega t} (f dt + i g dr)   for the star.

## Files

| File | Contents |
|---|---|
| `Source/Matter/ProcaField.hpp` | Matter class: grid variables, 3+1 Proca right-hand side, stress tensor, number density and Gauss constraint. It implements the GRChombo matter interface (`compute_emtensor`, `add_matter_rhs`), so the example uses `MatterCCZ4RHS<ProcaField>` directly. |
| `Source/Matter/ProcaConstraints.hpp` | Ham and Mom with the Proca sources, plus the diagnostics `N`, `mod_Xphi`, `rho`, `n` and `Gauss` (see below) |
| `Source/TaggingCriteria/ProcaAndChiExtractionTaggingCriterion.hpp` | Tagging on X_phi and chi, as `ComplexPhiAndChiExtractionTaggingCriterion` |
| `Source/InitialConditions/ProcaStars/ProcaStarSolution.hpp` | Port of the 1D solver: shooting, isotropic transformation, Gaussian perturbation, mini/quartic/solitonic potentials. Identical to the 2DChombo file. |
| `Source/InitialConditions/ProcaStars/ProcaStarBoost.hpp` | Boosted star: lab metric, lapse, shift and K_ij as for the boson stars, plus the transformation of A and F (shared with 2DChombo) |
| `Source/InitialConditions/ProcaStars/SingleProcaStar.hpp` | Initial data for a single star, at rest or boosted along x (`PS_rapidity`) |
| `Source/InitialConditions/ProcaStars/BinaryProcaStar.hpp` | Binaries, combined as `BinaryPlainSuperposition`, `BinaryEqualMassFix` and `BinaryUnequalMassFix` (`id_choice` 0, 1, 2) |
| `Examples/SingleProcaStar/` | Level, `params.txt` (quadrant symmetry, as SingleBosonStar), variables |
| `Examples/BinaryProcaStar/` | Head-on collision example (z-reflection, as the BBS examples) |
| `derivations/` | `check_proca3d.py`: numerical checks of the equations; `check_proca_boost.py`: derivation and checks of the boost; `proca3d_ref.py`: Python reference implementation |
| `tests/` | C++ vs Python test (`test_proca_cpp.*`, no Chombo needed) and a single-box test of the assembled code (`box_test/`) |

## Equations

**Variables.** Following Zilhão, Witek & Cardoso (2015), as in the 1D code: X_phi = -n^a A_a, X_i = gamma_i^a A_a,
E^i = gamma^{ij} F_{ja} n^a, and the Gauss-constraint damping variable Z. Each is complex, giving 16 real grid
variables (`Xx_Re ... Z_Im`).

**Evolution.** With F_ij = d_i X_j - d_j X_i and D_j(alpha F^{ij}) = (1/sqrt(gamma)) d_j(sqrt(gamma) alpha F^{ij}):

    d_t X_i   = L_beta X_i - alpha (E_i + D_i X_phi) - X_phi D_i alpha
    d_t E^i   = L_beta E^i + alpha K E^i + D_j(alpha F^{ij}) + alpha mu^2 X^i + alpha D^i Z
    d_t X_phi = L_beta X_phi + alpha K X_phi - alpha D_i X^i - X^i D_i alpha - alpha Z
    d_t Z     = L_beta Z + alpha (D_i E^i + mu^2 X_phi - kappa Z)

- The curl term `D_j(alpha F^{ij})` is the only addition to the 1D equations, where F_ij vanishes. It needs the second
  derivatives of X_i, which `ProcaField::Diff2Vars` requests.
- The Gauss constraint is C = D_i E^i + mu^2 X_phi. With Z, the constraint violations obey a damped wave equation with
  damping `proca_damping` (kappa, default 1 as in 1D).

**Sources.** Summed over the real and imaginary parts, with B^2 = F_ij F^ij / 2:

    rho  = (E_i E^i + B^2)/2 + mu^2 (X_phi^2 + X_i X^i)/2,      S_i = F_ij E^j + mu^2 X_phi X_i,
    S_ij = -E_i E_j + F_ik F_jl gamma^kl + mu^2 X_i X_j + gamma_ij [E^2/2 - B^2/2 - mu^2 (X_k X^k - X_phi^2)/2].

None of these need the lapse or shift.

**Free field only.** As in Spherical_BS_Evolver, only the free field is evolved. The solver still builds stars with
the quartic and solitonic potentials (for families), but the examples stop with an error if `proca_lambda != 0` or
`proca_solitonic = 1`.

**Reflections.** X_k and E^k are odd in x_k and everything else is even, so the usual GRChombo parities (1 2 3 and 0)
work. No variables are swapped, unlike the Dirac case.

**The star at rest**, with f, g, E = f' - omega g the isotropic profiles of the 1D solver and theta = phase pi:

    X_phi = -f e^{i theta}/alpha,   X_i = i g n_i e^{i theta},   E^i = E n_i e^{i theta}/(alpha psi^4),   Z = 0

so that |X_phi(0)| = f(0)/alpha(0) = A_central.

**Verification.** `derivations/check_proca3d.py` uses an arbitrary metric, gauge and complex A_a (no field equations
imposed) and checks:

- the 3+1 equations against the covariant 4D theory:
  - d_t X_i - RHS = 0;
  - d_t E^i - RHS = alpha gamma^i_b (nabla_a F^{ab} - mu^2 A^b);
  - d_t X_phi - RHS = alpha nabla_a A^a;
  - C = n_b (nabla_a F^{ab} - mu^2 A^b);
  - all to 2e-11 or better;
- the sources against T_ab, and n against -n_a j^a with j^a = Im(F^{ab} Abar_b) (1e-16);
- that mirrored data evolve as the mirror image for each of x, y and z (exact);
- the reduction to Spherical_BS_Evolver's `ComplexVectorField` (RHS, sources, Gauss, n) for spherical data (1e-12).

## Boosted stars and binaries

**Coordinates.** As for the boson and Dirac stars:

    t~ = cosh(xi) t + sinh(xi) x,   x~ = sinh(xi) t + cosh(xi) x

so a star with rapidity xi > 0 moves towards -x. The lab 3-metric, lapse, shift and K_ij are exactly those of
`BosonStarHelperFunction`, with its conformal factor P = psi^2.

**The field.** A_mu is a covector and F_mu nu a 2-form, so the lab components follow from the coordinate Jacobian
alone. Unlike the spinor, there is no local Lorentz transformation of a frame. In the rest frame,
A = e^{-i omega t~}(f dt~ + i g n~_i dx~^i), F_{t~ i~} = -E n~_i e^{-i omega t~} and F_{i~ j~} = 0. In the lab, with
e_i = F_{t~ i~}:

    A_t = cosh A_t~ + sinh A_x~,   A_x = sinh A_t~ + cosh A_x~,   A_y, A_z unchanged,
    F_tx = e_x,   F_ty = cosh e_y,   F_tz = cosh e_z,   F_xy = sinh e_y,   F_xz = sinh e_z,   F_yz = 0.

The 3+1 variables then follow from the lab slicing:

    X_i = A_i,   X_phi = -(A_t - beta^x A_x)/lapse,
    E_x = -e_x/lapse,   E_{y,z} = -(cosh - beta^x sinh) e_{y,z}/lapse,   E^i = gamma^{ij} E_j.

A boosted star carries a magnetic field (F_xy, F_xz) and a Poynting flux.

**Checks.** `derivations/check_proca_boost.py` uses arbitrary profiles and t != 0, and checks:

- the closed forms against the generic tensor transformation, including K_ij from the time derivative of the lab
  metric (2e-12);
- that the lab right-hand side with the closed-form data equals the exact lab time derivatives up to the rest-frame
  residuals of the field equations (1e-12);
- that the x-reflection maps rapidity xi to -xi with X_x and E^x odd, so equal-mass head-on binaries are mirror
  symmetric (exact).

**Binaries.** X_phi, X_i and E^i of the two stars are added (Z = 0). The metrics are combined as for the boson-star
binaries.

- The superposition violates the Gauss constraint where the stars overlap, as it does the Hamiltonian constraint; Z
  damps this.
- Both stars populate the same field, so their relative phase (`phase`, `phase2`) matters where they overlap.

## Tests run so far (no 3D evolutions)

- **C++ vs Python reference** at 200 random points: 5e-15.
- **Ported 1D solver vs Spherical_BS_Evolver** at A = 0.005, 0.02 and 0.05:
  - M, N, omega and r_99 are identical (12 digits);
  - the isotropic profiles, including the perturbed ones, agree to the precision of the 1D output (5e-10 for psi,
    5e-12 for f, g, E).
- **Single-box test (`tests/box_test`)**, initial data plus one RHS and constraint evaluation on [-1.5, 1.5]^3 around the
  star:

  | Star | Quantity | dx = 0.1 | dx = 0.05 |
  |---|---|---|---|
  | A = 0.02 | \|d_t Proca + i omega Proca\| / \|omega Proca\| | 3.8e-7 | 2.4e-8 |
  | A = 0.02 | metric RHS | 9.6e-9 | 6.8e-10 |
  | A = 0.02 | Ham / 16 pi rho | 5.7e-7 | 4.0e-8 |
  | A = 0.02 | Gauss / \|mu^2 X_phi\| | 3.4e-7 | 2.2e-8 |
  | A = 0.06 (unstable) | \|d_t Proca + i omega Proca\| / \|omega Proca\| | 1.0e-5 | 6.3e-7 |
  | A = 0.06, perturbed | Ham / 16 pi rho | 2.4e-6 | 1.5e-7 |
  | A = 0.06, perturbed | Gauss / \|mu^2 X_phi\| | 6.2e-6 | 3.9e-7 |

  - These converge at 4th order. For the perturbed star d_t stays at a resolution-independent 3.3e-3, as it should.
  - The diagnostic n agrees with -g E/(alpha psi^4) from the 1D profiles to 1e-15.
- **Boosted star (rapidity 0.3, `box_test` mode 1, 1D solution with 120001 points):** the RHS reproduces the exact time
  derivative of the moving solution, and the constraints converge to zero:

  | Quantity | dx = 0.1 | dx = 0.05 | factor |
  |---|---|---|---|
  | d_t Proca error | 5.2e-7 | 3.1e-8 | 16.8 |
  | Ham / 16 pi rho | 5.9e-7 | 3.7e-8 | 16.0 |
  | Mom | 2.4e-9 | 1.6e-10 | 15.2 |
  | Gauss / \|mu^2 X_phi\| | 5.9e-7 | 3.7e-8 | 16.0 |

  With the default 30001 points (dr = 0.005) Mom has a floor of about 3e-10 from the interpolated psi' and alpha'.
- **Binary (mode 2):** for an equal-mass head-on binary all three `id_choice` options give data that are exactly
  mirror-symmetric, with X_x and E^x odd.
- **Two-axis aspherical perturbation (mode 3):** m = 1 in the x-y plane and n = 2 in the x-z plane, with
  delta = epsilon = 0.05.
  - The Proca variables are exactly F times the unperturbed ones, and the metric is unchanged.
  - Mom stays zero (1e-17).
  - Away from the two lines where phi and phi' are undefined, Gauss / |mu^2 X_phi| converges: 3.3e-4 -> 2.5e-5.
  - Next to those lines, Ham / 16 pi rho and Gauss grow with resolution. This is the discontinuity of Eq. (11)
    described under Perturbations.
- **Smoothed perturbation (mode 4):** the same with `smooth = true`. Maxima for r > 0.5, next to the lines (within 0.3
  of them) or anywhere:

  | | mode 3, dx = 0.1 | mode 3, dx = 0.05 | mode 4, dx = 0.1 | mode 4, dx = 0.05 |
  |---|---|---|---|---|
  | Ham / 16 pi rho, next to the lines | 0.317 | 0.573 | 0.166 | 0.168 |
  | Gauss / \|mu^2 X_phi\|, r > 0.5 | 2.0e-2 | 3.6e-2 | 4.4e-4 | 4.0e-5 |

  - Smoothed, Gauss converges everywhere except in the few cells around the centre, and Ham is the
    resolution-independent O(delta) violation of the unsolved constraint.
  - Next to the centre, Gauss / |mu^2 X_phi| stays O(delta) in both modes (0.03 -> 0.08 smoothed, 0.03 -> 0.05
    unsmoothed). There X_phi does not vanish and F stays direction-dependent; Z damps this.
- **Parameters:** the example runs its initial data with m = 1, n = 8 in the quadrant. It stops with an error for an
  odd m on a reflective first axis and for aspherical perturbations of binaries.
- **Compilation and initial data:**
  - Both examples compile against GRChombo (main branch) and Chombo (3D, MPI, gcc).
  - Both build their initial data and write all diagnostics with `max_steps = 0`.
  - For the default star in the quadrant (coarse grid, dx = 2 and 1.3), `ParticleNumber` equals the 1D Noether charge
    N = 1.053243 to 9 digits, and the L2 norms of Ham and Gauss fall with resolution.

## The family (mu = G = 1)

- **Stability:** the ground state is perturbatively stable for A_c < 0.0374 (f(0) < 0.0274), up to the maximum mass
  M = 1.0581 (N = 1.0880).
- **Binding energy:** E_b = M - mu N is most negative (-0.0299) at the maximum mass and changes sign at A_c = 0.141.
- **Next turning point:** the mass has a minimum (M = 0.496) at A_c = 1.50.
- **1D runs:** unstable stars at A = 0.06, 0.10 and 0.20 with a perturbation a = +0.01 formed an apparent horizon. With
  a = -0.01 they expanded off the unstable branch.

The data, the generator (`make_family.cpp`) and the plot script are in 2DChombo's `Examples/ProcaStar_2D/family/`.

## Diagnostics

All of these are plottable (`plot_vars`).

| Name | Meaning |
|---|---|
| `n` | particle number density measured by the normal observers, n = -n_a j^a = Im(E^i Xbar_i) |
| `N` | n chi^{-3/2}, the integrand summed into `ParticleNumber` (the 1D Noether charge) |
| `mod_Xphi` | \|X_phi\|, which equals A_central at the centre of a star at rest |
| `rho` | energy density |
| `Gauss` | \|D_i E^i + mu^2 X_phi\| (its L2 norm is the last column of `constraint_norms`) |

There is a single (complex) field, so n is also the density of that field.

## Perturbations

`proca_perturb = 1` perturbs a single star. In this generalised formalism the azimuthal index
`proca_perturb_m` selects the perturbation, and the spherical perturbation of Spherical_BS_Evolver is its m = 0 member.

- **m = 0 (spherical):** the 1D solver applies (f, g) -> (1 + a exp(-(r - r0)^2 / w^2)) (f, g) and re-solves the
  constraints (metric and Gauss law), as before. Here a = `proca_perturb_amp`, w = `proca_perturb_spread` and
  r0 = `proca_perturb_center`.
- **m >= 1 (aspherical):** the perturbation of arXiv:2609.30021, Eq. (11). The matter fields (X_phi, X_i and E^i) are multiplied
  by exp(-2 delta cos(m phi)), with delta = `proca_perturb_amp`.
  - phi = atan2(x_b, x_a) is the azimuthal angle about the star centre, with reference axes
    (a, b) = `proca_perturb_axes` (0 = x, 1 = y, 2 = z; default x, y).
- **Two-axis perturbations:** a second factor exp(-2 epsilon cos(n phi')), prepared in an orthogonal plane, is added
  with n = `proca_perturb_n` (0: none), epsilon = `proca_perturb_amp2` and the reference axes `proca_perturb_axes2`.
  These must share exactly one axis with `proca_perturb_axes` (default x, z). This is the configuration of the paper's
  runs IC and IIC, which leaves no symmetry axis.
- **Switching off:** `proca_perturb_aspherical` (default 0) must be 1 for the aspherical perturbations. With 0 they are
  disabled and the perturbation is spherical whatever m is.
- **Constraints:** as in the paper, they are not re-solved after an aspherical perturbation, so Ham is violated at
  O(delta).
  - For a Proca star at rest Gauss still holds wherever the factor is smooth, because E^i is radial and the factor
    depends only on angles.
  - phi is undefined on the line through the centre normal to its plane. The factor jumps there by up to
    exp(4 delta), so the fields become discontinuous at that line. The constraint violation next to it grows with
    resolution. This is inherent in Eq. (11).
- **Smoothed version:** `proca_perturb_smooth = 1` (default 0, the paper's convention) replaces each cos(m phi) by
  cos(m phi) sin^m(theta), with theta the angle from that line, i.e. by Re[(x_a + i x_b)^m] / r^m, the angular shape of
  the sectoral harmonic Y_mm.
  - The factor is then smooth everywhere except at the centre point itself (a direction-dependent value with finite
    gradient energy).
  - Gauss then converges everywhere except in the few cells around the centre, where X_phi is nonzero (see Tests).
  - It agrees with the unsmoothed factor in the plane of the reference axes. Like it, it depends only on angles.
  - Recommended unless the aim is to reproduce the paper's runs. The 2D cartoon codes do not need it: there the
    perturbation is axisymmetric and already smooth away from the centre.
- **Symmetries:**
  - An odd index is odd under the reflection of its first reference axis (phi -> pi - phi).
  - With the quadrant symmetry of `params.txt` (reflective y and z), odd m or n therefore need x as the first
    reference axis. The code checks the reflective boundaries and stops otherwise.
  - Example: the paper's IC run (m = 1, n = 8) is `proca_perturb_axes = 0 1` and `proca_perturb_axes2 = 0 2`.
- **Single stars only:** the binary example accepts only the spherical perturbation.

The code is in `Source/utils/AsphericalPerturbation.hpp`, shared with 2DChombo (where the perturbation is axisymmetric).

## Running

    cd Examples/SingleProcaStar && make all && mpirun ./Main_SingleProcaStar3d*.ex params.txt

**Initial-data parameters** (`central_amplitude_proca`, `proca_*`) mirror Spherical_BS_Evolver's `BSParams.par` with
`proca = 1`:

- `proca_perturb*` give the spherical and aspherical perturbations described below.
- `proca_solver_radius` should be at least about 4 r_99. Proca stars are larger than boson stars of the same mass:
  r_99 = 15 at A = 0.02.
- The frequency search brackets automatically.
- Beyond 0.8 `proca_solver_radius` the field is continued as exp(-sqrt(mu^2 - omega^2) r)/r. For a perturbed star this
  cuts off the small Coulomb-like tail of E, a tiny Gauss violation there.

**Binaries.** `PS_separation`, `PS_impact_parameter`, `PS_rapidity2`, `central_amplitude_proca2`, `phase2`,
`mass_ratio` and `id_choice` work as for the boson-star binaries.

**Outputs:**

- `ParticleNumber`;
- `mod_Xphi_max`, which equals A_central for a star at rest;
- `constraint_norms` (with `L^2_Gauss`) and `min_chi`, as for the boson stars.

**Dissipation.** The Proca system is first order in space and needs Kreiss-Oliger dissipation, which `MatterCCZ4RHS`
applies to all variables (`sigma = 0.3`). In 1D a larger value was needed at the origin, where the 3D code has no
coordinate singularity.

The grid, AMR and tagging values in `params.txt` (and `../BinaryProcaStar/params.txt`) are copied from the boson-star
examples and untested for Proca stars. Star tracking is not included in the binary example.
