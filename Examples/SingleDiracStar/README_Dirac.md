# Dirac stars in 3D (SingleDiracStar)

This example evolves the Dirac stars of Spherical_BS_Evolver (`dirac-stars` branch) and 2DChombo (`dirac-stars`
branch) in full 3D: two Dirac spinors of mass mu coupled to CCZ4. The conventions are the same, with signature
(-,+,+,+).

## Files

| File | Contents |
|---|---|
| `Source/Matter/DiracField.hpp` | Matter class: grid variables, 3+1 Dirac right-hand side, stress tensor. Implements the GRChombo matter interface (`compute_emtensor`, `add_matter_rhs`), so `MatterCCZ4RHS<DiracField>` works. |
| `Source/Matter/DiracCCZ4RHS.hpp` | The same RHS as `MatterCCZ4RHS<DiracField>` (bit-identical), with the Dirac part computed once per cell instead of twice |
| `Source/Matter/DiracConstraints.hpp` | Ham/Mom with the Dirac sources (needs the lapse and shift, unlike `MatterConstraints`); also `N` (particle-number integrand), `mod_psi` and `rho` |
| `Source/TaggingCriteria/DiracAndChiExtractionTaggingCriterion.hpp` | As `ComplexPhiAndChiExtractionTaggingCriterion` |
| `Source/InitialConditions/DiracStars/DiracStarSolution.hpp` | Port of the 1D solver (shooting, isotropic transformation, Gaussian perturbation); identical to the 2DChombo file |
| `Source/InitialConditions/DiracStars/SingleDiracStar.hpp` | Initial data for a single star, at rest or boosted along x (`DS_rapidity`) |
| `Source/InitialConditions/DiracStars/DiracStarBoost.hpp` | Boosted star: lab metric, lapse, shift and K_ij as for the boson stars, plus the spinor transformation (shared with 2DChombo) |
| `Source/InitialConditions/DiracStars/DiracStarHelperFunction.hpp` | The 3D variables (metric and the 16 spinor variables) of one boosted star, as `BosonStarHelperFunction` |
| `Source/InitialConditions/DiracStars/BinaryDiracStar.hpp` | Binaries, combined as `BinaryPlainSuperposition`, `BinaryEqualMassFix` and `BinaryUnequalMassFix` (`id_choice` 0, 1, 2) |
| `Examples/BinaryDiracStar/` | Head-on collision example (z-reflection, as the BBS examples) |
| `Examples/SingleDiracStar/` | Level, `params.txt` (quadrant symmetry, as SingleBosonStar), variables |
| `derivations/` | `check_dirac3d.py`: numerical checks of the equations; `check_dirac_boost.py`: derivation and checks of the boost; `dirac3d_ref.py`: Python reference implementation |
| `tests/` | C++ vs Python test (`test_dirac_cpp.*`, no Chombo needed) and a single-box test of the assembled code (`box_test/`) |

## Equations

**Tetrad.** Time gauge e_0 = n and the Cholesky (Gram-Schmidt) triad, gamma_ij = L L^T with L lower triangular.

- It is smooth everywhere, needs no extra variables, and has closed-form derivatives.
- It is covariant under the coordinate reflections, so reflective boundaries work.
- For the conformally flat initial data it equals the symmetric triad used in 1D/2D.

**Evolution equation.**

    d_t psi = beta^k d_k psi - alpha Gamma_n psi + alpha gamma^0 gamma^b e_b^k (d_k + Gamma_k) psi + alpha mu gamma^0 psi

The spin connection includes the rotation of the triad as gamma_ij evolves. The sources are T_A + T_B, with the time
derivatives eliminated through the equations.

**Grid variables.** Both spinors are evolved: psi_A (m_j = +1/2) and psi_B (m_j = -1/2), 16 real variables. They are
stored as eight complex combinations `D_e, D_x, D_y, D_z, D_xy, D_yz, D_xz, D_xyz` of (psi_A, psi_B).

- Each combination is odd exactly in the directions named by its suffix (GRChombo parities 0-7 in this order).
- The reflections act on the spinors as R_x = -tau_x (x) gamma5 gamma^x, R_y = tau_y (x) gamma5 gamma^y and
  R_z = -tau_z (x) gamma5 gamma^z. The y and x reflections swap m_j = +1/2 and -1/2.
- In the D basis the reflections are diagonal, so `lo_boundary = 1 2 2` (and x-reflection for head-on binaries)
  works with ordinary parities.
- For the star at rest, with f, g the physical radial spinor functions of the 1D solver (g(0) = A_central):

      D_e = g/sqrt(2 pi),   D_x = -i f n_x/sqrt(2 pi),   D_y = f n_y/sqrt(2 pi),   D_z = -i f n_z/sqrt(2 pi)

  and the other D vanish.

**Verification.** `derivations/check_dirac3d.py` checks:

- the 3+1 form against the covariant 4D Dirac operator for arbitrary metric functions (3e-14);
- that mirrored data evolve as the mirror image for each of x, y and z (exact), and that the D parities are as named;
- reduction to the 1D equations and sources for spherical data (1e-14).

## Boosted stars and binaries

**Coordinates.** As for the boson stars:

    t~ = cosh(xi) t + sinh(xi) x,   x~ = sinh(xi) t + cosh(xi) x

so a star with rapidity xi > 0 moves towards -x. The lab 3-metric, lapse, shift and K_ij are exactly those of
`BosonStarHelperFunction`, with its conformal factor P = psi^2.

**The spinor.** The spinor is a coordinate scalar, but its components refer to the tetrad.

- The lab tetrad (e_0 = n, e_x along d_x) is the star's rest-frame orthonormal tetrad boosted in the 0-x plane, with
  a local rapidity eta given by

      tanh(eta) = tanh(xi) alpha(r~) / psi(r~)^2

  This is the velocity of the lab's normal observers relative to the star, measured in the star's frame.
- So the lab components are

      psi_lab(t, x) = U(eta) e^{-i omega t~} psi_0(x~),   U(eta) = exp((eta/2) gamma^0 gamma^1) = cosh(eta/2) - sinh(eta/2) alpha_x

  with alpha_x = gamma_D^0 gamma_D^1. A scalar field only picks up the phase; the spinor also gets the local Lorentz
  transformation U.
- U commutes with the y and z reflections, and the x reflection maps U(eta) to U(-eta). So the grid-variable parities
  are unchanged, and equal-mass head-on binaries are mirror-symmetric about x = 0.

`derivations/check_dirac_boost.py` checks:

- the representation property of U (the other sign fails);
- that the covariant 4D Dirac operator computed in lab coordinates with the lab tetrad equals U times the operator
  computed in the rest frame, for an arbitrary background and spinor field (1.5e-13), while U(-eta) or no U fails at
  O(0.1);
- that the lab 3+1 right-hand side is consistent with this;
- the reflection properties above.

**Binaries.** Both stars' spinors are added and the metrics are combined as for the boson-star binaries. Since both
stars populate the same two spinor fields, the fields interfere where the stars overlap, as boson stars do, so the
relative phase (`phase`, `phase2`) matters. A binary of four independent spinors (one pair per star, which never
interfere) would need two more spinor pairs in `DiracField`.

## Tests run so far (no 3D evolutions)

- **C++ vs Python reference** at 200 random points: 4e-15.
- **Ported 1D solver:** identical to the one in 2DChombo, which reproduces Spherical_BS_Evolver (M, N, omega, r_99,
  isotropic and perturbed profiles).
- **Single-box test (`tests/box_test`)**, initial data plus one RHS and constraint evaluation on [-1.5, 1.5]^3 around
  the star:

  | Star | Quantity | dx = 0.1 | dx = 0.05 |
  |---|---|---|---|
  | A = 0.15 | \|d_t psi + i omega psi\| / \|omega psi\| | 6.7e-7 | 5.3e-8 |
  | A = 0.15 | metric RHS | 5.4e-8 | 4.2e-9 |
  | A = 0.15 | Ham / 16 pi rho | 4.4e-7 | 3.5e-8 |
  | A = 0.30 | \|d_t psi + i omega psi\| / \|omega psi\| | 3.2e-5 | 1.9e-6 |
  | A = 0.30, perturbed | Ham / 16 pi rho | 1.3e-5 | 8.9e-7 |

  These converge at roughly 4th order. For the perturbed star d_t psi stays at a resolution-independent 1.6e-3, as it
  should. `DiracCCZ4RHS` and `MatterCCZ4RHS<DiracField>` agree bit for bit.
- **Boosted star (rapidity 0.3, `box_test` mode 1):** the RHS reproduces the exact time derivative of the moving
  solution, and the Hamiltonian and momentum constraints converge to zero:

  | Quantity | dx = 0.1 | dx = 0.05 | factor |
  |---|---|---|---|
  | d_t psi error | 6.2e-7 | 3.8e-8 | 16.4 |
  | Ham / 16 pi rho | 2.6e-7 | 1.7e-8 | 16.0 |
  | Mom | 4.3e-8 | 2.7e-9 | 15.9 |

  These use a 1D solution with 80001 points. The default 20001 points (dr = 0.005) leave a floor of about 1e-8 from the
  interpolated psi' and alpha', so use `dirac_gridpoints ~ 80000` for very fine boosted runs.
- **Binary (mode 2):** for an equal-mass head-on binary all three `id_choice` options give data that are exactly
  mirror-symmetric with the parities of the grid variables.
- **Compilation:** the SingleDiracStar and BinaryDiracStar examples compile against GRChombo (main branch) and Chombo
  (3D, MPI, gcc).

## The family (mu = G = 1)

- **Stability:** the ground state is perturbatively stable for A_c < 0.2288 (maximum M = 0.70891, N = 0.73686).
- **Binding energy:** E_b = M - mu N changes sign at A_c = 0.460.
- **Further turning points:** at A_c = 1.240 and 2.363.

The data and the plot script are in 2DChombo's `Examples/DiracStar_2D/family/`.

## Running

    cd Examples/SingleDiracStar && make all && mpirun ./Main_SingleDiracStar3d*.ex params.txt

**Initial-data parameters** (`central_amplitude_dirac`, `dirac_*`) mirror Spherical_BS_Evolver's `BSParams.par`,
including the Gaussian perturbation (`dirac_perturb*`).

**Outputs:**

- `ParticleNumber` (both spinors);
- `mod_psi_max`, which equals the central amplitude for a star at rest;
- `constraint_norms` and `min_chi`, as for the boson stars.

The grid, AMR and tagging values in `params.txt` (and in `../BinaryDiracStar/params.txt`, from BBSPlainSuperposition)
are copied from the boson-star examples and untested for Dirac stars. Star tracking is not included in the binary
example.
