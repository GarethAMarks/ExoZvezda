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
| `Source/InitialConditions/DiracStars/SingleDiracStar.hpp` | Initial data for a single star at rest |
| `Examples/SingleDiracStar/` | Level, `params.txt` (quadrant symmetry, as SingleBosonStar), variables |
| `derivations/` | `check_dirac3d.py`: numerical checks of the equations; `dirac3d_ref.py`: Python reference implementation |
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
- **Compilation:** the example compiles against GRChombo (main branch) and Chombo (3D, MPI, gcc).

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

The grid, AMR and tagging values in `params.txt` are copied from SingleBosonStar and untested for Dirac stars. Boosted
stars and binaries are not implemented yet: a boost needs the spinor Lorentz transformation, and the variables and
parities already allow x-reflection.
