/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef BINARYPROCASTAR_HPP_
#define BINARYPROCASTAR_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "MatterCCZ4RHS.hpp"
#include "MayDay.H"
#include "ProcaField.hpp"
#include "ProcaStarBoost.hpp"
#include "SingleProcaStar.hpp" // ProcaStar_params_t
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include "WeightFunction.hpp"
#include "parstream.H"
#include "simd.hpp"

//! Parameters of a Proca star binary (names and conventions as for the boson-star binaries)
struct BinaryProcaStar_params_t
{
    ProcaStar_params_t star1;   //!< solver, rapidity and phase of star 1 (its star_centre is the binary's centre)
    ProcaStar_params_t star2;   //!< solver, rapidity and phase of star 2 (star_centre unused)
    double separation;          //!< separation along x
    double impact_parameter;    //!< separation along y
    double mass_ratio;          //!< q: star 1 sits at x = q d/(1+q), star 2 at x = -d/(1+q) from the centre
    int id_choice;              //!< 0: plain superposition, 1: equal-mass fix (Helfer et al. 2022), 2: unequal-mass
                                //!< fix with weight functions (Evstafyeva et al. 2023)
    double radius_width1;       //!< weight-function widths (id_choice 2)
    double radius_width2;
    int conformal_factor_power; //!< conformal power used by the unequal-mass fix (id_choice 2)
};

/// Initial data for two Proca stars, boosted along x, superposed as the boson-star binaries are
/**
 * Each star is a (boosted) single star from ProcaStarBoost.hpp: star 1 with rapidity star1.rapidity, star 2 with
 * -star2.rapidity (so that positive rapidities make the stars approach, as in the boson-star examples). The Proca
 * variables X_phi, X_i and E^i are added (Z = 0), the lapse is sqrt(alpha_1^2 + alpha_2^2 - 1), the shifts and the mixed
 * extrinsic curvatures K^i_j are added, and the 3-metric is
 *   id_choice 0: gamma_1 + gamma_2 - delta,
 *   id_choice 1: gamma_1 + gamma_2 - gamma_1(x_2)   (equal masses; Helfer et al. 2022),
 *   id_choice 2: plain superposition with the conformal factor corrected at both centres by weight functions
 *                (Evstafyeva et al. 2023),
 * exactly as BinaryPlainSuperposition, BinaryEqualMassFix and BinaryUnequalMassFix do for boson stars.
 * The superposition violates the Gauss constraint where the stars overlap (as it does the Hamiltonian constraint); Z
 * damps this. Both stars populate the same field, so where they overlap they interfere and the relative phase
 * (star1.phase - star2.phase) matters.
 **/
class BinaryProcaStar
{
  public:
    BinaryProcaStar(BinaryProcaStar_params_t a_params, double a_dx, int a_verbosity = 1)
        : m_params(a_params), m_dx(a_dx), m_verbosity(a_verbosity)
    {
    }

    void compute_1d_solution()
    {
        if (!m_1d_sol.solve(m_params.star1.solver, pout(), m_verbosity == 0) ||
            !m_1d_sol2.solve(m_params.star2.solver, pout(), m_verbosity == 0))
            MayDay::Error("BinaryProcaStar: the 1D Proca star solver failed (see pout)");
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        using namespace TensorAlgebra;
        typename MatterCCZ4RHS<ProcaField>::template Vars<data_t> vars;
        VarsTools::assign(vars, 0.);
        Coordinates<data_t> coords(current_cell, m_dx, m_params.star1.star_centre);

        const double q = m_params.mass_ratio;
        const double d = m_params.separation;
        const double b = m_params.impact_parameter;
        const double xi1 = m_params.star1.rapidity;
        const double xi2 = -m_params.star2.rapidity;
        const BoostedProcaStarVars s1 = boosted_proca_star(m_1d_sol, coords.x - q * d / (q + 1.),
                                                           coords.y + q * b / (q + 1.), coords.z, xi1,
                                                           m_params.star1.phase);
        const BoostedProcaStarVars s2 = boosted_proca_star(m_1d_sol2, coords.x + d / (q + 1.), coords.y - b / (q + 1.),
                                                           coords.z, xi2, m_params.star2.phase);

        FOR(i)
        {
            vars.X_Re[i] = s1.X[i].real() + s2.X[i].real();
            vars.X_Im[i] = s1.X[i].imag() + s2.X[i].imag();
            vars.E_Re[i] = s1.E[i].real() + s2.E[i].real();
            vars.E_Im[i] = s1.E[i].imag() + s2.E[i].imag();
        }
        vars.Xphi_Re = s1.Xphi.real() + s2.Xphi.real();
        vars.Xphi_Im = s1.Xphi.imag() + s2.Xphi.imag();
        vars.shift[0] = s1.shiftx + s2.shiftx;
        vars.lapse = sqrt(s1.lapse * s1.lapse + s2.lapse * s2.lapse - 1.);

        Tensor<2, data_t> g1, g2, K1, K2, gammaLL, KLL;
        FOR(i, j)
        {
            g1[i][j] = s1.gLL[i][j];
            g2[i][j] = s2.gLL[i][j];
            K1[i][j] = s1.KLL[i][j];
            K2[i][j] = s2.KLL[i][j];
            gammaLL[i][j] = g1[i][j] + g2[i][j] - ((i == j) ? 1. : 0.); // plain superposition
            KLL[i][j] = 0.;
        }
        const auto g1UU = compute_inverse_sym(g1);
        const auto g2UU = compute_inverse_sym(g2);

        if (m_params.id_choice == 1)
        {
            // subtract star 1's metric at the position of star 2 instead of delta (as BinaryEqualMassFix)
            const BoostedProcaStarVars h = boosted_proca_star(m_1d_sol, -d, b, 0., xi1, 0.);
            FOR(i, j) gammaLL[i][j] = g1[i][j] + g2[i][j] - h.gLL[i][j];
        }

        FOR(i, j, k, l) { KLL[i][j] += gammaLL[i][l] * (g1UU[l][k] * K1[k][j] + g2UU[l][k] * K2[k][j]); }
        const auto gammaUU = compute_inverse_sym(gammaLL);

        data_t chi_h; // conformal factor used for h_ij and A_ij
        if (m_params.id_choice == 2)
        {
            // as BinaryUnequalMassFix
            const BoostedProcaStarVars c1 = boosted_proca_star(m_1d_sol, -d, b, 0., xi1, 0.);   // star 1 at x_2
            const BoostedProcaStarVars c2 = boosted_proca_star(m_1d_sol2, d, -b, 0., xi2, 0.);  // star 2 at x_1
            const BoostedProcaStarVars c11 = boosted_proca_star(m_1d_sol, 0., 0., 0., xi1, 0.); // star 1 at x_1
            const BoostedProcaStarVars c22 = boosted_proca_star(m_1d_sol2, 0., 0., 0., xi2, 0.);
            const double n_power = m_params.conformal_factor_power / 12.0;
            auto det_diag = [](const double g[3][3], const double h[3][3]) {
                return (g[0][0] + h[0][0] - 1.) * (g[1][1] + h[1][1] - 1.) * (g[2][2] + h[2][2] - 1.);
            };
            const data_t chi_plain = pow(gammaLL[0][0] * gammaLL[1][1] * gammaLL[2][2], n_power);
            const double chi_1 = pow(det_diag(c11.gLL, c2.gLL), n_power);
            const double chi_2 = pow(det_diag(c22.gLL, c1.gLL), n_power);
            const double chi1_1 = pow(c11.gLL[0][0] * c11.gLL[1][1] * c11.gLL[2][2], n_power);
            const double chi2_2 = pow(c22.gLL[0][0] * c22.gLL[1][1] * c22.gLL[2][2], n_power);
            const double delta_1 = chi1_1 - chi_1;
            const double delta_2 = chi2_2 - chi_2;

            WeightFunction weight;
            const double w1 = m_params.radius_width1, w2 = m_params.radius_width2;
            const data_t profile1 =
                weight.profile_chi((coords.x - q * d / (q + 1)) * cosh(xi1), coords.y + q * b / (q + 1.), coords.z, w1);
            const data_t profile2 =
                weight.profile_chi((coords.x + d / (q + 1)) * cosh(xi2), coords.y - b / (q + 1.), coords.z, w2);
            const double profile_11 = weight.profile_chi(0., 0., 0., w1);
            const double profile_12 = weight.profile_chi((d / (q + 1)) * (cosh(xi2) + q * cosh(xi1)), -b, 0., w1);
            const double profile_21 = weight.profile_chi((d / (q + 1)) * (-q * cosh(xi1) - cosh(xi2)), b, 0., w2);
            const double profile_22 = weight.profile_chi(0., 0., 0., w2);
            const double det = profile_11 * profile_22 - profile_12 * profile_21;
            const double value1 = (-profile_21 * delta_2 + profile_22 * delta_1) / det;
            const double value2 = (profile_11 * delta_2 - profile_12 * delta_1) / det;
            const data_t chi_corrected = chi_plain + profile1 * value1 + profile2 * value2;
            vars.chi = pow(chi_corrected, -4.0 / m_params.conformal_factor_power);
            chi_h = pow(chi_plain, -4.0 / m_params.conformal_factor_power);
        }
        else
        {
            vars.chi = pow(gammaLL[0][0] * gammaLL[1][1] * gammaLL[2][2], -1. / 3.);
            chi_h = vars.chi;
        }

        FOR(i, j) vars.h[i][j] = chi_h * gammaLL[i][j];
        vars.K = 0.;
        FOR(i, j) vars.K += KLL[i][j] * gammaUU[i][j];
        FOR(i, j) vars.A[i][j] = chi_h * (KLL[i][j] - vars.K * gammaLL[i][j] / 3.);

        current_cell.store_vars(vars);
    }

    ProcaStarSolution m_1d_sol;
    ProcaStarSolution m_1d_sol2;

  protected:
    BinaryProcaStar_params_t m_params;
    double m_dx;
    int m_verbosity;
};

#endif /* BINARYPROCASTAR_HPP_ */
