/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SINGLEPROCASTAR_HPP_
#define SINGLEPROCASTAR_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "MatterCCZ4RHS.hpp"
#include "MayDay.H"
#include "ProcaField.hpp"
#include "ProcaStarBoost.hpp"
#include "ProcaStarSolution.hpp"
#include "Tensor.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include "parstream.H"
#include "simd.hpp"
#include <array>
#include <cmath>

//! Parameters of a single Proca star
struct ProcaStar_params_t
{
    ProcaStarSolution::params_t solver;          //!< 1D solver parameters (mu, A_central, eigen, perturbation, ...)
    std::array<double, CH_SPACEDIM> star_centre; //!< centre of the star
    double phase;                                //!< overall phase of the field, in units of pi
    double rapidity = 0.;                        //!< boost along x (a star with rapidity > 0 moves towards -x)
    bool write_1d_profiles;                      //!< write the 1D solution to ProcaStar_1d_profiles.dat (rank 0)
};

//! Copies the metric quantities of a boosted star into the CCZ4 variables (chi, h, K, A, lapse, shift)
template <class vars_t> void set_ccz4_from_boosted_star(vars_t &vars, const double gLL_in[3][3], const double KLL_in[3][3])
{
    using namespace TensorAlgebra;
    Tensor<2, double> gLL, KLL;
    FOR(i, j)
    {
        gLL[i][j] = gLL_in[i][j];
        KLL[i][j] = KLL_in[i][j];
    }
    const auto gUU = compute_inverse_sym(gLL);
    vars.chi = pow(gLL[0][0] * gLL[1][1] * gLL[2][2], -1. / 3.);
    FOR(i, j) vars.h[i][j] = vars.chi * gLL[i][j];
    vars.K = 0.;
    FOR(i, j) vars.K += KLL[i][j] * gUU[i][j];
    FOR(i, j) vars.A[i][j] = vars.chi * (KLL[i][j] - vars.K * gLL[i][j] / 3.);
}

/// Initial data for a single Proca star, at rest or boosted along x
/**
 * Solves the spherical star with ProcaStarSolution (a port of the Spherical_BS_Evolver solver, optionally perturbed,
 * shared with 2DChombo) and sets, for a star at rest in isotropic coordinates centred on star_centre,
 *   chi = psi^{-4}, h_ij = delta_ij, lapse = alpha, K = A_ij = Theta = shift = B = 0,
 *   X_phi = -f e^{i theta} / alpha,  X_i = i g n_i e^{i theta},  E^i = E n_i e^{i theta} / (alpha psi^4),  Z = 0,
 * with f, g, E the isotropic profiles of the 1D solver and theta = phase pi (|X_phi(0)| = A_central). Gamma^i must be
 * computed afterwards (GammaCalculator). Use with disable_simd().
 * With rapidity != 0 the star is boosted along x as the boson stars are: lab metric, lapse, shift and K_ij, and the
 * vector potential and field strength transformed as a covector and a 2-form (ProcaStarBoost.hpp).
 **/
class SingleProcaStar
{
  public:
    SingleProcaStar(ProcaStar_params_t a_params, double a_dx, int a_verbosity = 1)
        : m_params(a_params), m_dx(a_dx), m_verbosity(a_verbosity)
    {
    }

    //! Computes the 1d solution (every rank solves it; it takes about a second)
    void compute_1d_solution()
    {
        if (!m_1d_sol.solve(m_params.solver, pout(), m_verbosity == 0))
            MayDay::Error("SingleProcaStar: the 1D Proca star solver failed (see pout); try a different "
                          "proca_frequency_guess / proca_alpha_central_guess or a larger proca_solver_radius");
#ifdef CH_MPI
        int rank;
        MPI_Comm_rank(Chombo_MPI::comm, &rank);
#else
        int rank = 0;
#endif
        if (m_params.write_1d_profiles && rank == 0)
            m_1d_sol.write_profiles("ProcaStar_1d_profiles.dat");
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        typename MatterCCZ4RHS<ProcaField>::template Vars<data_t> vars;
        VarsTools::assign(vars, 0.);

        Coordinates<data_t> coords(current_cell, m_dx, m_params.star_centre);
        const BoostedProcaStarVars s =
            boosted_proca_star(m_1d_sol, coords.x, coords.y, coords.z, m_params.rapidity, m_params.phase);

        set_ccz4_from_boosted_star(vars, s.gLL, s.KLL);
        vars.lapse = s.lapse;
        vars.shift[0] = s.shiftx;
        FOR(i)
        {
            vars.X_Re[i] = s.X[i].real();
            vars.X_Im[i] = s.X[i].imag();
            vars.E_Re[i] = s.E[i].real();
            vars.E_Im[i] = s.E[i].imag();
        }
        vars.Xphi_Re = s.Xphi.real();
        vars.Xphi_Im = s.Xphi.imag();
        current_cell.store_vars(vars);
    }

    ProcaStarSolution m_1d_sol;

  protected:
    ProcaStar_params_t m_params;
    double m_dx;
    int m_verbosity;
};

#endif /* SINGLEPROCASTAR_HPP_ */
