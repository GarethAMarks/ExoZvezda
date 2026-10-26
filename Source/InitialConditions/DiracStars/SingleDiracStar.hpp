/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SINGLEDIRACSTAR_HPP_
#define SINGLEDIRACSTAR_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DiracField.hpp"
#include "DiracStarSolution.hpp"
#include "MatterCCZ4RHS.hpp"
#include "MayDay.H"
#include "Tensor.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include "parstream.H"
#include "simd.hpp"
#include <array>
#include <cmath>

//! Parameters of a single Dirac star
struct DiracStar_params_t
{
    DiracStarSolution::params_t solver;          //!< 1D solver parameters (mu, A_central, eigen, perturbation, ...)
    std::array<double, CH_SPACEDIM> star_centre; //!< centre of the star
    double phase;                                //!< overall phase of both spinors, in units of pi
    bool write_1d_profiles;                      //!< write the 1D solution to DiracStar_1d_profiles.dat (rank 0)
};

/// Initial data for a single Dirac star at rest
/**
 * Solves the spherical star with DiracStarSolution (a port of the Spherical_BS_Evolver solver, optionally perturbed,
 * shared with 2DChombo) and sets, in isotropic coordinates centred on star_centre,
 *   chi = psi^{-4}, h_ij = delta_ij, lapse = alpha, K = A_ij = Theta = shift = B = 0,
 *   D_e = g e^{i theta} / sqrt(2 pi), D_x = -i f n_x e^{i theta} / sqrt(2 pi), D_y = f n_y e^{i theta} / sqrt(2 pi),
 *   D_z = -i f n_z e^{i theta} / sqrt(2 pi), the other D = 0,
 * i.e. the two spinors psi_m = (g chi_m, -i f (sigma.n) chi_m) / sqrt(4 pi), m = +-1/2, theta = phase pi (see
 * DiracField.hpp). Gamma^i must be computed afterwards (GammaCalculator). Use with disable_simd().
 **/
class SingleDiracStar
{
  public:
    SingleDiracStar(DiracStar_params_t a_params, double a_dx, int a_verbosity = 1)
        : m_params(a_params), m_dx(a_dx), m_verbosity(a_verbosity)
    {
    }

    //! Computes the 1d solution (every rank solves it; it takes well under a second)
    void compute_1d_solution()
    {
        if (!m_1d_sol.solve(m_params.solver, pout(), m_verbosity == 0))
            MayDay::Error("SingleDiracStar: the 1D Dirac star solver failed (see pout); try a different "
                          "dirac_frequency_guess / dirac_alpha_central_guess or a larger dirac_solver_radius");
#ifdef CH_MPI
        int rank;
        MPI_Comm_rank(Chombo_MPI::comm, &rank);
#else
        int rank = 0;
#endif
        if (m_params.write_1d_profiles && rank == 0)
            m_1d_sol.write_profiles("DiracStar_1d_profiles.dat");
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        typename MatterCCZ4RHS<DiracField>::template Vars<data_t> vars;
        VarsTools::assign(vars, 0.);

        Coordinates<data_t> coords(current_cell, m_dx, m_params.star_centre);
        const data_t x = coords.x;
        const double y = coords.y, z = coords.z;
        const data_t r = sqrt(x * x + y * y + z * z);

        const data_t g = m_1d_sol.get_g(r);
        const data_t f = m_1d_sol.get_f(r);
        const data_t psi_conf = m_1d_sol.get_psi(r);

        vars.chi = pow(psi_conf, -4.);
        FOR(i) vars.h[i][i] = 1.;
        vars.lapse = m_1d_sol.get_lapse(r);

        // unit radial vector (f vanishes at the centre, so n is irrelevant there)
        const data_t rinv = (r > 1e-12 * m_dx) ? 1. / r : 0.;
        const data_t n[3] = {x * rinv, y * rinv, z * rinv};
        const double norm = 1. / sqrt(2. * M_PI);
        const double th = m_params.phase * M_PI;
        const double c = cos(th), s = sin(th);
        // D_e = g e^{i th};  D_{x,z} = -i f n e^{i th} = f n (sin th - i cos th);  D_y = f n_y e^{i th}
        vars.D_Re[0] = norm * g * c;
        vars.D_Im[0] = norm * g * s;
        vars.D_Re[1] = norm * f * n[0] * s;
        vars.D_Im[1] = -norm * f * n[0] * c;
        vars.D_Re[2] = norm * f * n[1] * c;
        vars.D_Im[2] = norm * f * n[1] * s;
        vars.D_Re[3] = norm * f * n[2] * s;
        vars.D_Im[3] = -norm * f * n[2] * c;

        current_cell.store_vars(vars);
    }

    DiracStarSolution m_1d_sol;

  protected:
    DiracStar_params_t m_params;
    double m_dx;
    int m_verbosity;
};

#endif /* SINGLEDIRACSTAR_HPP_ */
