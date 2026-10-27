/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SINGLEDIRACSTAR_HPP_
#define SINGLEDIRACSTAR_HPP_

#include "AsphericalPerturbation.hpp"
#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DiracField.hpp"
#include "DiracStarHelperFunction.hpp"
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
    double rapidity = 0.;                        //!< boost along x (a star with rapidity > 0 moves towards -x)
    bool write_1d_profiles;                      //!< write the 1D solution to DiracStar_1d_profiles.dat (rank 0)

    //! aspherical perturbation (arXiv:2609.30021, Eq. (11), with an optional second one in an orthogonal plane; see
    //! AsphericalPerturbation.hpp). The spherical one (m = 0) is in solver (perturb, perturb_amp, ...). Single stars only.
    AsphericalPerturbation aspherical;
};

/// Initial data for a single Dirac star, at rest or boosted along x
/**
 * Solves the spherical star with DiracStarSolution (a port of the Spherical_BS_Evolver solver, optionally perturbed,
 * shared with 2DChombo) and sets, in isotropic coordinates centred on star_centre,
 *   chi = psi^{-4}, h_ij = delta_ij, lapse = alpha, K = A_ij = Theta = shift = B = 0,
 *   D_e = g e^{i theta} / sqrt(2 pi), D_x = -i f n_x e^{i theta} / sqrt(2 pi), D_y = f n_y e^{i theta} / sqrt(2 pi),
 *   D_z = -i f n_z e^{i theta} / sqrt(2 pi), the other D = 0,
 * i.e. the two spinors psi_m = (g chi_m, -i f (sigma.n) chi_m) / sqrt(4 pi), m = +-1/2, theta = phase pi (see
 * DiracField.hpp). Gamma^i must be computed afterwards (GammaCalculator). Use with disable_simd().
 * With rapidity != 0 the star is boosted along x as the boson stars are (lab metric, lapse, shift and K_ij from
 * DiracStarBoost.hpp) and both spinors are transformed by the local Lorentz boost U(eta) between the star's rest
 * frame and the lab tetrad (DiracStarHelperFunction.hpp); for rapidity = 0 this reduces to the formulas above.
 * Perturbations: the spherical one (m = 0) is applied by the 1D solver, with the constraints re-solved; the aspherical
 * one (aspherical.active) multiplies the matter fields by exp(-2 delta cos(m phi)) [exp(-2 epsilon cos(n phi'))], phi
 * (phi') the azimuthal angle about the star centre in the plane of the reference axes (arXiv:2609.30021, Eq. (11);
 * AsphericalPerturbation.hpp).
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
        using namespace TensorAlgebra;
        typename MatterCCZ4RHS<DiracField>::template Vars<data_t> vars;
        VarsTools::assign(vars, 0.);

        Coordinates<data_t> coords(current_cell, m_dx, m_params.star_centre);
        DiracStarHelperFunction helper;
        const auto star = helper.compute_star_vars(coords.x, coords.y, coords.z, m_params.rapidity, 0., 0.,
                                                   m_1d_sol, m_params.phase);

        Tensor<2, data_t> gLL, KLL;
        FOR(i, j)
        {
            gLL[i][j] = star.gLL[i][j];
            KLL[i][j] = star.KLL[i][j];
        }
        const auto gUU = compute_inverse_sym(gLL);
        vars.chi = pow(gLL[0][0] * gLL[1][1] * gLL[2][2], -1. / 3.);
        FOR(i, j) vars.h[i][j] = vars.chi * gLL[i][j];
        vars.K = 0.;
        FOR(i, j) vars.K += KLL[i][j] * gUU[i][j];
        FOR(i, j) vars.A[i][j] = vars.chi * (KLL[i][j] - vars.K * gLL[i][j] / 3.);
        vars.lapse = star.lapse;
        vars.shift[0] = star.shiftx;
        // aspherical perturbation about the star centre (F = 1 unless active)
        const double F = m_params.aspherical.factor(coords.x, coords.y, coords.z);
        for (int k = 0; k < 8; ++k)
        {
            vars.D_Re[k] = F * star.D_Re[k];
            vars.D_Im[k] = F * star.D_Im[k];
        }
        current_cell.store_vars(vars);
    }

    DiracStarSolution m_1d_sol;

  protected:
    DiracStar_params_t m_params;
    double m_dx;
    int m_verbosity;
};

#endif /* SINGLEDIRACSTAR_HPP_ */
