/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

// General includes
#include "GRParmParse.hpp"
#include "SimulationParametersBase.hpp"

// Problem specific includes:
#include "ProcaField.hpp"
#include "SingleProcaStar.hpp"

class SimulationParameters : public SimulationParametersBase
{
  public:
    SimulationParameters(GRParmParse &pp) : SimulationParametersBase(pp)
    {
        readParams(pp);
        check_params();
    }

    void readParams(GRParmParse &pp)
    {
        pp.load("G_Newton", G_Newton, 1.0);
        pp.load("proca_mass", proca_params.mass, 1.0);
        pp.load("proca_damping", proca_params.damping, 1.0); // kappa of the Gauss-constraint damping

        // Proca star initial data (parameter meanings as in Spherical_BS_Evolver's BSParams.par)
        ProcaStarSolution::params_t &s = procastar_params.solver;
        s.mu = proca_params.mass;
        s.G_Newton = G_Newton;
        pp.load("central_amplitude_proca", s.A_central);
        pp.load("proca_eigen", s.eigen, 0);
        pp.load("proca_lambda", s.lambda, 0.);
        pp.load("proca_solitonic", s.solitonic, false);
        pp.load("proca_sigma", s.sigma, 0.2);
        pp.load("proca_solver_radius", s.R, 150.);
        pp.load("proca_gridpoints", s.n_gridpoints, 30001);
        pp.load("proca_frequency_guess", s.frequency_guess, 1.5);
        pp.load("proca_alpha_central_guess", s.alpha_central, 0.9);
        pp.load("proca_r_match_fac", s.r_match_fac, 0.85);
        pp.load("proca_freq_epsilon", s.freq_epsilon, 1e-19);
        pp.load("proca_perturb", s.perturb, false);
        pp.load("proca_perturb_amp", s.perturb_amp, 0.);
        pp.load("proca_perturb_spread", s.perturb_spread, 3.);
        pp.load("proca_perturb_center", s.perturb_center, 0.);
        pp.load("star_centre", procastar_params.star_centre, center);
        pp.load("phase", procastar_params.phase, 0.0);
        pp.load("PS_rapidity", procastar_params.rapidity, 0.0); // boost along x (> 0 moves towards -x)
        pp.load("write_1d_profiles", procastar_params.write_1d_profiles, false);

        positionA = procastar_params.star_centre;
#ifdef USE_AHFINDER
        pp.load("AH_initial_guess", AH_initial_guess, 0.5);
#endif
        // tagging
        pp.load("regrid_threshold_Xphi", regrid_threshold_Xphi);
        pp.load("regrid_threshold_chi", regrid_threshold_chi);

        // mass extraction
        pp.load("activate_mass_extraction", activate_mass_extraction, 0);
        pp.load("mass_write_extraction", mass_extraction_params.write_extraction, false);
        pp.load("num_mass_extraction_radii", mass_extraction_params.num_extraction_radii, 1);
        pp.load("mass_extraction_levels", mass_extraction_params.extraction_levels,
                mass_extraction_params.num_extraction_radii, 0);
        pp.load("mass_extraction_radii", mass_extraction_params.extraction_radii,
                mass_extraction_params.num_extraction_radii, 0.1);
        pp.load("num_points_phi_mass", mass_extraction_params.num_points_phi, 2);
        pp.load("num_points_theta_mass", mass_extraction_params.num_points_theta, 4);
        pp.load("mass_extraction_center", mass_extraction_params.extraction_center, center);

        pp.load("calculate_particle_number", calculate_particle_number, true);
    }

    void check_params()
    {
        warn_parameter("proca_mass", proca_params.mass, proca_params.mass > 0., "must be positive");
        check_parameter("central_amplitude_proca", procastar_params.solver.A_central,
                        procastar_params.solver.A_central > 0., "must be positive");
        check_parameter("proca_gridpoints", procastar_params.solver.n_gridpoints,
                        procastar_params.solver.n_gridpoints > 100, "must be > 100");
        // as in Spherical_BS_Evolver, only the free field is evolved
        check_parameter("proca_lambda", procastar_params.solver.lambda,
                        procastar_params.solver.lambda == 0. && !procastar_params.solver.solitonic,
                        "the evolution (ProcaField) is for the free Proca field: set proca_lambda = 0 and "
                        "proca_solitonic = 0");
    }

    Real regrid_threshold_Xphi, regrid_threshold_chi;
    double G_Newton;
    ProcaField::params_t proca_params;
    ProcaStar_params_t procastar_params;
    int activate_mass_extraction;
    extraction_params_t mass_extraction_params;
    bool calculate_particle_number;
    std::array<double, CH_SPACEDIM> positionA;
#ifdef USE_AHFINDER
    double AH_initial_guess;
#endif
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
