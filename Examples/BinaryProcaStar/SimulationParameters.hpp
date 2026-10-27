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
#include "BinaryProcaStar.hpp"

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

        // Proca star initial data (parameter meanings as in Spherical_BS_Evolver's BSParams.par; binary parameters
        // as for the boson-star binaries)
        ProcaStar_params_t &p1 = binary_params.star1;
        ProcaStarSolution::params_t &s = p1.solver;
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
        pp.load("proca_perturb_aspherical", perturb_aspherical, false); // single stars only (checked below)
        pp.load("proca_perturb_amp", s.perturb_amp, 0.);
        pp.load("proca_perturb_spread", s.perturb_spread, 3.);
        pp.load("proca_perturb_center", s.perturb_center, 0.);
        pp.load("star_centre", p1.star_centre, center);
        pp.load("phase", p1.phase, 0.0);
        pp.load("PS_rapidity", p1.rapidity, 0.0);
        p1.write_1d_profiles = false;

        // star 2: identical to star 1 unless specified
        binary_params.star2 = p1;
        binary_params.star2.phase = 0.;
        pp.load("central_amplitude_proca2", binary_params.star2.solver.A_central, s.A_central);
        pp.load("PS_rapidity2", binary_params.star2.rapidity, p1.rapidity);
        pp.load("phase2", binary_params.star2.phase, 0.0);

        pp.load("PS_separation", binary_params.separation, 0.0);
        pp.load("PS_impact_parameter", binary_params.impact_parameter, 0.0);
        pp.load("mass_ratio", binary_params.mass_ratio, 1.0);
        pp.load("id_choice", binary_params.id_choice, 2);
        pp.load("radius_width1", binary_params.radius_width1, 10.);
        pp.load("radius_width2", binary_params.radius_width2, 20.);
        pp.load("conformal_factor_power", binary_params.conformal_factor_power, -4);

        positionA = binary_params.star1.star_centre;
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
        check_parameter("proca_perturb_aspherical", perturb_aspherical, !perturb_aspherical,
                        "aspherical perturbations are only implemented for single stars; the binary takes the "
                        "spherical (m = 0) perturbation only");
        warn_parameter("proca_mass", proca_params.mass, proca_params.mass > 0., "must be positive");
        check_parameter("central_amplitude_proca", binary_params.star1.solver.A_central,
                        binary_params.star1.solver.A_central > 0., "must be positive");
        check_parameter("central_amplitude_proca2", binary_params.star2.solver.A_central,
                        binary_params.star2.solver.A_central > 0., "must be positive");
        check_parameter("proca_gridpoints", binary_params.star1.solver.n_gridpoints,
                        binary_params.star1.solver.n_gridpoints > 100, "must be > 100");
        check_parameter("proca_lambda", binary_params.star1.solver.lambda,
                        binary_params.star1.solver.lambda == 0. && !binary_params.star1.solver.solitonic,
                        "the evolution (ProcaField) is for the free Proca field: set proca_lambda = 0 and "
                        "proca_solitonic = 0");
        check_parameter("id_choice", binary_params.id_choice,
                        binary_params.id_choice >= 0 && binary_params.id_choice <= 2, "must be 0, 1 or 2");
        warn_parameter("mass_ratio", binary_params.mass_ratio,
                       binary_params.id_choice != 1 || binary_params.mass_ratio == 1.,
                       "the equal-mass fix (id_choice 1) assumes equal stars");
    }

    Real regrid_threshold_Xphi, regrid_threshold_chi;
    double G_Newton;
    ProcaField::params_t proca_params;
    BinaryProcaStar_params_t binary_params;
    int activate_mass_extraction;
    extraction_params_t mass_extraction_params;
    bool calculate_particle_number;
    bool perturb_aspherical;
    std::array<double, CH_SPACEDIM> positionA;
#ifdef USE_AHFINDER
    double AH_initial_guess;
#endif
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
