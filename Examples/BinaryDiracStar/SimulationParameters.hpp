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
#include "DiracField.hpp"
#include "BinaryDiracStar.hpp"

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
        pp.load("dirac_mass", dirac_params.mass, 1.0);

        // Dirac star initial data (parameter meanings as in Spherical_BS_Evolver's BSParams.par; binary parameters
        // as for the boson-star binaries)
        DiracStar_params_t &p1 = binary_params.star1;
        DiracStarSolution::params_t &s = p1.solver;
        s.mu = dirac_params.mass;
        s.G_Newton = G_Newton;
        pp.load("central_amplitude_dirac", s.A_central);
        pp.load("dirac_eigen", s.eigen, 0);
        pp.load("dirac_solver_radius", s.R, 100.);
        pp.load("dirac_gridpoints", s.n_gridpoints, 20001);
        pp.load("dirac_frequency_guess", s.frequency_guess, 1.8);
        pp.load("dirac_alpha_central_guess", s.alpha_central, 0.95);
        pp.load("dirac_r_match_fac", s.r_match_fac, 0.85);
        pp.load("dirac_freq_epsilon", s.freq_epsilon, 1e-19);
        pp.load("dirac_perturb", s.perturb, false);
        pp.load("dirac_perturb_aspherical", perturb_aspherical, false); // single stars only (checked below)
        pp.load("dirac_perturb_amp", s.perturb_amp, 0.);
        pp.load("dirac_perturb_spread", s.perturb_spread, 3.);
        pp.load("dirac_perturb_center", s.perturb_center, 0.);
        pp.load("star_centre", p1.star_centre, center);
        pp.load("phase", p1.phase, 0.0);
        pp.load("DS_rapidity", p1.rapidity, 0.0);
        p1.write_1d_profiles = false;

        // star 2: identical to star 1 unless specified
        binary_params.star2 = p1;
        binary_params.star2.phase = 0.;
        pp.load("central_amplitude_dirac2", binary_params.star2.solver.A_central, s.A_central);
        pp.load("DS_rapidity2", binary_params.star2.rapidity, p1.rapidity);
        pp.load("phase2", binary_params.star2.phase, 0.0);

        pp.load("DS_separation", binary_params.separation, 0.0);
        pp.load("DS_impact_parameter", binary_params.impact_parameter, 0.0);
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
        pp.load("regrid_threshold_psi", regrid_threshold_psi);
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
        check_parameter("dirac_perturb_aspherical", perturb_aspherical, !perturb_aspherical,
                        "aspherical perturbations are only implemented for single stars; the binary takes the "
                        "spherical (m = 0) perturbation only");
        warn_parameter("dirac_mass", dirac_params.mass, dirac_params.mass > 0., "must be positive");
        check_parameter("central_amplitude_dirac", binary_params.star1.solver.A_central,
                        binary_params.star1.solver.A_central > 0., "must be positive");
        check_parameter("central_amplitude_dirac2", binary_params.star2.solver.A_central,
                        binary_params.star2.solver.A_central > 0., "must be positive");
        check_parameter("dirac_gridpoints", binary_params.star1.solver.n_gridpoints,
                        binary_params.star1.solver.n_gridpoints > 100, "must be > 100");
        check_parameter("id_choice", binary_params.id_choice,
                        binary_params.id_choice >= 0 && binary_params.id_choice <= 2, "must be 0, 1 or 2");
        warn_parameter("mass_ratio", binary_params.mass_ratio,
                       binary_params.id_choice != 1 || binary_params.mass_ratio == 1.,
                       "the equal-mass fix (id_choice 1) assumes equal stars");
    }

    Real regrid_threshold_psi, regrid_threshold_chi;
    double G_Newton;
    DiracField::params_t dirac_params;
    BinaryDiracStar_params_t binary_params;
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
