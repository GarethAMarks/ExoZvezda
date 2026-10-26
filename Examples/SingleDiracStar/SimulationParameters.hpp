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
#include "SingleDiracStar.hpp"

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

        // Dirac star initial data (parameter meanings as in Spherical_BS_Evolver's BSParams.par)
        DiracStarSolution::params_t &s = diracstar_params.solver;
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
        pp.load("dirac_perturb_amp", s.perturb_amp, 0.);
        pp.load("dirac_perturb_spread", s.perturb_spread, 3.);
        pp.load("dirac_perturb_center", s.perturb_center, 0.);
        pp.load("star_centre", diracstar_params.star_centre, center);
        pp.load("phase", diracstar_params.phase, 0.0);
        pp.load("write_1d_profiles", diracstar_params.write_1d_profiles, false);

        positionA = diracstar_params.star_centre;
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
        warn_parameter("dirac_mass", dirac_params.mass, dirac_params.mass > 0., "must be positive");
        check_parameter("central_amplitude_dirac", diracstar_params.solver.A_central,
                        diracstar_params.solver.A_central > 0., "must be positive");
        check_parameter("dirac_gridpoints", diracstar_params.solver.n_gridpoints,
                        diracstar_params.solver.n_gridpoints > 100, "must be > 100");
    }

    Real regrid_threshold_psi, regrid_threshold_chi;
    double G_Newton;
    DiracField::params_t dirac_params;
    DiracStar_params_t diracstar_params;
    int activate_mass_extraction;
    extraction_params_t mass_extraction_params;
    bool calculate_particle_number;
    std::array<double, CH_SPACEDIM> positionA;
#ifdef USE_AHFINDER
    double AH_initial_guess;
#endif
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
