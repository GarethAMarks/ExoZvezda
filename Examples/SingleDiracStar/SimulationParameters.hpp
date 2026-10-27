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
        // Perturbations. With dirac_perturb = 1 the azimuthal index dirac_perturb_m selects the perturbation:
        //   m = 0: the spherical perturbation (f, g) -> (1 + a exp(-(r - r0)^2 / w^2)) (f, g) of the 1D solver, with the
        //          constraints re-solved (a = dirac_perturb_amp, w = dirac_perturb_spread, r0 = dirac_perturb_center);
        //   m >= 1: the aspherical perturbation of arXiv:2609.30021, Eq. (11): all matter fields multiplied by
        //          exp(-2 delta cos(m phi)), delta = dirac_perturb_amp, phi = atan2(x_b, x_a) about the star centre
        //          with (a, b) = dirac_perturb_axes.
        // Optionally a second perturbation (the paper's two-axis runs), with index n = dirac_perturb_n (0: none) and
        // magnitude epsilon = dirac_perturb_amp2, in an orthogonal plane dirac_perturb_axes2.
        // dirac_perturb_aspherical = 0 (default) disables the aspherical perturbations: the perturbation is then the
        // spherical one whatever m is.
        bool perturb, perturb_aspherical;
        int perturb_m;
        double perturb_amp;
        pp.load("dirac_perturb", perturb, false);
        pp.load("dirac_perturb_aspherical", perturb_aspherical, false);
        pp.load("dirac_perturb_m", perturb_m, 0);
        pp.load("dirac_perturb_amp", perturb_amp, 0.);
        pp.load("dirac_perturb_spread", s.perturb_spread, 3.);
        pp.load("dirac_perturb_center", s.perturb_center, 0.);
        AsphericalPerturbation &asph = diracstar_params.aspherical;
        asph.active = perturb && perturb_aspherical;
        asph.m = perturb_m;
        asph.delta = perturb_amp;
        std::array<int, 2> axes, axes2;
        pp.load("dirac_perturb_axes", axes, {0, 1});
        pp.load("dirac_perturb_n", asph.n, 0);
        pp.load("dirac_perturb_amp2", asph.epsilon, 0.);
        pp.load("dirac_perturb_axes2", axes2, {0, 2});
        for (int i = 0; i < 2; ++i)
        {
            asph.axes[i] = axes[i];
            asph.axes2[i] = axes2[i];
        }
        s.perturb = perturb && (!perturb_aspherical || perturb_m == 0);
        s.perturb_amp = perturb_amp;
        pp.load("star_centre", diracstar_params.star_centre, center);
        pp.load("phase", diracstar_params.phase, 0.0);
        pp.load("DS_rapidity", diracstar_params.rapidity, 0.0); // boost along x (> 0 moves towards -x)
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
        const AsphericalPerturbation &asph = diracstar_params.aspherical;
        const std::string asph_error = asph.check(CH_SPACEDIM);
        check_parameter("dirac_perturb_aspherical", asph.active, asph_error.empty(), asph_error);
        // the perturbation must have the symmetries of the reflective boundaries (an odd index is odd under the
        // reflection of the first reference axis)
        for (int dir = 0; dir < CH_SPACEDIM; ++dir)
        {
            const bool reflective = boundary_params.lo_boundary[dir] == BoundaryConditions::REFLECTIVE_BC ||
                                    boundary_params.hi_boundary[dir] == BoundaryConditions::REFLECTIVE_BC;
            check_parameter("dirac_perturb_axes", dir, !reflective || asph.even_under_reflection(dir),
                            "an odd-index aspherical perturbation whose first reference axis is this direction breaks "
                            "the reflection symmetry of the boundary: use another reference axis or an even index");
        }
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
