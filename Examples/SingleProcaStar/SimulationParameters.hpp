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
        // Perturbations. With proca_perturb = 1 the azimuthal index proca_perturb_m selects the perturbation:
        //   m = 0: the spherical perturbation (f, g) -> (1 + a exp(-(r - r0)^2 / w^2)) (f, g) of the 1D solver, with the
        //          constraints re-solved (a = proca_perturb_amp, w = proca_perturb_spread, r0 = proca_perturb_center);
        //   m >= 1: the aspherical perturbation of arXiv:2609.30021, Eq. (11): all matter fields multiplied by
        //          exp(-2 delta cos(m phi)), delta = proca_perturb_amp, phi = atan2(x_b, x_a) about the star centre
        //          with (a, b) = proca_perturb_axes.
        // Optionally a second perturbation (the paper's two-axis runs), with index n = proca_perturb_n (0: none) and
        // magnitude epsilon = proca_perturb_amp2, in an orthogonal plane proca_perturb_axes2.
        // proca_perturb_smooth = 1 replaces each cos(m phi) by cos(m phi) sin^m(theta) = Re[(x_a + i x_b)^m] / r^m, theta
        // measured from the line normal to the plane, which removes the jump of the factor across that line (default 0:
        // the paper's convention).
        // proca_perturb_aspherical = 0 (default) disables the aspherical perturbations: the perturbation is then the
        // spherical one whatever m is.
        bool perturb, perturb_aspherical;
        int perturb_m;
        double perturb_amp;
        pp.load("proca_perturb", perturb, false);
        pp.load("proca_perturb_aspherical", perturb_aspherical, false);
        pp.load("proca_perturb_m", perturb_m, 0);
        pp.load("proca_perturb_amp", perturb_amp, 0.);
        pp.load("proca_perturb_spread", s.perturb_spread, 3.);
        pp.load("proca_perturb_center", s.perturb_center, 0.);
        AsphericalPerturbation &asph = procastar_params.aspherical;
        asph.active = perturb && perturb_aspherical;
        asph.m = perturb_m;
        asph.delta = perturb_amp;
        std::array<int, 2> axes, axes2;
        pp.load("proca_perturb_axes", axes, {0, 1});
        pp.load("proca_perturb_n", asph.n, 0);
        pp.load("proca_perturb_amp2", asph.epsilon, 0.);
        pp.load("proca_perturb_axes2", axes2, {0, 2});
        pp.load("proca_perturb_smooth", asph.smooth, false);
        for (int i = 0; i < 2; ++i)
        {
            asph.axes[i] = axes[i];
            asph.axes2[i] = axes2[i];
        }
        s.perturb = perturb && (!perturb_aspherical || perturb_m == 0);
        s.perturb_amp = perturb_amp;
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
        const AsphericalPerturbation &asph = procastar_params.aspherical;
        const std::string asph_error = asph.check(CH_SPACEDIM);
        check_parameter("proca_perturb_aspherical", asph.active, asph_error.empty(), asph_error);
        // the perturbation must have the symmetries of the reflective boundaries (an odd index is odd under the
        // reflection of the first reference axis)
        for (int dir = 0; dir < CH_SPACEDIM; ++dir)
        {
            const bool reflective = boundary_params.lo_boundary[dir] == BoundaryConditions::REFLECTIVE_BC ||
                                    boundary_params.hi_boundary[dir] == BoundaryConditions::REFLECTIVE_BC;
            check_parameter("proca_perturb_axes", dir, !reflective || asph.even_under_reflection(dir),
                            "an odd-index aspherical perturbation whose first reference axis is this direction breaks "
                            "the reflection symmetry of the boundary: use another reference axis or an even index");
        }
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
