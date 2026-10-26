// Single-box unit test of the assembled 3D Dirac code (no time evolution; a few seconds on one core):
// initial data for a star -> one evaluation of DiracCCZ4RHS (and of MatterCCZ4RHS<DiracField>, which must agree) and
// of DiracConstraints, at dx = 0.1 and 0.05 on a box around the centre of the star. For an unperturbed star
// d_t psi = -i omega psi and the metric RHS and Ham vanish up to truncation error, which should fall by ~16 between
// the two resolutions (4th order); for a perturbed star only Ham -> 0.
// Modes (third argument): 0 star at rest (default); 1 boosted star (rapidity 0.3): the Hamiltonian and momentum
// constraints must converge to zero, and d_t psi from the RHS must equal the exact time derivative of the boosted
// solution psi_lab(t, x) = U(eta) psi_rest(t~, x~) (computed from DiracStarHelperFunction at t = +-h); 2 equal-mass
// head-on binary: the grid variables must have the x-reflection parities of their names about the centre.
// Build: make all (uses the example's UserVariables.hpp), run: ./box_test_dirac*.ex <A_central> <perturb 0|1> [mode] [1D gridpoints]
#include "BoxIterator.H"
#include "BoxLoops.hpp"
#include "DiracCCZ4RHS.hpp"
#include "DiracConstraints.hpp"
#include "FArrayBox.H"
#include "GammaCalculator.hpp"
#include "MatterCCZ4RHS.hpp"
#include "MovingPunctureGauge.hpp"
#include "SetValue.hpp"
#include "BinaryDiracStar.hpp"
#include "SingleDiracStar.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv)
{
#ifdef CH_MPI
    MPI_Init(&argc, &argv);
#endif
    if (argc < 3)
    {
        printf("usage: %s <A_central> <perturb 0|1>\n", argv[0]);
        return 1;
    }
    const double A = atof(argv[1]);
    const int mode = (argc > 3) ? atoi(argv[3]) : 0;
    DiracStar_params_t idp;
    idp.solver.A_central = A;
    idp.solver.perturb = atoi(argv[2]);
    idp.solver.perturb_amp = 0.01;
    idp.solver.perturb_spread = 3.;
    idp.star_centre = {0.013, -0.021, 0.017}; // off the cell centres and corners
    idp.phase = 0.3;
    idp.write_1d_profiles = false;
    idp.rapidity = (mode == 1) ? 0.3 : 0.;
    if (argc > 4)
        idp.solver.n_gridpoints = atoi(argv[4]); // 1D solver resolution (default 20001 on R = 100)
    DiracField dirac(DiracField::params_t{1.});

    CCZ4_params_t<MovingPunctureGauge::params_t> params;
    params.kappa1 = 0.1; params.kappa2 = 0.; params.kappa3 = 1.; params.covariantZ4 = true;
    params.lapse_advec_coeff = 1.; params.lapse_power = 1.; params.lapse_coeff = 2.;
    params.shift_Gamma_coeff = 0.75; params.shift_advec_coeff = 0.; params.eta = 1.;

    if (mode == 2)
    {
        // equal-mass head-on binary, both stars boosted towards each other, centred on the origin
        BinaryDiracStar_params_t bp;
        bp.star1 = idp;
        bp.star1.star_centre = {0., 0., 0.};
        bp.star1.rapidity = 0.2;
        bp.star2 = bp.star1;
        bp.star1.phase = bp.star2.phase = 0.25;
        bp.separation = 4.;
        bp.impact_parameter = 0.;
        bp.mass_ratio = 1.;
        bp.radius_width1 = bp.radius_width2 = 100.;
        bp.conformal_factor_power = 4;
        const double dx = 0.1;
        const int nx = 30;
        Box box(IntVect(-nx, -nx, -nx), IntVect(nx - 1, nx - 1, nx - 1));
        for (int id = 0; id < 3; ++id)
        {
            bp.id_choice = id;
            FArrayBox in(box, NUM_VARS);
            BinaryDiracStar binary(bp, dx, 0);
            binary.compute_1d_solution();
            BoxLoops::loop(make_compute_pack(SetValue(0.), binary), in, in, box, disable_simd());
            double e_par = 0., e_metric = 0., maxD = 0.;
            const char *names[8] = {"e", "x", "y", "z", "xy", "yz", "xz", "xyz"};
            for (BoxIterator bit(box); bit.ok(); ++bit)
            {
                IntVect iv = bit(), ivm = iv;
                ivm[0] = -iv[0] - 1; // mirror cell about x = 0
                for (int m = 0; m < 8; ++m)
                {
                    const double eps = (std::string(names[m]).find('x') != std::string::npos) ? -1. : 1.;
                    for (int ri = 0; ri < 2; ++ri)
                    {
                        const int c = c_D_e_Re + 2 * m + ri;
                        e_par = std::max(e_par, std::abs(in(ivm, c) - eps * in(iv, c)));
                        maxD = std::max(maxD, std::abs(in(iv, c)));
                    }
                }
                // metric: chi, h_xx even; h_xy odd in x; shift^x odd; lapse even
                e_metric = std::max(e_metric, std::abs(in(ivm, c_chi) - in(iv, c_chi)) +
                                                  std::abs(in(ivm, c_lapse) - in(iv, c_lapse)) +
                                                  std::abs(in(ivm, c_shift1) + in(iv, c_shift1)) +
                                                  std::abs(in(ivm, c_K) - in(iv, c_K)));
            }
            printf("binary id_choice %d: max |D(-x) - parity D(x)| = %.1e (max |D| %.2e), metric/gauge mirror error "
                   "%.1e\n",
                   id, e_par, maxD, e_metric);
        }
#ifdef CH_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    for (double dx : {0.1, 0.05})
    {
        const int nx = static_cast<int>(round(1.5 / dx));
        Box box(IntVect(-nx, -nx, -nx), IntVect(nx - 1, nx - 1, nx - 1));
        Box gbox = box; gbox.grow(6);
        Box mbox = box; mbox.grow(3);
        FArrayBox in(gbox, NUM_VARS), rhs(box, NUM_VARS), rhs2(box, NUM_VARS), diag(box, NUM_DIAGNOSTIC_VARS);
        SingleDiracStar star(idp, dx, 0);
        star.compute_1d_solution();
        BoxLoops::loop(make_compute_pack(SetValue(0.), star), in, in, gbox, disable_simd());
        BoxLoops::loop(GammaCalculator(dx), in, in, mbox, disable_simd());
        BoxLoops::loop(DiracCCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives>(dirac, params, dx, 0.3, 0, 1.), in,
                       rhs, box);
        BoxLoops::loop(MatterCCZ4RHS<DiracField, MovingPunctureGauge, FourthOrderDerivatives>(dirac, params, dx, 0.3,
                                                                                              0, 1.),
                       in, rhs2, box);
        BoxLoops::loop(DiracConstraints(dirac, dx, 1., c_Ham, Interval(c_Mom1, c_Mom3), c_N, c_mod_psi, c_rho), in,
                       diag, box);

        const double w = star.m_1d_sol.omega;
        // exact d_t of the boosted solution (mode 1), from the helper at lab times +-h
        FArrayBox exact_dt(box, 16);
        if (mode == 1)
        {
            DiracStarHelperFunction helper;
            const double h = 1e-4;
            for (BoxIterator bit(box); bit.ok(); ++bit)
            {
                IntVect iv = bit();
                const double x = (iv[0] + 0.5) * dx - idp.star_centre[0], y = (iv[1] + 0.5) * dx - idp.star_centre[1],
                             z = (iv[2] + 0.5) * dx - idp.star_centre[2];
                const auto p = helper.compute_star_vars(x, y, z, idp.rapidity, 0., 0., star.m_1d_sol, idp.phase, h);
                const auto mm = helper.compute_star_vars(x, y, z, idp.rapidity, 0., 0., star.m_1d_sol, idp.phase, -h);
                for (int m = 0; m < 8; ++m)
                {
                    exact_dt(iv, 2 * m) = (p.D_Re[m] - mm.D_Re[m]) / (2. * h);
                    exact_dt(iv, 2 * m + 1) = (p.D_Im[m] - mm.D_Im[m]) / (2. * h);
                }
            }
        }
        double e_psi = 0., n_psi = 0., e_metric = 0., ham = 0., rho16 = 0., mom = 0., modpsi = 0., e_generic = 0.;
        for (BoxIterator bit(box); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            for (int m = 0; m < 8; ++m)
            {
                const double re = in(iv, c_D_e_Re + 2 * m), im = in(iv, c_D_e_Im + 2 * m);
                const double dre = (mode == 1) ? exact_dt(iv, 2 * m) : w * im;      // exact d_t Re
                const double dim = (mode == 1) ? exact_dt(iv, 2 * m + 1) : -w * re; // exact d_t Im
                e_psi = std::max(e_psi, std::max(std::abs(rhs(iv, c_D_e_Re + 2 * m) - dre),
                                                 std::abs(rhs(iv, c_D_e_Im + 2 * m) - dim)));
                n_psi = std::max(n_psi, std::sqrt(dre * dre + dim * dim));
            }
            for (int c = 0; c < c_D_e_Re; ++c)
                e_metric = std::max(e_metric, std::abs(rhs(iv, c)));
            for (int c = 0; c < NUM_VARS; ++c)
                e_generic = std::max(e_generic, std::abs(rhs(iv, c) - rhs2(iv, c)));
            ham = std::max(ham, std::abs(diag(iv, c_Ham)));
            rho16 = std::max(rho16, 16. * M_PI * std::abs(diag(iv, c_rho)));
            mom = std::max(mom, std::abs(diag(iv, c_Mom1)) + std::abs(diag(iv, c_Mom2)) + std::abs(diag(iv, c_Mom3)));
            modpsi = std::max(modpsi, diag(iv, c_mod_psi));
        }
        if (mode == 1)
            e_metric = 0.; // a boosted star moves: the metric RHS is not zero
        printf("mode %d A = %.2f perturb %d dx = %.3f: |d_t psi - exact|/|exact| = %.2e  max|metric rhs| = %.2e  "
               "max|Ham|/max(16 pi rho) = %.2e  max|Mom| = %.2e  max mod_psi = %.5f (A_central %.5f)  "
               "|DiracCCZ4RHS - MatterCCZ4RHS<DiracField>| = %.1e\n",
               mode, A, idp.solver.perturb, dx, e_psi / n_psi, e_metric, ham / rho16, mom, modpsi, star.m_1d_sol.A_central,
               e_generic);
    }
#ifdef CH_MPI
    MPI_Finalize();
#endif
    return 0;
}
