// Single-box unit test of the assembled 3D Proca code (no time evolution; a few seconds on one core):
// initial data for a star -> one evaluation of MatterCCZ4RHS<ProcaField> and of ProcaConstraints, at dx = 0.1 and 0.05
// on a box around the centre of the star. For an unperturbed star d_t (X_i, E^i, X_phi) = -i omega (X_i, E^i, X_phi),
// d_t Z = 0, and the metric RHS, Ham, Mom and the Gauss constraint vanish up to truncation error, which should fall by
// ~16 between the two resolutions (4th order); for a perturbed star only the constraints -> 0.
// Modes (third argument): 0 star at rest (default); 1 boosted star (rapidity 0.3): the constraints must converge to
// zero, and the RHS of the Proca variables must equal the exact time derivative of the boosted solution (computed
// from boosted_proca_star at t = +-h); 2 equal-mass head-on binary: the grid variables must have the x-reflection
// parities (X_x, E^x odd) about the centre.
// Build: make all (uses the example's UserVariables.hpp), run:
//   ./box_test_proca*.ex <A_central> <perturb 0|1> [mode] [1D gridpoints]
#include "BinaryProcaStar.hpp"
#include "BoxIterator.H"
#include "BoxLoops.hpp"
#include "FArrayBox.H"
#include "GammaCalculator.hpp"
#include "MatterCCZ4RHS.hpp"
#include "MovingPunctureGauge.hpp"
#include "ProcaConstraints.hpp"
#include "SetValue.hpp"
#include "SingleProcaStar.hpp"
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv)
{
#ifdef CH_MPI
    MPI_Init(&argc, &argv);
#endif
    if (argc < 3)
    {
        printf("usage: %s <A_central> <perturb 0|1> [mode] [1D gridpoints]\n", argv[0]);
        return 1;
    }
    const double A = atof(argv[1]);
    const int mode = (argc > 3) ? atoi(argv[3]) : 0;
    ProcaStar_params_t idp;
    idp.solver.A_central = A;
    idp.solver.perturb = atoi(argv[2]);
    idp.solver.perturb_amp = 0.01;
    idp.solver.perturb_spread = 3.;
    idp.star_centre = {0.013, -0.021, 0.017}; // off the cell centres and corners
    idp.phase = 0.3;
    idp.write_1d_profiles = false;
    idp.rapidity = (mode == 1) ? 0.3 : 0.;
    if (argc > 4)
        idp.solver.n_gridpoints = atoi(argv[4]); // 1D solver resolution (default 30001 on R = 150)
    const double kappa = 1.;
    ProcaField proca(ProcaField::params_t{1., kappa});

    CCZ4_params_t<MovingPunctureGauge::params_t> params;
    params.kappa1 = 0.1; params.kappa2 = 0.; params.kappa3 = 1.; params.covariantZ4 = true;
    params.lapse_advec_coeff = 1.; params.lapse_power = 1.; params.lapse_coeff = 2.;
    params.shift_Gamma_coeff = 0.75; params.shift_advec_coeff = 0.; params.eta = 1.;

    if (mode == 2)
    {
        // equal-mass head-on binary, both stars boosted towards each other, centred on the origin
        BinaryProcaStar_params_t bp;
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
            BinaryProcaStar binary(bp, dx, 0);
            binary.compute_1d_solution();
            BoxLoops::loop(make_compute_pack(SetValue(0.), binary), in, in, box, disable_simd());
            double e_par = 0., e_metric = 0., maxP = 0.;
            for (BoxIterator bit(box); bit.ok(); ++bit)
            {
                IntVect iv = bit(), ivm = iv;
                ivm[0] = -iv[0] - 1; // mirror cell about x = 0
                for (int c = c_Xx_Re; c < NUM_VARS; ++c)
                {
                    const bool odd_x = (c == c_Xx_Re || c == c_Xx_Im || c == c_Ex_Re || c == c_Ex_Im);
                    const double eps = odd_x ? -1. : 1.;
                    e_par = std::max(e_par, std::abs(in(ivm, c) - eps * in(iv, c)));
                    maxP = std::max(maxP, std::abs(in(iv, c)));
                }
                e_metric = std::max(e_metric, std::abs(in(ivm, c_chi) - in(iv, c_chi)) +
                                                  std::abs(in(ivm, c_lapse) - in(iv, c_lapse)) +
                                                  std::abs(in(ivm, c_shift1) + in(iv, c_shift1)) +
                                                  std::abs(in(ivm, c_K) - in(iv, c_K)));
            }
            printf("binary id_choice %d: max |Proca var(-x) - parity var(x)| = %.1e (max |var| %.2e), metric/gauge "
                   "mirror error %.1e\n",
                   id, e_par, maxP, e_metric);
        }
#ifdef CH_MPI
        MPI_Finalize();
#endif
        return 0;
    }

    const int nP = NUM_VARS - c_Xx_Re; // 16 Proca variables
    for (double dx : {0.1, 0.05})
    {
        const int nx = static_cast<int>(round(1.5 / dx));
        Box box(IntVect(-nx, -nx, -nx), IntVect(nx - 1, nx - 1, nx - 1));
        Box gbox = box; gbox.grow(6);
        Box mbox = box; mbox.grow(3);
        FArrayBox in(gbox, NUM_VARS), rhs(box, NUM_VARS), diag(box, NUM_DIAGNOSTIC_VARS);
        SingleProcaStar star(idp, dx, 0);
        star.compute_1d_solution();
        const ProcaStarSolution &sol = star.m_1d_sol;
        BoxLoops::loop(make_compute_pack(SetValue(0.), star), in, in, gbox, disable_simd());
        BoxLoops::loop(GammaCalculator(dx), in, in, mbox, disable_simd());
        BoxLoops::loop(MatterCCZ4RHS<ProcaField, MovingPunctureGauge, FourthOrderDerivatives>(proca, params, dx, 0.3,
                                                                                              0, 1.),
                       in, rhs, box);
        BoxLoops::loop(ProcaConstraints(proca, dx, 1., c_Ham, Interval(c_Mom1, c_Mom3), c_N, c_mod_Xphi, c_rho, c_n,
                                        c_Gauss),
                       in, diag, box);

        const double w = sol.omega;
        double e_P = 0., n_P = 0., e_metric = 0., ham = 0., rho16 = 0., mom = 0., gauss = 0., mu2X = 0.;
        double modX = 0., e_n = 0., n_max = 0.;
        for (BoxIterator bit(box); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            const double x = (iv[0] + 0.5) * dx - idp.star_centre[0], y = (iv[1] + 0.5) * dx - idp.star_centre[1],
                         z = (iv[2] + 0.5) * dx - idp.star_centre[2];
            double exact[16];
            if (mode == 1)
            {
                // exact d_t of the boosted solution, from the closed form at lab times +-h
                const double h = 1e-4;
                const auto p = boosted_proca_star(sol, x, y, z, idp.rapidity, idp.phase, h);
                const auto m = boosted_proca_star(sol, x, y, z, idp.rapidity, idp.phase, -h);
                auto d = [&](std::complex<double> a, std::complex<double> b) { return (a - b) / (2. * h); };
                for (int i = 0; i < 3; ++i)
                {
                    exact[i] = d(p.X[i], m.X[i]).real();
                    exact[3 + i] = d(p.X[i], m.X[i]).imag();
                    exact[6 + i] = d(p.E[i], m.E[i]).real();
                    exact[9 + i] = d(p.E[i], m.E[i]).imag();
                }
                exact[12] = d(p.Xphi, m.Xphi).real();
                exact[13] = d(p.Xphi, m.Xphi).imag();
            }
            else
            {
                // -i omega: d_t Re = omega Im, d_t Im = -omega Re (pairs (Re, Im) are 3 apart for the vectors)
                for (int i = 0; i < 3; ++i)
                {
                    exact[i] = w * in(iv, c_Xx_Im + i);
                    exact[3 + i] = -w * in(iv, c_Xx_Re + i);
                    exact[6 + i] = w * in(iv, c_Ex_Im + i);
                    exact[9 + i] = -w * in(iv, c_Ex_Re + i);
                }
                exact[12] = w * in(iv, c_Xphi_Im);
                exact[13] = -w * in(iv, c_Xphi_Re);
            }
            exact[14] = exact[15] = 0.; // Z stays zero
            for (int k = 0; k < nP; ++k)
            {
                e_P = std::max(e_P, std::abs(rhs(iv, c_Xx_Re + k) - exact[k]));
                n_P = std::max(n_P, std::abs(exact[k]));
            }
            for (int c = 0; c < c_Xx_Re; ++c)
                e_metric = std::max(e_metric, std::abs(rhs(iv, c)));
            ham = std::max(ham, std::abs(diag(iv, c_Ham)));
            rho16 = std::max(rho16, 16. * M_PI * std::abs(diag(iv, c_rho)));
            mom = std::max(mom, std::abs(diag(iv, c_Mom1)) + std::abs(diag(iv, c_Mom2)) + std::abs(diag(iv, c_Mom3)));
            gauss = std::max(gauss, diag(iv, c_Gauss));
            mu2X = std::max(mu2X, std::sqrt(in(iv, c_Xphi_Re) * in(iv, c_Xphi_Re) + in(iv, c_Xphi_Im) * in(iv, c_Xphi_Im)));
            modX = std::max(modX, diag(iv, c_mod_Xphi));
            if (mode == 0)
            {
                // number density of the star at rest: n = -g E / (alpha psi^4) (isotropic profiles)
                const double r = std::sqrt(x * x + y * y + z * z), psi = sol.get_psi(r);
                const double n_exact = -sol.get_g(r) * sol.get_E(r) / (sol.get_lapse(r) * pow(psi, 4));
                e_n = std::max(e_n, std::abs(diag(iv, c_n) - n_exact));
                n_max = std::max(n_max, std::abs(n_exact));
            }
        }
        if (mode == 1)
            e_metric = 0.; // a boosted star moves: the metric RHS is not zero
        printf("mode %d A = %.3f perturb %d dx = %.3f: |d_t Proca - exact|/|exact| = %.2e  max|metric rhs| = %.2e  "
               "max|Ham|/max(16 pi rho) = %.2e  max|Mom| = %.2e  max|Gauss|/max|mu^2 X_phi| = %.2e\n",
               mode, A, idp.solver.perturb, dx, e_P / n_P, e_metric, ham / rho16, mom, gauss / mu2X);
        printf("   max mod_Xphi = %.6f (A_central %.6f)", modX, sol.A_central);
        if (mode == 0)
            printf("   max |n - n_exact| / max n_exact = %.1e (max n_exact %.3e)", e_n / n_max, n_max);
        printf("\n");
    }
#ifdef CH_MPI
    MPI_Finalize();
#endif
    return 0;
}
