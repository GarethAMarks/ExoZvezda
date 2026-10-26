// Single-box unit test of the assembled 3D Dirac code (no time evolution; a few seconds on one core):
// initial data for a star -> one evaluation of DiracCCZ4RHS (and of MatterCCZ4RHS<DiracField>, which must agree) and
// of DiracConstraints, at dx = 0.1 and 0.05 on a box around the centre of the star. For an unperturbed star
// d_t psi = -i omega psi and the metric RHS and Ham vanish up to truncation error, which should fall by ~16 between
// the two resolutions (4th order); for a perturbed star only Ham -> 0.
// Build: make all (uses the example's UserVariables.hpp), run: ./box_test_dirac*.ex <A_central> <perturb 0|1>
#include "BoxIterator.H"
#include "BoxLoops.hpp"
#include "DiracCCZ4RHS.hpp"
#include "DiracConstraints.hpp"
#include "FArrayBox.H"
#include "GammaCalculator.hpp"
#include "MatterCCZ4RHS.hpp"
#include "MovingPunctureGauge.hpp"
#include "SetValue.hpp"
#include "SingleDiracStar.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    DiracStar_params_t idp;
    idp.solver.A_central = A;
    idp.solver.perturb = atoi(argv[2]);
    idp.solver.perturb_amp = 0.01;
    idp.solver.perturb_spread = 3.;
    idp.star_centre = {0.013, -0.021, 0.017}; // off the cell centres and corners
    idp.phase = 0.3;
    idp.write_1d_profiles = false;
    DiracField dirac(DiracField::params_t{1.});

    CCZ4_params_t<MovingPunctureGauge::params_t> params;
    params.kappa1 = 0.1; params.kappa2 = 0.; params.kappa3 = 1.; params.covariantZ4 = true;
    params.lapse_advec_coeff = 1.; params.lapse_power = 1.; params.lapse_coeff = 2.;
    params.shift_Gamma_coeff = 0.75; params.shift_advec_coeff = 0.; params.eta = 1.;

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
        double e_psi = 0., n_psi = 0., e_metric = 0., ham = 0., rho16 = 0., mom = 0., modpsi = 0., e_generic = 0.;
        for (BoxIterator bit(box); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            for (int m = 0; m < 8; ++m)
            {
                const double re = in(iv, c_D_e_Re + 2 * m), im = in(iv, c_D_e_Im + 2 * m);
                e_psi = std::max(e_psi, std::max(std::abs(rhs(iv, c_D_e_Re + 2 * m) - w * im),
                                                 std::abs(rhs(iv, c_D_e_Im + 2 * m) + w * re)));
                n_psi = std::max(n_psi, w * std::sqrt(re * re + im * im));
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
        printf("A = %.2f perturb %d dx = %.3f: |d_t psi + i w psi|/|w psi| = %.2e  max|metric rhs| = %.2e  "
               "max|Ham|/max(16 pi rho) = %.2e  max|Mom| = %.2e  max mod_psi = %.5f (A_central %.5f)  "
               "|DiracCCZ4RHS - MatterCCZ4RHS<DiracField>| = %.1e\n",
               A, idp.solver.perturb, dx, e_psi / n_psi, e_metric, ham / rho16, mom, modpsi, star.m_1d_sol.A_central,
               e_generic);
    }
#ifdef CH_MPI
    MPI_Finalize();
#endif
    return 0;
}
