/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIRACSTARHELPERFUNCTION_HPP_
#define DIRACSTARHELPERFUNCTION_HPP_

#include "DiracStarBoost.hpp"
#include "DiracStarSolution.hpp"
#include <cmath>
#include <complex>

/*
 The 3D variables of a single (possibly boosted) Dirac star: lab metric, lapse, shift, extrinsic curvature (as
 BosonStarHelperFunction) and the 8 complex grid variables D of the two spinors (see DiracField.hpp), built from the
 rest-frame spinors psi_m = e^{-i omega t~} (g chi_m, -i f (sigma.n~) chi_m) / sqrt(4 pi), m = +-1/2, by the local
 boost U(eta) = cosh(eta/2) - sinh(eta/2) alpha_x (derived and checked in
 Examples/SingleDiracStar/derivations/check_dirac_boost.py). Binaries superpose these as the boson-star binaries do.
*/
class DiracStarHelperFunction
{
  public:
    struct DS_3d_vars
    {
        double lapse;
        double shiftx;
        double gLL[3][3];
        double KLL[3][3];
        double D_Re[8]; //!< order e, x, y, z, xy, yz, xz, xyz
        double D_Im[8];
    };

    //! x, y, z are the lab coordinates of the point relative to the grid centre used by the caller; the star sits at
    //! x = -separation_offset, y = -impact_offset (as in BosonStarHelperFunction). t is the lab time (0 for initial
    //! data; other values are used by the tests).
    DS_3d_vars compute_star_vars(double x, double y, double z, double rapidity, double separation_offset,
                                 double impact_offset, const DiracStarSolution &sol, double phase_offset,
                                 double t = 0.) const
    {
        using cd = std::complex<double>;
        const BoostedDiracStarVars b =
            boosted_dirac_star(sol, x + separation_offset, y + impact_offset, z, rapidity, phase_offset, t);
        DS_3d_vars out;
        out.lapse = b.lapse;
        out.shiftx = b.shiftx;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                out.gLL[i][j] = b.gLL[i][j];
                out.KLL[i][j] = b.KLL[i][j];
            }

        // rest-frame spinors (standard Pauli matrices, spin along z): sigma.n chi_+ = (n_z, n_x + i n_y),
        // sigma.n chi_- = (n_x - i n_y, -n_z)
        const cd I(0., 1.);
        const double nx = b.n[0], ny = b.n[1], nz = b.n[2];
        const cd pref = b.phase / sqrt(4. * M_PI);
        cd A[4] = {b.g, 0., -I * b.f * nz, -I * b.f * cd(nx, ny)};
        cd B[4] = {0., b.g, -I * b.f * cd(nx, -ny), I * b.f * nz};
        // lab components: U = cosh(eta/2) - sinh(eta/2) alpha_x, alpha_x = [[0, sigma_1], [sigma_1, 0]]
        const double ch = b.cosh_half_eta, sh = b.sinh_half_eta;
        auto boost = [&](cd *p) {
            const cd u0 = p[0], u1 = p[1], l0 = p[2], l1 = p[3];
            p[0] = pref * (ch * u0 - sh * l1);
            p[1] = pref * (ch * u1 - sh * l0);
            p[2] = pref * (ch * l0 - sh * u1);
            p[3] = pref * (ch * l1 - sh * u0);
        };
        boost(A);
        boost(B);

        // grid combinations (DiracField.hpp): e = A0+B1, x = A3+B2, y = A3-B2, z = A2-B3, xy = A0-B1, yz = A1+B0,
        // xz = A1-B0, xyz = A2+B3, all / sqrt2
        const double r2 = 1. / sqrt(2.);
        const cd D[8] = {r2 * (A[0] + B[1]), r2 * (A[3] + B[2]), r2 * (A[3] - B[2]), r2 * (A[2] - B[3]),
                         r2 * (A[0] - B[1]), r2 * (A[1] + B[0]), r2 * (A[1] - B[0]), r2 * (A[2] + B[3])};
        for (int k = 0; k < 8; ++k)
        {
            out.D_Re[k] = D[k].real();
            out.D_Im[k] = D[k].imag();
        }
        return out;
    }
};

#endif /* DIRACSTARHELPERFUNCTION_HPP_ */
