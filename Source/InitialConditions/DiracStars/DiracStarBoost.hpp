/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIRACSTARBOOST_HPP_
#define DIRACSTARBOOST_HPP_

#include "DiracStarSolution.hpp"
#include <cmath>
#include <complex>

/*
 A Dirac star boosted along x: the lab-frame 3+1 variables and the ingredients of the spinor, at a point (x, y, z)
 relative to the star's centre (and, for the spinor, at lab time t; the metric quantities are for t = 0, as in the
 boson star codes). Chombo-free and shared by 2DChombo and ExoZvezda; the derivation and checks are in ExoZvezda's
 Examples/SingleDiracStar/derivations/check_dirac_boost.py.

 Coordinates (identical to BosonStarHelperFunction / 2DChombo's BosonStar):
   t~ = cosh(xi) t + sinh(xi) x,  x~ = sinh(xi) t + cosh(xi) x,  y~ = y,  z~ = z,   xi = rapidity,
 so a star with xi > 0 moves towards -x. With the rest-frame isotropic solution alpha(r~), psi(r~) (gamma = psi^4 delta)
 and P = psi^2 (the boson-star codes' conformal factor):
   gamma_xx = P^2 cosh^2 - alpha^2 sinh^2,  gamma_yy = gamma_zz = P^2,  lapse = alpha P / sqrt(gamma_xx),
   beta^x = sinh cosh (P^2 - alpha^2) / gamma_xx,  K_ij as in BosonStarHelperFunction.

 Spinor: psi_lab = U(eta) psi_rest(t~, x~), with
   psi_rest = e^{-i omega t~} e^{i pi phase} (g Omega, -i f (sigma.n~) Omega) / sqrt(4 pi)   (n~ = x~ / r~),
   U(eta) = exp((eta/2) gamma^0 gamma^1) = cosh(eta/2) - sinh(eta/2) alpha_x,   tanh(eta) = tanh(xi) alpha / P,
 where alpha_x = gamma_D^0 gamma_D^1 is block off-diagonal with the Pauli matrix along x in both blocks. U relates
 the rest-frame orthonormal tetrad to the lab tetrad (e_0 = n, e_x along d_x), which differ by a local boost of
 rapidity eta in the 0-x plane. The caller builds psi_rest and applies U in its own spinor basis (sigma_x = sigma_1 in
 ExoZvezda, sigma_3 in the 2D code).
*/
struct BoostedDiracStarVars
{
    double lapse;
    double shiftx;
    double gLL[3][3];
    double KLL[3][3];
    // spinor ingredients
    double g, f;                  //!< physical radial spinor functions at r~
    double n[3];                  //!< rest-frame unit radial vector x~ / r~ (zero at the centre)
    std::complex<double> phase;   //!< e^{-i omega t~} e^{i pi phase_offset}
    double cosh_half_eta;         //!< U = cosh(eta/2) - sinh(eta/2) alpha_x
    double sinh_half_eta;
};

inline BoostedDiracStarVars boosted_dirac_star(const DiracStarSolution &sol, double x, double y, double z,
                                               double rapidity, double phase_offset, double t = 0.)
{
    BoostedDiracStarVars out;
    const double c_ = cosh(rapidity), s_ = sinh(rapidity), v_ = tanh(rapidity);
    const double tt = c_ * t + s_ * x; // t~
    const double xx = s_ * t + c_ * x; // x~
    const double r = sqrt(xx * xx + y * y + z * z);

    const double alpha = sol.get_lapse(r), dalpha = sol.get_dlapse(r);
    const double psi = sol.get_psi(r), dpsi = sol.get_dpsi(r);
    // the boson-star formulas use P = psi^2 (gamma = P^2 delta)
    const double P = psi * psi, dP = 2. * psi * dpsi;

    const double pc_os = P * P * c_ * c_ - alpha * alpha * s_ * s_;
    const double lapse = alpha * P / sqrt(pc_os);
    out.lapse = lapse;
    out.shiftx = s_ * c_ * (P * P - alpha * alpha) / pc_os;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            out.gLL[i][j] = 0.;
            out.KLL[i][j] = 0.;
        }
    out.gLL[0][0] = pc_os;
    out.gLL[1][1] = P * P;
    out.gLL[2][2] = P * P;
    if (r > 0.)
    {
        out.KLL[2][2] = -lapse * s_ * xx * dP / (r * P);
        out.KLL[1][1] = out.KLL[2][2];
        out.KLL[0][1] = lapse * c_ * s_ * (y / r) * (dP / P - dalpha / alpha);
        out.KLL[0][2] = lapse * c_ * s_ * (z / r) * (dP / P - dalpha / alpha);
        out.KLL[1][0] = out.KLL[0][1];
        out.KLL[2][0] = out.KLL[0][2];
        out.KLL[0][0] = lapse * (xx / r) * s_ * c_ * c_ *
                        (dP / P - 2. * dalpha / alpha + v_ * v_ * alpha * dalpha / (P * P));
    }

    out.g = sol.get_g(r);
    out.f = sol.get_f(r);
    const double rinv = (r > 0.) ? 1. / r : 0.;
    out.n[0] = xx * rinv;
    out.n[1] = y * rinv;
    out.n[2] = z * rinv;
    out.phase = std::exp(std::complex<double>(0., -sol.omega * tt + M_PI * phase_offset));
    const double eta = atanh(v_ * alpha / P);
    out.cosh_half_eta = cosh(0.5 * eta);
    out.sinh_half_eta = sinh(0.5 * eta);
    return out;
}

#endif /* DIRACSTARBOOST_HPP_ */
