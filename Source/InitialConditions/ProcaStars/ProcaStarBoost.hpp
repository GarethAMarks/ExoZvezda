/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef PROCASTARBOOST_HPP_
#define PROCASTARBOOST_HPP_

#include "ProcaStarSolution.hpp"
#include <cmath>
#include <complex>

/*
 A Proca star boosted along x: the lab-frame 3+1 variables at a point (x, y, z) relative to the star's centre at lab
 time t. Chombo-free and shared by 2DChombo and ExoZvezda; the derivation and checks are in ExoZvezda's
 Examples/SingleProcaStar/derivations/check_proca_boost.py (also in 2DChombo's Examples/ProcaStar_2D/derivations).

 Coordinates (identical to BosonStarHelperFunction / 2DChombo's BosonStar and the Dirac stars):
   t~ = cosh(xi) t + sinh(xi) x,  x~ = sinh(xi) t + cosh(xi) x,  y~ = y,  z~ = z,   xi = rapidity,
 so a star with xi > 0 moves towards -x. With the rest-frame isotropic solution alpha(r~), psi(r~) (gamma = psi^4 delta)
 and P = psi^2 (the boson-star codes' conformal factor):
   gamma_xx = P^2 cosh^2 - alpha^2 sinh^2,  gamma_yy = gamma_zz = P^2,  lapse = alpha P / sqrt(gamma_xx),
   beta^x = sinh cosh (P^2 - alpha^2) / gamma_xx,  K_ij as in BosonStarHelperFunction.

 Vector field. In the rest frame A = e^{-i omega t~ + i pi phase} (f dt~ + i g n~_i dx~^i) and F_{t~ i~} = -E n~_i (same
 phase), F_{i~ j~} = 0, with f, g, E the isotropic profiles of ProcaStarSolution (n~ = x~/r~). A_mu is a covector and F a
 2-form, so the lab components follow from the coordinate Jacobian alone:
   A_t = cosh A_t~ + sinh A_x~,   A_x = sinh A_t~ + cosh A_x~,   A_y = A_y~,   A_z = A_z~,
   F_tx = e_x, F_ty = cosh e_y, F_tz = cosh e_z, F_xy = sinh e_y, F_xz = sinh e_z, F_yz = 0   (e_i = F_{t~ i~}),
 and the 3+1 variables follow from the lab slicing:
   X_i = A_i,   X_phi = -(A_t - beta^x A_x) / lapse,
   E_x = -e_x / lapse,   E_{y,z} = -(cosh - beta^x sinh) e_{y,z} / lapse,   E^i = gamma^{ij} E_j  (gamma diagonal),
   Z = 0.
*/
struct BoostedProcaStarVars
{
    double lapse;
    double shiftx;
    double gLL[3][3];
    double KLL[3][3];
    std::complex<double> Xphi; //!< X_phi = -n^a A_a
    std::complex<double> X[3]; //!< X_i = A_i
    std::complex<double> E[3]; //!< E^i (upper index)
};

inline BoostedProcaStarVars boosted_proca_star(const ProcaStarSolution &sol, double x, double y, double z,
                                               double rapidity, double phase_offset, double t = 0.)
{
    using cplx = std::complex<double>;
    BoostedProcaStarVars out;
    const double c_ = cosh(rapidity), s_ = sinh(rapidity), v_ = tanh(rapidity);
    const double tt = c_ * t + s_ * x; // t~
    const double xx = s_ * t + c_ * x; // x~
    const double r = sqrt(xx * xx + y * y + z * z);

    const double alpha = sol.get_lapse(r), dalpha = sol.get_dlapse(r);
    const double psi = sol.get_psi(r), dpsi = sol.get_dpsi(r);
    const double P = psi * psi, dP = 2. * psi * dpsi;

    const double pc_os = P * P * c_ * c_ - alpha * alpha * s_ * s_;
    const double lapse = alpha * P / sqrt(pc_os);
    const double bx = s_ * c_ * (P * P - alpha * alpha) / pc_os;
    out.lapse = lapse;
    out.shiftx = bx;
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

    // rest-frame A and F (g and E are odd, so g n and E n are regular at the centre)
    const double rinv = (r > 0.) ? 1. / r : 0.;
    const double n[3] = {xx * rinv, y * rinv, z * rinv};
    const cplx ph = std::exp(cplx(0., -sol.omega * tt + M_PI * phase_offset));
    const double f = sol.get_f(r), g = sol.get_g(r), E = sol.get_E(r);
    const cplx At_r = f * ph;
    cplx Ai_r[3], e[3];
    for (int i = 0; i < 3; ++i)
    {
        Ai_r[i] = cplx(0., g * n[i]) * ph;
        e[i] = -E * n[i] * ph;
    }

    const cplx At = c_ * At_r + s_ * Ai_r[0];
    const cplx Ax = s_ * At_r + c_ * Ai_r[0];
    out.X[0] = Ax;
    out.X[1] = Ai_r[1];
    out.X[2] = Ai_r[2];
    out.Xphi = -(At - bx * Ax) / lapse;
    out.E[0] = -e[0] / (lapse * pc_os);
    out.E[1] = -(c_ - bx * s_) * e[1] / (lapse * P * P);
    out.E[2] = -(c_ - bx * s_) * e[2] / (lapse * P * P);
    return out;
}

#endif /* PROCASTARBOOST_HPP_ */
