/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(DIRACSTARSOLUTION_HPP_)
#error "This file should only be included through DiracStarSolution.hpp"
#endif

#ifndef DIRACSTARSOLUTION_IMPL_HPP_
#define DIRACSTARSOLUTION_IMPL_HPP_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace DiracStarSolutionHelpers
{
inline int bound(int n, int lower, int upper) { return std::max(lower, std::min(n, upper)); }

//! Lagrange interpolation through (X[i], Y[i])
inline double lagrange(double x, const double *X, const double *Y, int n)
{
    double out = 0.;
    for (int i = 0; i < n; ++i)
    {
        if (x == X[i])
            return Y[i];
        double li = 1.;
        for (int j = 0; j < n; ++j)
            if (j != i)
                li *= (x - X[j]) / (X[i] - X[j]);
        out += li * Y[i];
    }
    return out;
}

//! cubic interpolation on the uniform grid r_j = j dr through the points j0 .. j0 + 3
inline double cubic(double r, double f0, double f1, double f2, double f3, int j0, double dr)
{
    const double X[4] = {j0 * dr, (j0 + 1) * dr, (j0 + 2) * dr, (j0 + 3) * dr};
    const double Y[4] = {f0, f1, f2, f3};
    return lagrange(r, X, Y, 4);
}
} // namespace DiracStarSolutionHelpers

inline void DiracStarSolution::matter_terms(double r, long double frequency, const DiracFieldState &s, double &rho,
                                            double &p_r, double &n) const
{
    const double alpha = exp(s.phi);
    const double w_a = static_cast<double>(frequency) / alpha;
    double Gr, Fr, FG_r3; // G/r, F/r and F G / r^3
    if (r == 0.)
    {
        Gr = A_central * sqrt(alpha);
        Fr = 0.;
        FG_r3 = (mu - w_a) * Gr * Gr / 3.;
    }
    else
    {
        Gr = s.G / r;
        Fr = s.F / r;
        FG_r3 = Fr * Gr / r;
    }
    const double norm = 1. / (2. * M_PI * alpha);
    n = norm * (Fr * Fr + Gr * Gr);
    rho = w_a * n;
    p_r = norm * (w_a * (Fr * Fr + Gr * Gr) - mu * (Gr * Gr - Fr * Fr) + 2. * FG_r3);
}

inline DiracFieldState DiracStarSolution::state_RHS(double r, long double frequency, const DiracFieldState &s,
                                                    bool asymptotic) const
{
    double rho, p_r, n;
    matter_terms(r, frequency, s, rho, p_r, n);
    const double w_a = static_cast<double>(frequency) * exp(-s.phi);
    const double T1 = (r == 0.) ? 0. : 0.5 * (s.X * s.X - 1.) / r;
    const double dX = s.X * (4. * M_PI * Gn * r * s.X * s.X * rho - T1);
    const double dPhi = T1 + 4. * M_PI * Gn * r * s.X * s.X * p_r;
    if (asymptotic)
        return DiracFieldState{0., 0., dX, dPhi};
    const double Gr = (r == 0.) ? A_central * exp(0.5 * s.phi) : s.G / r;
    const double Fr = (r == 0.) ? 0. : s.F / r;
    const double dF = s.X * (-Fr + (mu - w_a) * s.G);
    const double dG = s.X * (Gr + (mu + w_a) * s.F);
    return DiracFieldState{dF, dG, dX, dPhi};
}

namespace DiracStarSolutionHelpers
{
inline DiracFieldState axpy(const DiracFieldState &s, double h, const DiracFieldState &k)
{
    return DiracFieldState{s.F + h * k.F, s.G + h * k.G, s.X + h * k.X, s.phi + h * k.phi};
}
inline DiracFieldState rk4_combine(const DiracFieldState &s, double h, const DiracFieldState &k1,
                                   const DiracFieldState &k2, const DiracFieldState &k3, const DiracFieldState &k4)
{
    return DiracFieldState{s.F + h / 6. * (k1.F + 2. * k2.F + 2. * k3.F + k4.F),
                           s.G + h / 6. * (k1.G + 2. * k2.G + 2. * k3.G + k4.G),
                           s.X + h / 6. * (k1.X + 2. * k2.X + 2. * k3.X + k4.X),
                           s.phi + h / 6. * (k1.phi + 2. * k2.phi + 2. * k3.phi + k4.phi)};
}
} // namespace DiracStarSolutionHelpers

inline void DiracStarSolution::rk4_solve(long double freq)
{
    using namespace DiracStarSolutionHelpers;
    state.assign(n_gridpoints, DiracFieldState{});
    radius_array.assign(n_gridpoints, 0.);
    state[0] = DiracFieldState{0., 0., 1., log(m_p.alpha_central)};
    blowup_point = n_gridpoints;
    for (int j = 0; j < n_gridpoints - 1; j++)
    {
        const double r = j * dr;
        const DiracFieldState s1 = state_RHS(r, freq, state[j], false);
        const DiracFieldState s2 = state_RHS(r + dr / 2., freq, axpy(state[j], 0.5 * dr, s1), false);
        const DiracFieldState s3 = state_RHS(r + dr / 2., freq, axpy(state[j], 0.5 * dr, s2), false);
        const DiracFieldState s4 = state_RHS(r + dr, freq, axpy(state[j], dr, s3), false);
        state[j + 1] = rk4_combine(state[j], dr, s1, s2, s3, s4);
        radius_array[j + 1] = (j + 1) * dr;
        if (std::isnan(state[j].F) || std::isnan(state[j].G) || std::isnan(state[j].X) || std::isnan(state[j].phi) ||
            std::isinf(state[j].G) || std::isinf(state[j].X))
        {
            blowup_point = j;
            return;
        }
    }
}

inline double DiracStarSolution::m(int j) const
{
    return 0.5 * radius_array[j] * (1. - 1. / (state[j].X * state[j].X));
}

inline int DiracStarSolution::count_zero_crossings() const
{
    int zc = 0;
    for (int j = 2; j < blowup_point - 1; j++)
        if ((state[j].G == 0.) || (state[j].G > 0. && state[j - 1].G < 0.) || (state[j].G < 0. && state[j - 1].G > 0.))
            zc++;
    return zc;
}

inline long double DiracStarSolution::find_frequency(std::ostream &os, bool quiet)
{
    const int eigen = m_p.eigen;
    const long double epsilon = m_p.freq_epsilon;
    long double lower_guess = m_p.frequency_guess;

    rk4_solve(lower_guess);
    // the bisection needs a lower bound with <= eigen nodes and an upper bound (2 x lower) with more: bracket upwards
    // first if the guess is too small (the 1D code only warns here)
    while (count_zero_crossings() <= eigen && lower_guess < 1e4)
    {
        lower_guess *= 2.;
        rk4_solve(lower_guess);
    }
    while (count_zero_crossings() > eigen && lower_guess > epsilon)
    {
        lower_guess /= 2.;
        rk4_solve(lower_guess);
    }
    if (lower_guess <= epsilon)
        os << "DiracStarSolution ERROR: could not find a lower frequency bound for A_central = " << A_central
            << std::endl;

    long double upper_guess = 2. * lower_guess;
    int num_solves = 0;
    while ((upper_guess - lower_guess) > epsilon && num_solves < 100)
    {
        const long double midpoint = 0.5 * (upper_guess + lower_guess);
        rk4_solve(midpoint);
        if (count_zero_crossings() > eigen)
            upper_guess = midpoint;
        else
            lower_guess = midpoint;
        num_solves++;
    }
    omega_ld = lower_guess;
    rk4_solve(omega_ld);
    if (count_zero_crossings() != eigen)
        os << "DiracStarSolution WARNING: zero crossing count may be incorrect for A_central = " << A_central
            << ", found " << count_zero_crossings() << std::endl;
    if (!quiet)
        os << "DiracStarSolution: found eigenstate " << eigen << " with unrescaled frequency "
            << static_cast<double>(omega_ld) << std::endl;
    return lower_guess;
}

inline int DiracStarSolution::find_last_minimum() const
{
    double G_max = 0.;
    for (int j = 0; j < blowup_point - 1 && j < n_gridpoints; j++)
        G_max = std::max(G_max, std::abs(state[j].G));
    for (int j = blowup_point - 2; j > 1; j--)
        if (std::abs(state[j].G) <= std::abs(state[j - 1].G) && std::abs(state[j].G) <= std::abs(state[j + 1].G) &&
            std::abs(state[j].G) < 0.05 * G_max)
            return j;
    if (blowup_point == n_gridpoints &&
        std::abs(state[n_gridpoints - 1].G) <= std::abs(state[n_gridpoints - 2].G))
        return n_gridpoints - 1;
    return -1;
}

inline void DiracStarSolution::rescale_lapse(double phi_shift)
{
    const double fac = exp(0.5 * phi_shift);
    for (DiracFieldState &s : state)
    {
        s.phi += phi_shift;
        s.F *= fac;
        s.G *= fac;
    }
    omega_ld *= exp(phi_shift);
}

inline double DiracStarSolution::get_noether_charge() const
{
    double Q = 0.;
    for (int j = 1; j < n_gridpoints; j++)
    {
        const double w = (j == n_gridpoints - 1) ? 0.5 : 1.;
        Q += w * dr * 2. * (state[j].F * state[j].F + state[j].G * state[j].G) * state[j].X * exp(-state[j].phi);
    }
    return Q;
}

inline void DiracStarSolution::compute_r99()
{
    int j = 0;
    while (j < n_gridpoints - 1 && m(j) < 0.99 * M)
    {
        if (m(j + 1) != m(j))
            r_99 = dr * j + dr * (0.99 * M - m(j)) / (m(j + 1) - m(j));
        j++;
    }
}

inline bool DiracStarSolution::fill_asymptotic(std::ostream &os, bool quiet)
{
    using namespace DiracStarSolutionHelpers;
    const int min_index = find_last_minimum();
    if (min_index == -1)
    {
        os << "DiracStarSolution ERROR: no local minimum in |G| found for A_central = " << A_central << std::endl;
        return false;
    }
    const int j_match = static_cast<int>(round(m_p.r_match_fac * min_index));
    const double r_match = dr * j_match;
    const double alpha_match = exp(state[j_match].phi);
    const double k =
        state[j_match].X * sqrt(mu * mu - static_cast<double>((omega_ld / alpha_match) * (omega_ld / alpha_match)));
    const double F_match = state[j_match].F, G_match = state[j_match].G;
    if (std::isnan(k))
    {
        os << "DiracStarSolution ERROR: omega / alpha exceeds mu at the matching point for A_central = "
            << A_central << std::endl;
        return false;
    }
    for (int j = j_match; j < n_gridpoints - 1; j++)
    {
        const double r = j * dr;
        const DiracFieldState s1 = state_RHS(r, omega_ld, state[j], true);
        const DiracFieldState s2 = state_RHS(r + dr / 2., omega_ld, axpy(state[j], 0.5 * dr, s1), true);
        const DiracFieldState s3 = state_RHS(r + dr / 2., omega_ld, axpy(state[j], 0.5 * dr, s2), true);
        const DiracFieldState s4 = state_RHS(r + dr, omega_ld, axpy(state[j], dr, s3), true);
        state[j + 1] = rk4_combine(state[j], dr, s1, s2, s3, s4);
        radius_array[j + 1] = (j + 1) * dr;
        const double decay = exp(-k * (radius_array[j + 1] - r_match));
        state[j + 1].F = F_match * decay;
        state[j + 1].G = G_match * decay;
        if (std::isnan(state[j].F) || std::isnan(state[j].X) || std::isnan(state[j].phi) || std::isnan(state[j].G))
        {
            os << "DiracStarSolution WARNING: nan in the asymptotic region for A_central = " << A_central
                << std::endl;
            return false;
        }
    }
    // alpha(infinity) = 1 via the Schwarzschild condition phi = -ln X at the outer edge
    const double phi_shift = -log(state[n_gridpoints - 1].X) - state[n_gridpoints - 1].phi;
    omega_pre_rescale = omega_ld;
    rescale_lapse(phi_shift);
    M = m(n_gridpoints - 1);
    compute_r99();
    noether_charge = get_noether_charge();
    binding_energy = M - mu * noether_charge;
    if (!quiet)
        os << "DiracStarSolution: A_central = " << A_central << ", omega = " << static_cast<double>(omega_ld)
            << ", M = " << M << ", N = " << noether_charge << ", binding energy = " << binding_energy
            << ", r_99 = " << r_99 << std::endl;
    return true;
}

inline void DiracStarSolution::add_perturbation(double a, double k, double center, std::ostream &os, bool quiet)
{
    using namespace DiracStarSolutionHelpers;
    const int N = n_gridpoints;
    std::vector<double> u(N), v(N);
    for (int j = 0; j < N; j++)
    {
        const double r = j * dr;
        const double sa = exp(0.5 * state[j].phi);
        const double fac = 1. + a * exp(-pow(r - center, 2.) / (k * k));
        u[j] = fac * ((j == 0) ? A_central : state[j].G / (sa * r));
        v[j] = fac * ((j == 0) ? 0. : state[j].F / (sa * r));
    }
    auto deriv = [&](const std::vector<double> &y, bool odd) {
        std::vector<double> d(N);
        auto at = [&](int j) { return (j >= 0) ? y[j] : (odd ? -y[-j] : y[-j]); };
        for (int j = 0; j < N; j++)
        {
            if (j <= N - 3)
                d[j] = (at(j - 2) - 8. * at(j - 1) + 8. * at(j + 1) - y[j + 2]) / (12. * dr);
            else
                d[j] = (3. * y[j - 4] - 16. * y[j - 3] + 36. * y[j - 2] - 48. * y[j - 1] + 25. * y[j]) / (12. * dr);
        }
        return d;
    };
    const std::vector<double> du = deriv(u, false), dv = deriv(v, true);
    auto interp = [&](const std::vector<double> &y, double r) {
        const int j0 = bound(static_cast<int>(floor(r / dr)) - 1, 0, N - 4);
        return cubic(r, y[j0], y[j0 + 1], y[j0 + 2], y[j0 + 3], j0, dr);
    };
    auto rhs = [&](double r, double X, double uu, double vv, double duu, double dvv, double &dX, double &dPhi) {
        const double S_rr = (vv * duu - uu * dvv) / (2. * M_PI * X);
        const double v_r = (r == 0.) ? dvv : vv / r;
        const double S_tt = -uu * v_r / (2. * M_PI);
        const double rho = mu * (uu * uu - vv * vv) / (2. * M_PI) + S_rr + 2. * S_tt;
        const double T1 = (r == 0.) ? 0. : 0.5 * (X * X - 1.) / r;
        dX = X * (4. * M_PI * Gn * r * X * X * rho - T1);
        dPhi = T1 + 4. * M_PI * Gn * r * X * X * S_rr;
    };
    state[0].X = 1.;
    for (int j = 0; j < N - 1; j++)
    {
        const double r = j * dr, rm = r + 0.5 * dr;
        const double X = state[j].X, phi = state[j].phi;
        const double um = interp(u, rm), vm = interp(v, rm), dum = interp(du, rm), dvm = interp(dv, rm);
        double k1X, k1p, k2X, k2p, k3X, k3p, k4X, k4p;
        rhs(r, X, u[j], v[j], du[j], dv[j], k1X, k1p);
        rhs(rm, X + 0.5 * dr * k1X, um, vm, dum, dvm, k2X, k2p);
        rhs(rm, X + 0.5 * dr * k2X, um, vm, dum, dvm, k3X, k3p);
        rhs(r + dr, X + dr * k3X, u[j + 1], v[j + 1], du[j + 1], dv[j + 1], k4X, k4p);
        state[j + 1].X = X + (dr / 6.) * (k1X + 2. * k2X + 2. * k3X + k4X);
        state[j + 1].phi = phi + (dr / 6.) * (k1p + 2. * k2p + 2. * k3p + k4p);
        if (std::isnan(state[j + 1].X) || std::isnan(state[j + 1].phi))
        {
            os << "DiracStarSolution ERROR: constraint solve for the perturbed star failed at r = " << r + dr
                << std::endl;
            return;
        }
    }
    for (int j = 0; j < N; j++)
    {
        const double r = j * dr;
        const double sa = exp(0.5 * state[j].phi);
        state[j].G = sa * r * u[j];
        state[j].F = sa * r * v[j];
    }
    A_central = u[0];
    rescale_lapse(-log(state[N - 1].X) - state[N - 1].phi);
    const double M_old = M, N_old = noether_charge;
    M = m(N - 1);
    noether_charge = get_noether_charge();
    binding_energy = M - mu * noether_charge;
    compute_r99();
    if (!quiet)
        os << "DiracStarSolution: perturbed with amplitude " << a << ", width " << k << ", centre " << center
            << ": M = " << M_old << " -> " << M << ", N = " << N_old << " -> " << noether_charge
            << ", binding energy = " << binding_energy << std::endl;
}

inline double DiracStarSolution::f_RHS(double r, double f) const
{
    using namespace DiracStarSolutionHelpers;
    const int j_low = static_cast<int>(floor((r / R) * (n_gridpoints - 1)));
    const int j0 = bound(j_low - 1, 0, n_gridpoints - 4);
    const double X = cubic(r, state[j0].X, state[j0 + 1].X, state[j0 + 2].X, state[j0 + 3].X, j0, dr);
    const double r2 = (r == 0.) ? 10e-10 : r;
    return f * (X - 1.) / r2;
}

inline double DiracStarSolution::r_areal(int j_iso) const
{
    using namespace DiracStarSolutionHelpers;
    if (j_iso == 0)
        return 0.;
    const double r_iso = dr * j_iso;
    // first index with r_iso_array[j] >= r_iso (r_iso_array is increasing)
    const int j_areal =
        static_cast<int>(std::lower_bound(r_iso_array.begin(), r_iso_array.end(), r_iso) - r_iso_array.begin());
    if (j_areal == n_gridpoints)
        return r_iso + M + M * M / (4. * r_iso);
    const int j0 = bound(j_areal - 2, -2, n_gridpoints - 4);
    double rr[4], ra[4];
    for (int i = 0; i < 4; ++i)
    {
        rr[i] = (j0 + i >= 0) ? r_iso_array[j0 + i] : 0.;
        ra[i] = (j0 + i) * dr;
    }
    // near the centre use the odd reflection r(-R) = -r(R)
    for (int i = 0; i < 4; ++i)
        if (j0 + i < 0)
            rr[i] = -r_iso_array[-(j0 + i)];
    return lagrange(r_iso, rr, ra, 4);
}

inline void DiracStarSolution::fill_isotropic_arrays()
{
    using namespace DiracStarSolutionHelpers;
    std::vector<double> f_phys(n_gridpoints), g_phys(n_gridpoints);
    for (int j = 0; j < n_gridpoints; j++)
    {
        const double r = j * dr;
        const double sa = exp(0.5 * state[j].phi);
        f_phys[j] = (j == 0) ? 0. : state[j].F / (sa * r);
        g_phys[j] = (j == 0) ? A_central : state[j].G / (sa * r);
    }
    std::vector<double> f_array(n_gridpoints), r_areal_array(n_gridpoints);
    f_array[0] = 1.;
    r_iso_array.assign(n_gridpoints, 0.);
    psi_iso_array.assign(n_gridpoints, 0.);
    phi_iso_array.assign(n_gridpoints, 0.);
    f_iso_array.assign(n_gridpoints, 0.);
    g_iso_array.assign(n_gridpoints, 0.);
    for (int j = 0; j < n_gridpoints - 1; j++)
    {
        const double r = j * dr;
        const double f1 = f_RHS(r, f_array[j]);
        const double f2 = f_RHS(r + dr / 2., f_array[j] + 0.5 * dr * f1);
        const double f3 = f_RHS(r + dr / 2., f_array[j] + 0.5 * dr * f2);
        const double f4 = f_RHS(r + dr, f_array[j] + dr * f3);
        f_array[j + 1] = f_array[j] + (dr / 6.) * (f1 + 2 * f2 + 2 * f3 + f4);
        r_iso_array[j + 1] = f_array[j + 1] * (r + dr);
    }
    const double iso_scale_factor = (R - M - M * M / (4. * (R - M))) / r_iso_array[n_gridpoints - 1];
    for (double &r_iso : r_iso_array)
        r_iso *= iso_scale_factor;
    for (double &f : f_array)
        f *= iso_scale_factor;
    for (int j = 0; j < n_gridpoints; j++)
    {
        r_areal_array[j] = r_areal(j);
        const int j_areal_low = static_cast<int>(floor(r_areal_array[j] / dr));
        const int j_areal_high = static_cast<int>(ceil(r_areal_array[j] / dr));
        double f;
        if (j_areal_high <= n_gridpoints - 1)
        {
            const int j0 = bound(j_areal_low - 1, 0, n_gridpoints - 4);
            const double ra = r_areal_array[j];
            phi_iso_array[j] =
                cubic(ra, state[j0].phi, state[j0 + 1].phi, state[j0 + 2].phi, state[j0 + 3].phi, j0, dr);
            f_iso_array[j] = cubic(ra, f_phys[j0], f_phys[j0 + 1], f_phys[j0 + 2], f_phys[j0 + 3], j0, dr);
            g_iso_array[j] = cubic(ra, g_phys[j0], g_phys[j0 + 1], g_phys[j0 + 2], g_phys[j0 + 3], j0, dr);
            f = cubic(ra, f_array[j0], f_array[j0 + 1], f_array[j0 + 2], f_array[j0 + 3], j0, dr);
        }
        else
        {
            const int N = n_gridpoints;
            const double dR = r_areal_array[j] - R;
            phi_iso_array[j] = state[N - 1].phi + dR * (state[N - 1].phi - state[N - 2].phi) / dr;
            f_iso_array[j] = f_phys[N - 1] + dR * (f_phys[N - 1] - f_phys[N - 2]) / dr;
            g_iso_array[j] = g_phys[N - 1] + dR * (g_phys[N - 1] - g_phys[N - 2]) / dr;
            f = f_array[N - 1] + dR * (f_array[N - 1] - f_array[N - 2]) / dr;
        }
        // blend to the asymptotic Schwarzschild ratio near the edge, as in the 1D code
        const double l_frac = 0.8, u_frac = 0.9;
        if (j > n_gridpoints * l_frac)
        {
            const double portion = std::min((j - n_gridpoints * l_frac) / (n_gridpoints * (u_frac - l_frac)), 1.);
            f = f * (1 - portion) + pow(1 + M / (2. * dr * j), -2.) * portion;
        }
        psi_iso_array[j] = 1. / sqrt(f);
    }
}

inline bool DiracStarSolution::solve(const params_t &a_params, std::ostream &os, bool quiet)
{
    m_p = a_params;
    mu = m_p.mu;
    Gn = m_p.G_Newton;
    A_central = m_p.A_central;
    R = m_p.R;
    n_gridpoints = m_p.n_gridpoints;
    dr = R / (n_gridpoints - 1);

    find_frequency(os, quiet);
    if (!fill_asymptotic(os, quiet))
        return false;
    if (m_p.perturb)
        add_perturbation(m_p.perturb_amp, m_p.perturb_spread, m_p.perturb_center, os, quiet);
    omega = static_cast<double>(omega_ld);
    fill_isotropic_arrays();
    return true;
}

inline double DiracStarSolution::interp_iso(const std::vector<double> &y, double Riso, bool odd,
                                            bool derivative) const
{
    using namespace DiracStarSolutionHelpers;
    const int j_low = static_cast<int>(floor(Riso / dr));
    const int j0 = bound(j_low - 1, -2, n_gridpoints - 4);
    double X[4], Y[4];
    for (int i = 0; i < 4; ++i)
    {
        const int j = j0 + i;
        X[i] = j * dr;
        Y[i] = (j >= 0) ? y[j] : (odd ? -y[-j] : y[-j]);
    }
    if (!derivative)
        return lagrange(Riso, X, Y, 4);
    // derivative of the cubic Lagrange interpolant
    double out = 0.;
    for (int i = 0; i < 4; ++i)
    {
        double dli = 0.;
        for (int m = 0; m < 4; ++m)
        {
            if (m == i)
                continue;
            double term = 1. / (X[i] - X[m]);
            for (int j = 0; j < 4; ++j)
                if (j != i && j != m)
                    term *= (Riso - X[j]) / (X[i] - X[j]);
            dli += term;
        }
        out += dli * Y[i];
    }
    return out;
}

// Beyond 0.8 R (where the 1D code starts blending the isotropic arrays towards Schwarzschild) the metric is isotropic
// Schwarzschild and the spinor is continued as exp(-sqrt(mu^2 - omega^2) R), so that any R works for the 2D domain
inline double DiracStarSolution::get_g(double Riso) const
{
    const double Rb = 0.8 * R;
    if (Riso < Rb)
        return interp_iso(g_iso_array, Riso, false);
    return interp_iso(g_iso_array, Rb, false) * exp(-sqrt(mu * mu - omega * omega) * (Riso - Rb));
}
inline double DiracStarSolution::get_f(double Riso) const
{
    const double Rb = 0.8 * R;
    if (Riso < Rb)
        return interp_iso(f_iso_array, Riso, true);
    return interp_iso(f_iso_array, Rb, true) * exp(-sqrt(mu * mu - omega * omega) * (Riso - Rb));
}
inline double DiracStarSolution::get_lapse(double Riso) const
{
    if (Riso < 0.8 * R)
        return exp(interp_iso(phi_iso_array, Riso, false));
    const double h = M / (2. * Riso);
    return (1. - h) / (1. + h);
}
inline double DiracStarSolution::get_psi(double Riso) const
{
    if (Riso < 0.8 * R)
        return interp_iso(psi_iso_array, Riso, false);
    return 1. + M / (2. * Riso);
}
inline double DiracStarSolution::get_dlapse(double Riso) const
{
    if (Riso < 0.8 * R)
        return exp(interp_iso(phi_iso_array, Riso, false)) * interp_iso(phi_iso_array, Riso, false, true);
    const double h = M / (2. * Riso);
    return M / (Riso * Riso * (1. + h) * (1. + h));
}
inline double DiracStarSolution::get_dpsi(double Riso) const
{
    if (Riso < 0.8 * R)
        return interp_iso(psi_iso_array, Riso, false, true);
    return -M / (2. * Riso * Riso);
}

inline void DiracStarSolution::write_profiles(const std::string &filename) const
{
    std::ofstream out(filename);
    out << "# R_iso psi lapse f g   (isotropic)   |   r_areal X phi F G   (areal, same index)\n";
    for (int j = 0; j < n_gridpoints; ++j)
        out << std::setprecision(12) << j * dr << " " << psi_iso_array[j] << " " << exp(phi_iso_array[j]) << " "
            << f_iso_array[j] << " " << g_iso_array[j] << "   " << radius_array[j] << " " << state[j].X << " "
            << state[j].phi << " " << state[j].F << " " << state[j].G << "\n";
}

#endif /* DIRACSTARSOLUTION_IMPL_HPP_ */
