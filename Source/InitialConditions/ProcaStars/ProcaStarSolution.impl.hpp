/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(PROCASTARSOLUTION_HPP_)
#error "This file should only be included through ProcaStarSolution.hpp"
#endif

#ifndef PROCASTARSOLUTION_IMPL_HPP_
#define PROCASTARSOLUTION_IMPL_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>

namespace ProcaStarSolutionHelpers
{
//! true if x is NaN or +-infinity. This tests the bit pattern rather than using std::isnan / std::isinf, which compilers
//! fold to false when they may assume finite arithmetic (-ffast-math, -ffinite-math-only, and the default
//! -fp-model=fast of Intel's icx/icpx). The shooting relies on detecting the blowup of the trial solutions, so with
//! std::isnan the frequency search fails in such builds.
inline bool not_finite(double x)
{
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL;
}

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

inline ProcaFieldState axpy(const ProcaFieldState &s, double h, const ProcaFieldState &k)
{
    return ProcaFieldState{s.f + h * k.f, s.X + h * k.X, s.phi + h * k.phi, s.E + h * k.E};
}
inline ProcaFieldState rk4_combine(const ProcaFieldState &s, double h, const ProcaFieldState &k1,
                                   const ProcaFieldState &k2, const ProcaFieldState &k3, const ProcaFieldState &k4)
{
    return ProcaFieldState{s.f + h / 6. * (k1.f + 2. * k2.f + 2. * k3.f + k4.f),
                           s.X + h / 6. * (k1.X + 2. * k2.X + 2. * k3.X + k4.X),
                           s.phi + h / 6. * (k1.phi + 2. * k2.phi + 2. * k3.phi + k4.phi),
                           s.E + h / 6. * (k1.E + 2. * k2.E + 2. * k3.E + k4.E)};
}
} // namespace ProcaStarSolutionHelpers

inline double ProcaStarSolution::V(double Psi) const
{
    if (!m_p.solitonic)
        return mu * mu * Psi + m_p.lambda * Psi * Psi;
    return mu * mu * Psi * pow(1. - 2. * Psi / (m_p.sigma * m_p.sigma), 2) + m_p.lambda * Psi * Psi;
}

inline double ProcaStarSolution::dV(double Psi) const
{
    if (!m_p.solitonic)
        return mu * mu + 2. * Psi * m_p.lambda;
    const double s2 = m_p.sigma * m_p.sigma;
    return mu * mu - 8. * mu * mu * Psi / s2 + 12. * mu * mu * Psi * Psi / (s2 * s2) + 2. * Psi * m_p.lambda;
}

inline double ProcaStarSolution::ddV(double Psi) const
{
    if (!m_p.solitonic)
        return 2. * m_p.lambda;
    const double s2 = m_p.sigma * m_p.sigma;
    return -8. * mu * mu / s2 + 24. * mu * mu * Psi / (s2 * s2) + 2. * m_p.lambda;
}

// V'(Psi) g = -omega E / alpha^2 with Psi = -f^2/alpha^2 + g^2/X^2; also returns W = V'(Psi) and Psi
inline double ProcaStarSolution::solve_g(long double frequency, const ProcaFieldState &s, double &W, double &Psi)
{
    const double alpha2 = exp(2. * s.phi);
    const double X2 = s.X * s.X;
    const double src = static_cast<double>(-frequency * s.E / alpha2);
    const double Psi_t = -s.f * s.f / alpha2;
    if (!self_interacting())
    {
        W = mu * mu;
        const double g = src / W;
        Psi = Psi_t + g * g / X2;
        return g;
    }
    if (src == 0.)
    {
        Psi = Psi_t;
        W = dV(Psi);
        return 0.;
    }
    const double W_t = dV(Psi_t);
    double g = (W_t != 0.) ? src / W_t : src / (mu * mu);
    for (int it = 0; it < 50; it++)
    {
        Psi = Psi_t + g * g / X2;
        W = dV(Psi);
        const double hh = W * g - src;
        const double dh = W + 2. * g * g * ddV(Psi) / X2;
        if (dh == 0. || ProcaStarSolutionHelpers::not_finite(dh) || ProcaStarSolutionHelpers::not_finite(hh))
        {
            g_solve_failed = true;
            return 0.; // failure is signalled by g_solve_failed
        }
        const double step = hh / dh;
        g -= step;
        if (std::abs(step) <= 1e-14 * std::abs(g))
        {
            Psi = Psi_t + g * g / X2;
            W = dV(Psi);
            return g;
        }
    }
    g_solve_failed = true;
    return 0.; // no convergence: failure is signalled by g_solve_failed
}

// X', phi' (Hamiltonian constraint and polar slicing) and E' (Gauss law) for given f and g, D = 4
inline void ProcaStarSolution::matter(double radius, double X, double phi, double E, double f, double g, double &dX,
                                      double &dPhi, double &dE) const
{
    const double r = (radius == 0.) ? 0.0000001 : radius;
    const double alpha2 = exp(2. * phi);
    const double X2 = X * X;
    const double Psi = -f * f / alpha2 + g * g / X2;
    const double W = dV(Psi);
    const double T1 = (radius == 0.) ? 0. : 0.5 * (X2 - 1.) / r;
    const double T3 = (radius == 0.) ? 0. : E / r;
    const double F1 = 2. * M_PI * Gn * r * X2;
    const double E_term = E * E / (alpha2 * X2);
    const double two_rho = E_term + 2. * W * f * f / alpha2 + V(Psi);
    const double two_p_r = -E_term + 2. * W * g * g / X2 - V(Psi);
    dX = X * (F1 * two_rho - T1);
    dPhi = T1 + F1 * two_p_r;
    const double E_corr = (radius == 0.) ? (1. / 3.) : 1.;
    dE = E_corr * (W * f * X2 - 2. * T3 + E * (dPhi + dX / X));
}

inline ProcaFieldState ProcaStarSolution::state_RHS(double r, long double frequency, const ProcaFieldState &s,
                                                    bool asymptotic)
{
    double W, Psi;
    const double g = solve_g(frequency, s, W, Psi);
    double dX, dPhi, dE;
    matter(r, s.X, s.phi, s.E, s.f, g, dX, dPhi, dE);
    if (asymptotic)
        return ProcaFieldState{0., dX, dPhi, 0.};
    return ProcaFieldState{static_cast<double>(s.E + frequency * g), dX, dPhi, dE};
}

inline void ProcaStarSolution::rk4_solve(long double freq)
{
    using namespace ProcaStarSolutionHelpers;
    state.assign(n_gridpoints, ProcaFieldState{});
    radius_array.assign(n_gridpoints, 0.);
    state[0] = ProcaFieldState{A_central * m_p.alpha_central, 1., log(m_p.alpha_central), 0.};
    blowup_point = n_gridpoints;
    g_solve_failed = false;
    g_breakdown_r = -1.;
    for (int j = 0; j < n_gridpoints - 1; j++)
    {
        const double r = j * dr;
        const ProcaFieldState s1 = state_RHS(r, freq, state[j], false);
        const ProcaFieldState s2 = state_RHS(r + dr / 2., freq, axpy(state[j], 0.5 * dr, s1), false);
        const ProcaFieldState s3 = state_RHS(r + dr / 2., freq, axpy(state[j], 0.5 * dr, s2), false);
        const ProcaFieldState s4 = state_RHS(r + dr, freq, axpy(state[j], dr, s3), false);
        state[j + 1] = rk4_combine(state[j], dr, s1, s2, s3, s4);
        radius_array[j + 1] = (j + 1) * dr;
        if (not_finite(state[j].f) || not_finite(state[j].X) || not_finite(state[j].phi) || not_finite(state[j].E) ||
            g_solve_failed)
        {
            blowup_point = j;
            if (g_solve_failed)
                g_breakdown_r = r;
            return;
        }
    }
}

inline double ProcaStarSolution::m(int j) const
{
    return 0.5 * radius_array[j] * (1. - 1. / (state[j].X * state[j].X));
}

inline void ProcaStarSolution::fill_g_array()
{
    g_array.resize(state.size());
    double W, Psi;
    for (size_t j = 0; j < state.size(); j++)
        g_array[j] = solve_g(omega_ld, state[j], W, Psi);
}

inline int ProcaStarSolution::count_zero_crossings() const
{
    int zc = 0;
    for (int j = 1; j < blowup_point - 1; j++)
        if ((state[j].E == 0.) || (state[j].E > 0. && state[j - 1].E < 0.) || (state[j].E < 0. && state[j - 1].E > 0.))
            zc++;
    return zc;
}

inline long double ProcaStarSolution::find_frequency(std::ostream &os, bool quiet)
{
    const int eigen = m_p.eigen;
    const long double epsilon = m_p.freq_epsilon;
    long double lower_guess = m_p.frequency_guess;

    rk4_solve(lower_guess);
    // bracket upwards first if the guess is too small (the 1D code only warns here)
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
        os << "ProcaStarSolution ERROR: could not find a lower frequency bound for A_central = " << A_central
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
        os << "ProcaStarSolution WARNING: zero crossing count may be incorrect for A_central = " << A_central
           << ", found " << count_zero_crossings() << std::endl;
    if (!quiet)
        os << "ProcaStarSolution: found eigenstate " << eigen << " with unrescaled frequency "
           << static_cast<double>(omega_ld) << std::endl;
    return lower_guess;
}

inline int ProcaStarSolution::find_last_minimum() const
{
    double E_max = 0.;
    for (int j = 0; j < blowup_point - 1 && j < n_gridpoints; j++)
        E_max = std::max(E_max, std::abs(state[j].E));
    for (int j = blowup_point - 2; j > 1; j--)
        if (std::abs(state[j].E) <= std::abs(state[j - 1].E) && std::abs(state[j].E) <= std::abs(state[j + 1].E) &&
            std::abs(state[j].E) < 0.05 * E_max)
            return j;
    if (blowup_point == n_gridpoints &&
        std::abs(state[n_gridpoints - 1].E) <= std::abs(state[n_gridpoints - 2].E))
        return n_gridpoints - 1;
    return -1;
}

inline void ProcaStarSolution::rescale_lapse(double phi_shift)
{
    const double fac = exp(phi_shift);
    for (ProcaFieldState &s : state)
    {
        s.phi += phi_shift;
        s.f *= fac;
        s.E *= fac;
    }
    omega_ld *= fac;
}

// N = 4 pi int r^2 j^t alpha X dr, j^t = -g E / (alpha^2 X^2) (as the 1D code)
inline double ProcaStarSolution::get_noether_charge() const
{
    double Q = 0.;
    for (int j = 0; j < n_gridpoints; j++)
    {
        const double r = j * dr;
        Q += -r * r * dr * g_array[j] * state[j].E * exp(-state[j].phi) / state[j].X;
    }
    return 4. * M_PI * Q;
}

inline void ProcaStarSolution::compute_r99()
{
    int j = 0;
    while (j < n_gridpoints - 1 && m(j) < 0.99 * M)
    {
        if (m(j + 1) != m(j))
            r_99 = dr * j + dr * (0.99 * M - m(j)) / (m(j + 1) - m(j));
        j++;
    }
}

inline bool ProcaStarSolution::fill_asymptotic(std::ostream &os, bool quiet)
{
    using namespace ProcaStarSolutionHelpers;
    const int min_index = find_last_minimum();
    if (min_index == -1)
    {
        os << "ProcaStarSolution ERROR: no local minimum in |E| found for A_central = " << A_central << std::endl;
        return false;
    }
    const int j_match = static_cast<int>(round(m_p.r_match_fac * min_index));
    const double r_match = dr * j_match;
    const double phi_match = state[j_match].phi;
    const double E_match = state[j_match].E;
    // far from the star f ~ exp(-k r)/r, and E/f = -(mu^2/k^2)(1/r + k)
    const double k2 = mu * mu - static_cast<double>(pow(omega_ld / exp(phi_match), 2));
    const double k = (k2 > 0.) ? sqrt(k2) : 0.;
    if (!(k2 > 0.) || not_finite(k))
    {
        os << "ProcaStarSolution ERROR: omega / alpha exceeds mu at the matching point for A_central = "
           << A_central << std::endl;
        return false;
    }
    auto E_over_f = [&](double r) { return -(mu * mu / (k * k)) * (1. / r + k); };
    const double f_factor = (E_match / E_over_f(r_match)) * exp(k * r_match) * r_match;
    for (int j = j_match; j < n_gridpoints - 1; j++)
    {
        const double r = j * dr;
        const ProcaFieldState s1 = state_RHS(r, omega_ld, state[j], true);
        const ProcaFieldState s2 = state_RHS(r + dr / 2., omega_ld, axpy(state[j], 0.5 * dr, s1), true);
        const ProcaFieldState s3 = state_RHS(r + dr / 2., omega_ld, axpy(state[j], 0.5 * dr, s2), true);
        const ProcaFieldState s4 = state_RHS(r + dr, omega_ld, axpy(state[j], dr, s3), true);
        state[j + 1] = rk4_combine(state[j], dr, s1, s2, s3, s4);
        radius_array[j + 1] = (j + 1) * dr;
        const double r1 = radius_array[j + 1];
        state[j + 1].f = f_factor * exp(-k * r1) / r1;
        state[j + 1].E = E_over_f(r1) * state[j + 1].f;
        if (not_finite(state[j].f) || not_finite(state[j].X) || not_finite(state[j].phi) || not_finite(state[j].E))
        {
            os << "ProcaStarSolution WARNING: nan in the asymptotic region for A_central = " << A_central
               << std::endl;
            return false;
        }
    }
    const double phi_shift = -log(state[n_gridpoints - 1].X) - state[n_gridpoints - 1].phi;
    omega_pre_rescale = omega_ld;
    rescale_lapse(phi_shift);
    fill_g_array();
    M = m(n_gridpoints - 1);
    compute_r99();
    noether_charge = get_noether_charge();
    binding_energy = M - mu * noether_charge;
    if (!quiet)
        os << "ProcaStarSolution: A_central = " << A_central << ", omega = " << static_cast<double>(omega_ld)
           << ", M = " << M << ", N = " << noether_charge << ", binding energy = " << binding_energy
           << ", r_99 = " << r_99 << std::endl;
    return true;
}

inline void ProcaStarSolution::add_perturbation(double a, double k, double center, std::ostream &os, bool quiet)
{
    using namespace ProcaStarSolutionHelpers;
    const int N = n_gridpoints;
    std::vector<double> f_p(N), g_p(N);
    for (int j = 0; j < N; j++)
    {
        const double r = j * dr;
        const double fac = 1. + a * exp(-pow(r - center, 2.) / (k * k));
        f_p[j] = fac * state[j].f;
        g_p[j] = fac * g_array[j];
    }
    auto interp = [&](const std::vector<double> &y, double r) {
        const int j0 = bound(static_cast<int>(floor(r / dr)) - 1, 0, N - 4);
        return cubic(r, y[j0], y[j0 + 1], y[j0 + 2], y[j0 + 3], j0, dr);
    };
    state[0].E = 0.;
    for (int j = 0; j < N - 1; j++)
    {
        const double r = j * dr;
        const double X = state[j].X, phi = state[j].phi, E = state[j].E;
        const double fm = interp(f_p, r + 0.5 * dr), gm = interp(g_p, r + 0.5 * dr);
        double k1X, k1p, k1E, k2X, k2p, k2E, k3X, k3p, k3E, k4X, k4p, k4E;
        matter(r, X, phi, E, f_p[j], g_p[j], k1X, k1p, k1E);
        matter(r + 0.5 * dr, X + 0.5 * dr * k1X, phi + 0.5 * dr * k1p, E + 0.5 * dr * k1E, fm, gm, k2X, k2p, k2E);
        matter(r + 0.5 * dr, X + 0.5 * dr * k2X, phi + 0.5 * dr * k2p, E + 0.5 * dr * k2E, fm, gm, k3X, k3p, k3E);
        matter(r + dr, X + dr * k3X, phi + dr * k3p, E + dr * k3E, f_p[j + 1], g_p[j + 1], k4X, k4p, k4E);
        state[j + 1].X = X + (dr / 6.) * (k1X + 2. * k2X + 2. * k3X + k4X);
        state[j + 1].phi = phi + (dr / 6.) * (k1p + 2. * k2p + 2. * k3p + k4p);
        state[j + 1].E = E + (dr / 6.) * (k1E + 2. * k2E + 2. * k3E + k4E);
        if (not_finite(state[j + 1].X) || not_finite(state[j + 1].phi) || not_finite(state[j + 1].E))
        {
            os << "ProcaStarSolution ERROR: constraint solve for the perturbed star failed at r = " << r + dr
               << std::endl;
            return;
        }
    }
    for (int j = 0; j < N; j++)
    {
        state[j].f = f_p[j];
        g_array[j] = g_p[j];
    }
    rescale_lapse(-log(state[N - 1].X) - state[N - 1].phi);
    A_central = state[0].f * exp(-state[0].phi);
    const double M_old = M, N_old = noether_charge;
    M = m(N - 1);
    noether_charge = get_noether_charge();
    binding_energy = M - mu * noether_charge;
    compute_r99();
    if (!quiet)
        os << "ProcaStarSolution: perturbed with amplitude " << a << ", width " << k << ", centre " << center
           << ": M = " << M_old << " -> " << M << ", N = " << N_old << " -> " << noether_charge
           << ", binding energy = " << binding_energy << std::endl;
}

inline double ProcaStarSolution::f_RHS(double r, double f) const
{
    using namespace ProcaStarSolutionHelpers;
    const int j_low = static_cast<int>(floor((r / R) * (n_gridpoints - 1)));
    const int j0 = bound(j_low - 1, 0, n_gridpoints - 4);
    const double X = cubic(r, state[j0].X, state[j0 + 1].X, state[j0 + 2].X, state[j0 + 3].X, j0, dr);
    const double r2 = (r == 0.) ? 10e-10 : r;
    return f * (X - 1.) / r2;
}

inline double ProcaStarSolution::r_areal(int j_iso) const
{
    using namespace ProcaStarSolutionHelpers;
    if (j_iso == 0)
        return 0.;
    const double r_iso = dr * j_iso;
    const int j_areal =
        static_cast<int>(std::lower_bound(r_iso_array.begin(), r_iso_array.end(), r_iso) - r_iso_array.begin());
    if (j_areal == n_gridpoints)
        return r_iso + M + M * M / (4. * r_iso);
    const int j0 = bound(j_areal - 2, -2, n_gridpoints - 4);
    double rr[4], ra[4];
    for (int i = 0; i < 4; ++i)
    {
        rr[i] = (j0 + i >= 0) ? r_iso_array[j0 + i] : -r_iso_array[-(j0 + i)];
        ra[i] = (j0 + i) * dr;
    }
    return lagrange(r_iso, rr, ra, 4);
}

inline void ProcaStarSolution::fill_isotropic_arrays()
{
    using namespace ProcaStarSolutionHelpers;
    std::vector<double> f_array(n_gridpoints), r_areal_array(n_gridpoints);
    f_array[0] = 1.;
    r_iso_array.assign(n_gridpoints, 0.);
    psi_iso_array.assign(n_gridpoints, 0.);
    phi_iso_array.assign(n_gridpoints, 0.);
    f_iso_array.assign(n_gridpoints, 0.);
    g_iso_array.assign(n_gridpoints, 0.);
    E_iso_array.assign(n_gridpoints, 0.);
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
        double f, X_val, g_val, E_val;
        if (j_areal_high <= n_gridpoints - 1)
        {
            const int j0 = bound(j_areal_low - 1, 0, n_gridpoints - 4);
            const double ra = r_areal_array[j];
            auto c4 = [&](auto get) { return cubic(ra, get(j0), get(j0 + 1), get(j0 + 2), get(j0 + 3), j0, dr); };
            phi_iso_array[j] = c4([&](int i) { return state[i].phi; });
            f_iso_array[j] = c4([&](int i) { return state[i].f; });
            g_val = c4([&](int i) { return g_array[i]; });
            E_val = c4([&](int i) { return state[i].E; });
            X_val = c4([&](int i) { return state[i].X; });
            f = c4([&](int i) { return f_array[i]; });
        }
        else
        {
            const int N = n_gridpoints;
            const double dR = r_areal_array[j] - R;
            phi_iso_array[j] = state[N - 1].phi + dR * (state[N - 1].phi - state[N - 2].phi) / dr;
            f_iso_array[j] = state[N - 1].f + dR * (state[N - 1].f - state[N - 2].f) / dr;
            g_val = g_array[N - 1] + dR * (g_array[N - 1] - g_array[N - 2]) / dr;
            E_val = state[N - 1].E + dR * (state[N - 1].E - state[N - 2].E) / dr;
            X_val = state[N - 1].X + dR * (state[N - 1].X - state[N - 2].X) / dr;
            f = f_array[N - 1] + dR * (f_array[N - 1] - f_array[N - 2]) / dr;
        }
        const double l_frac = 0.8, u_frac = 0.9;
        if (j > n_gridpoints * l_frac)
        {
            const double portion = std::min((j - n_gridpoints * l_frac) / (n_gridpoints * (u_frac - l_frac)), 1.);
            f = f * (1 - portion) + pow(1 + M / (2. * dr * j), -2.) * portion;
        }
        psi_iso_array[j] = 1. / sqrt(f);
        g_iso_array[j] = g_val * psi_iso_array[j] * psi_iso_array[j] / X_val;
        E_iso_array[j] = E_val * psi_iso_array[j] * psi_iso_array[j] / X_val;
    }
}

inline bool ProcaStarSolution::solve(const params_t &a_params, std::ostream &os, bool quiet)
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
    {
        if (g_breakdown_r >= 0.)
            os << "ProcaStarSolution WARNING: the algebraic equation V'(Psi) g = -omega E / alpha^2 has no regular "
                  "root beyond r = "
               << g_breakdown_r << " (self-interaction breakdown)" << std::endl;
        return false;
    }
    if (m_p.perturb)
        add_perturbation(m_p.perturb_amp, m_p.perturb_spread, m_p.perturb_center, os, quiet);
    omega = static_cast<double>(omega_ld);
    f0 = state[0].f;
    fill_isotropic_arrays();
    return true;
}

inline double ProcaStarSolution::interp_iso(const std::vector<double> &y, double Riso, bool odd,
                                            bool derivative) const
{
    using namespace ProcaStarSolutionHelpers;
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
// Schwarzschild and the field is continued as exp(-sqrt(mu^2 - omega^2) R) / R, so that any R works for the grid
inline double ProcaStarSolution::tail(const std::vector<double> &y, double Riso, bool odd) const
{
    const double Rb = 0.8 * R;
    if (Riso < Rb)
        return interp_iso(y, Riso, odd);
    return interp_iso(y, Rb, odd) * (Rb / Riso) * exp(-sqrt(mu * mu - omega * omega) * (Riso - Rb));
}
inline double ProcaStarSolution::get_f(double Riso) const { return tail(f_iso_array, Riso, false); }
inline double ProcaStarSolution::get_g(double Riso) const { return tail(g_iso_array, Riso, true); }
inline double ProcaStarSolution::get_E(double Riso) const { return tail(E_iso_array, Riso, true); }
inline double ProcaStarSolution::get_lapse(double Riso) const
{
    if (Riso < 0.8 * R)
        return exp(interp_iso(phi_iso_array, Riso, false));
    const double hM = M / (2. * Riso);
    return (1. - hM) / (1. + hM);
}
inline double ProcaStarSolution::get_psi(double Riso) const
{
    if (Riso < 0.8 * R)
        return interp_iso(psi_iso_array, Riso, false);
    return 1. + M / (2. * Riso);
}
inline double ProcaStarSolution::get_dlapse(double Riso) const
{
    if (Riso < 0.8 * R)
        return exp(interp_iso(phi_iso_array, Riso, false)) * interp_iso(phi_iso_array, Riso, false, true);
    const double hM = M / (2. * Riso);
    return M / (Riso * Riso * (1. + hM) * (1. + hM));
}
inline double ProcaStarSolution::get_dpsi(double Riso) const
{
    if (Riso < 0.8 * R)
        return interp_iso(psi_iso_array, Riso, false, true);
    return -M / (2. * Riso * Riso);
}

inline void ProcaStarSolution::write_profiles(const std::string &filename) const
{
    std::ofstream out(filename);
    out << "# R_iso psi lapse f_iso g_iso E_iso   (isotropic)   |   r_areal X phi f g E   (areal, same index)\n";
    for (int j = 0; j < n_gridpoints; ++j)
        out << std::setprecision(12) << j * dr << " " << psi_iso_array[j] << " " << exp(phi_iso_array[j]) << " "
            << f_iso_array[j] << " " << g_iso_array[j] << " " << E_iso_array[j] << "   " << radius_array[j] << " "
            << state[j].X << " " << state[j].phi << " " << state[j].f << " " << g_array[j] << " " << state[j].E
            << "\n";
}

#endif /* PROCASTARSOLUTION_IMPL_HPP_ */
