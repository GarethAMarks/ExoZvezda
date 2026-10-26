/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIRACSTARSOLUTION_HPP_
#define DIRACSTARSOLUTION_HPP_

#include <iostream>
#include <string>
#include <vector>

/*
 Spherically symmetric Dirac stars (Einstein-Dirac solitons): a port of the DiracStar solver of Spherical_BS_Evolver
 (DiracStar.h/.cpp, validated there: M_max = 0.70891 at omega = 0.8246, omega_min = 0.7328, first law to 1e-5), with
 the same conventions and parameters, signature (-,+,+,+). It has no Chombo dependence so that it can be tested alone.

   metric:  ds^2 = -alpha^2 dt^2 + X^2 dr^2 + r^2 dOmega^2,   alpha = exp(phi), r areal
   matter:  two Dirac spinors of mass mu in the j = 1/2 states m_j = +-1/2,
            psi_m = e^{-i omega t} alpha^{-1/2} r^{-1} ( G Omega_{-1,m}, i F Omega_{+1,m} ),
   field equations:
       G' = X [  G/r + (mu + omega/alpha) F ],      F' = X [ -F/r + (mu - omega/alpha) G ],
       X' = X (1 - X^2)/(2r) + 4 pi G r X^3 rho,    phi' = (X^2 - 1)/(2r) + 4 pi G r X^2 p_r,
       n = (F^2 + G^2)/(2 pi alpha r^2), rho = (omega/alpha) n, p_r = [(omega/alpha)(F^2+G^2) - mu(G^2-F^2) + 2FG/r]/(2 pi alpha r^2).
 The central amplitude A_central = g(0), where g = G/(sqrt(alpha) r) and f = F/(sqrt(alpha) r) are the physical radial
 spinor functions (orthonormal-frame components, unchanged by r -> R). The frequency is found by shooting on the number
 of nodes of G (eigen = 0 for the ground state), the tail is continued as a decaying exponential, alpha(infinity) = 1 is
 imposed by rescaling, and the solution is transformed to isotropic coordinates, gamma_ij = psi^4 delta_ij.

 Optional perturbation (as in Spherical_BS_Evolver): (f, g) -> (1 + a exp(-(r - r0)^2 / w^2)) (f, g), then X and phi
 re-solved from the Hamiltonian constraint and the polar slicing condition with the 3+1 (non-stationary) sources.
 a > 0 makes the star denser (collapse on the unstable branch), a < 0 more dilute (migration).

 Differences from the 1D code: the frequency search also brackets upwards if frequency_guess is too small, and the
 areal radius lookup in the isotropic transformation uses a binary search (the grids here are larger).
*/

struct DiracFieldState
{
    double F{};   //!< lower radial spinor function (includes sqrt(alpha) r), vanishes as r^2 at the origin
    double G{};   //!< upper radial spinor function (includes sqrt(alpha) r), vanishes as r at the origin
    double X{};   //!< sqrt(g_rr)
    double phi{}; //!< log(lapse)
};

class DiracStarSolution
{
  public:
    struct params_t
    {
        double mu = 1.;                //!< fermion mass
        double G_Newton = 1.;          //!< Newton's constant
        double A_central = 0.1;        //!< central amplitude g(0)
        int eigen = 0;                 //!< number of nodes of G (0 = ground state)
        double alpha_central = 0.95;   //!< initial central lapse guess (rescaled afterwards)
        double frequency_guess = 1.8;  //!< initial frequency guess (the search brackets up or down from here)
        double freq_epsilon = 1e-19;   //!< tolerance for the frequency bisection
        double r_match_fac = 0.85;     //!< matching radius as a fraction of the last minimum of |G|
        double R = 100.;               //!< outer radius of the 1D solution (areal; isotropic arrays cover the same index range)
        int n_gridpoints = 20001;      //!< 1D gridpoints
        bool perturb = false;          //!< apply the Gaussian perturbation
        double perturb_amp = 0.;       //!< a
        double perturb_spread = 1.;    //!< w
        double perturb_center = 0.;    //!< r0
    };

    DiracStarSolution() = default;

    //! solve for the star (and perturb it if requested) and build the isotropic arrays; returns true on success
    bool solve(const params_t &a_params, std::ostream &os = std::cout, bool quiet = false);

    // isotropic-coordinate profiles at isotropic radius R (beyond the 1D grid: isotropic Schwarzschild, no spinor)
    double get_g(double R) const;     //!< physical upper spinor function g
    double get_f(double R) const;     //!< physical lower spinor function f (odd, f ~ R near the centre)
    double get_lapse(double R) const; //!< alpha
    double get_psi(double R) const;   //!< conformal factor, gamma_ij = psi^4 delta_ij
    double get_dlapse(double R) const; //!< d alpha / dR
    double get_dpsi(double R) const;   //!< d psi / dR

    double omega = 0.;          //!< frequency (with alpha(infinity) = 1)
    double M = 0.;              //!< ADM mass
    double noether_charge = 0.; //!< particle number of both spinors
    double binding_energy = 0.; //!< M - mu N
    double r_99 = 0.;           //!< areal radius containing 99% of the mass
    double A_central = 0.;      //!< central amplitude g(0) (after any perturbation)

    //! areal-coordinate solution and isotropic arrays, for tests
    void write_profiles(const std::string &filename) const;
    const std::vector<DiracFieldState> &areal_state() const { return state; }

  private:
    params_t m_p;
    double mu, Gn;
    long double omega_ld = 0., omega_pre_rescale = 0.;
    int n_gridpoints = 0, blowup_point = 0;
    double R = 0., dr = 0.;

    std::vector<DiracFieldState> state;
    std::vector<double> radius_array;
    std::vector<double> r_iso_array, psi_iso_array, phi_iso_array, f_iso_array, g_iso_array;

    void matter_terms(double r, long double frequency, const DiracFieldState &s, double &rho, double &p_r,
                      double &n) const;
    DiracFieldState state_RHS(double r, long double frequency, const DiracFieldState &s, bool asymptotic) const;
    void rk4_solve(long double freq);
    int count_zero_crossings() const;
    long double find_frequency(std::ostream &os, bool quiet);
    int find_last_minimum() const;
    bool fill_asymptotic(std::ostream &os, bool quiet);
    void rescale_lapse(double phi_shift);
    double m(int j) const;
    double get_noether_charge() const;
    void compute_r99();
    double f_RHS(double r, double f) const;
    double r_areal(int j_iso) const;
    void fill_isotropic_arrays();
    void add_perturbation(double a, double k, double center, std::ostream &os, bool quiet);
    double interp_iso(const std::vector<double> &y, double Riso, bool odd, bool derivative = false) const;
};

#include "DiracStarSolution.impl.hpp"

#endif /* DIRACSTARSOLUTION_HPP_ */
