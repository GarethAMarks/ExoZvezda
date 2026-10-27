/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef PROCASTARSOLUTION_HPP_
#define PROCASTARSOLUTION_HPP_

#include <iostream>
#include <string>
#include <vector>

/*
 Spherically symmetric Proca stars: a port of the ProcaStar solver of Spherical_BS_Evolver (ProcaStar.h/.cpp, branch
 proca-stars), with the same conventions and parameters, in D = 4, signature (-,+,+,+). It has no Chombo dependence so
 that it can be tested alone, and it is shared by 2DChombo and ExoZvezda.

   metric:  ds^2 = -alpha^2 dt^2 + X^2 dr^2 + r^2 dOmega^2,   alpha = exp(phi), r areal
   action:  S = int sqrt(-g) [ R / (16 pi G) - 1/4 F_ab Fbar^ab - 1/2 V(Psi) ],   Psi = A_a Abar^a
   field:   A = e^{-i omega t} ( f(r) dt + i g(r) dr ),   E := f' - omega g   (F_tr = -E e^{-i omega t})
   V = mu^2 Psi + lambda Psi^2 (mini/quartic) or mu^2 Psi (1 - 2 Psi/sigma^2)^2 + lambda Psi^2 (solitonic).
 The radial field equation V'(Psi) g = -omega E / alpha^2 is algebraic (solved by Newton iteration with
 self-interactions). The frequency is found by shooting on the number of nodes of E (eigen = 0: nodeless g and E, f with
 one node), the tail is matched to the decaying solution, alpha(infinity) = 1 is imposed by rescaling (alpha, omega, f, E
 -> k alpha, k omega, k f, k E), and the solution is transformed to isotropic coordinates, gamma_ij = psi^4 delta_ij,
 where f is unchanged and the radial components become A_R = g dr/dR and F_{Rt} = E dr/dR (dr/dR = psi^2 / X).
 A_central = f(0)/alpha(0) = sqrt(-Psi(0)), which equals |X_phi| at the centre (X_phi = -n^a A_a).

 Optional perturbation (as in Spherical_BS_Evolver): f, g -> (1 + a exp(-(r - r0)^2 / w^2)) (f, g), then X, phi and E
 re-solved from the Hamiltonian constraint, the polar slicing condition and the Gauss law, so both constraints hold.

 Differences from the 1D code: D = 4 only; the frequency search also brackets upwards if frequency_guess is too small;
 the areal radius lookup in the isotropic transformation uses a binary search; beyond 0.8 R the metric is isotropic
 Schwarzschild and f, g, E are continued as exp(-sqrt(mu^2 - omega^2) R) / R.
*/

struct ProcaFieldState
{
    double f{};   //!< A_t (up to the phase)
    double X{};   //!< sqrt(g_rr)
    double phi{}; //!< log(lapse)
    double E{};   //!< f' - omega g
};

class ProcaStarSolution
{
  public:
    struct params_t
    {
        double mu = 1.;                //!< vector field mass
        double lambda = 0.;            //!< quartic self-interaction
        bool solitonic = false;        //!< solitonic potential
        double sigma = 0.2;            //!< solitonic potential parameter
        double G_Newton = 1.;          //!< Newton's constant
        double A_central = 0.1;        //!< central amplitude f(0)/alpha(0)
        int eigen = 0;                 //!< number of nodes of E (0 = ground state)
        double alpha_central = 0.9;    //!< initial central lapse guess (rescaled afterwards)
        double frequency_guess = 1.5;  //!< initial frequency guess (the search brackets up or down from here)
        double freq_epsilon = 1e-19;   //!< tolerance for the frequency bisection
        double r_match_fac = 0.85;     //!< matching radius as a fraction of the last minimum of |E|
        double R = 150.;               //!< outer radius of the 1D solution
        int n_gridpoints = 30001;      //!< 1D gridpoints
        bool perturb = false;          //!< apply the Gaussian perturbation
        double perturb_amp = 0.;       //!< a
        double perturb_spread = 1.;    //!< w
        double perturb_center = 0.;    //!< r0
    };

    ProcaStarSolution() = default;

    //! solve for the star (and perturb it if requested) and build the isotropic arrays; returns true on success
    bool solve(const params_t &a_params, std::ostream &os = std::cout, bool quiet = false);

    // isotropic-coordinate profiles at isotropic radius R
    double get_f(double R) const;      //!< A_t (even)
    double get_g(double R) const;      //!< A_R (odd)
    double get_E(double R) const;      //!< F_{Rt} = (f' - omega g) dr/dR (odd)
    double get_lapse(double R) const;  //!< alpha
    double get_psi(double R) const;    //!< conformal factor, gamma_ij = psi^4 delta_ij
    double get_dlapse(double R) const; //!< d alpha / dR
    double get_dpsi(double R) const;   //!< d psi / dR

    double omega = 0.;          //!< frequency (with alpha(infinity) = 1)
    double M = 0.;              //!< ADM mass
    double noether_charge = 0.; //!< particle number
    double binding_energy = 0.; //!< M - mu N
    double r_99 = 0.;           //!< areal radius containing 99% of the mass
    double A_central = 0.;      //!< f(0)/alpha(0) (after any perturbation)
    double f0 = 0.;             //!< f(0) in the gauge alpha(infinity) = 1

    void write_profiles(const std::string &filename) const;
    const std::vector<ProcaFieldState> &areal_state() const { return state; }
    const std::vector<double> &areal_g() const { return g_array; }

  private:
    params_t m_p;
    double mu, Gn;
    long double omega_ld = 0., omega_pre_rescale = 0.;
    int n_gridpoints = 0, blowup_point = 0;
    double R = 0., dr = 0.;
    bool g_solve_failed = false;
    double g_breakdown_r = -1.;

    std::vector<ProcaFieldState> state;
    std::vector<double> radius_array, g_array;
    std::vector<double> r_iso_array, psi_iso_array, phi_iso_array, f_iso_array, g_iso_array, E_iso_array;

    bool self_interacting() const { return m_p.solitonic || m_p.lambda != 0.; }
    double V(double Psi) const;
    double dV(double Psi) const;
    double ddV(double Psi) const;
    double solve_g(long double frequency, const ProcaFieldState &s, double &W, double &Psi);
    void matter(double radius, double X, double phi, double E, double f, double g, double &dX, double &dPhi,
                double &dE) const;
    ProcaFieldState state_RHS(double r, long double frequency, const ProcaFieldState &s, bool asymptotic);
    void rk4_solve(long double freq);
    int count_zero_crossings() const;
    long double find_frequency(std::ostream &os, bool quiet);
    int find_last_minimum() const;
    bool fill_asymptotic(std::ostream &os, bool quiet);
    void fill_g_array();
    void rescale_lapse(double phi_shift);
    double m(int j) const;
    double get_noether_charge() const;
    void compute_r99();
    double f_RHS(double r, double f) const;
    double r_areal(int j_iso) const;
    void fill_isotropic_arrays();
    void add_perturbation(double a, double k, double center, std::ostream &os, bool quiet);
    double interp_iso(const std::vector<double> &y, double Riso, bool odd, bool derivative = false) const;
    double tail(const std::vector<double> &y, double Riso, bool odd) const;
};

#include "ProcaStarSolution.impl.hpp"

#endif /* PROCASTARSOLUTION_HPP_ */
