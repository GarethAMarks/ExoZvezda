/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIAGNOSTICVARIABLES_HPP
#define DIAGNOSTICVARIABLES_HPP

// assign an enum to each variable
enum
{
    c_mod_Xphi, // |X_phi|: the central amplitude A_central = f(0)/alpha(0) at the centre of a star at rest
    c_Madm,
    c_Jadm,
    c_N,     // particle-number integrand n chi^{-3/2}
    c_rho,   // energy density
    c_n,     // particle number density measured by the normal observers, Im(E^i Xbar_i)
    c_Gauss, // |D_i E^i + mu^2 X_phi|
    c_Ham,
    c_Mom1,
    c_Mom2,
    c_Mom3,

    NUM_DIAGNOSTIC_VARS
};

namespace DiagnosticVariables
{
static const std::array<std::string, NUM_DIAGNOSTIC_VARS> variable_names = {
    "mod_Xphi", "Madm", "Jadm", "N", "rho", "n", "Gauss", "Ham", "Mom1", "Mom2", "Mom3"};
}

#endif /* DIAGNOSTICVARIABLES_HPP */
