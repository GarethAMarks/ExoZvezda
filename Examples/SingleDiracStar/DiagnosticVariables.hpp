/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIAGNOSTICVARIABLES_HPP
#define DIAGNOSTICVARIABLES_HPP

// assign an enum to each variable
enum
{
    c_mod_psi, // sqrt(2 pi n): the central amplitude A_central = g(0) at the centre of the initial star
    c_Madm,
    c_Jadm,
    c_N,       // particle-number integrand n chi^{-3/2} of both spinors
    c_rho,
    c_Ham,
    c_Mom1,
    c_Mom2,
    c_Mom3,

    NUM_DIAGNOSTIC_VARS
};

namespace DiagnosticVariables
{
static const std::array<std::string, NUM_DIAGNOSTIC_VARS> variable_names = {
    "mod_psi", "Madm", "Jadm", "N", "rho", "Ham", "Mom1", "Mom2", "Mom3"};
}

#endif /* DIAGNOSTICVARIABLES_HPP */
