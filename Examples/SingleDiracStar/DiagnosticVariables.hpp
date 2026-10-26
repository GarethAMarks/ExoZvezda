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
    c_n,   // particle number density of both spinors, psi_A^dag psi_A + psi_B^dag psi_B
    c_n_A, // number density of spinor A
    c_n_B, // number density of spinor B
    c_Ham,
    c_Mom1,
    c_Mom2,
    c_Mom3,

    NUM_DIAGNOSTIC_VARS
};

namespace DiagnosticVariables
{
static const std::array<std::string, NUM_DIAGNOSTIC_VARS> variable_names = {
    "mod_psi", "Madm", "Jadm", "N", "rho", "n", "n_A", "n_B", "Ham", "Mom1", "Mom2", "Mom3"};
}

#endif /* DIAGNOSTICVARIABLES_HPP */
