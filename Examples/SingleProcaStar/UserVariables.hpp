/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef USERVARIABLES_HPP
#define USERVARIABLES_HPP

#include "ArrayTools.hpp"
#include "CCZ4UserVariables.hpp"
#include "DiagnosticVariables.hpp"
#include <array>
#include <string>

// assign an enum to each variable
enum
{
    // the complex Proca field (see Source/Matter/ProcaField.hpp): X_i, E^i, X_phi = -n^a A_a and the Gauss-constraint
    // damping variable Z; the three components of each vector must be consecutive
    c_Xx_Re = NUM_CCZ4_VARS,
    c_Xy_Re,
    c_Xz_Re,
    c_Xx_Im,
    c_Xy_Im,
    c_Xz_Im,
    c_Ex_Re,
    c_Ey_Re,
    c_Ez_Re,
    c_Ex_Im,
    c_Ey_Im,
    c_Ez_Im,
    c_Xphi_Re,
    c_Xphi_Im,
    c_Z_Re,
    c_Z_Im,

    NUM_VARS
};

namespace UserVariables
{
static const std::array<std::string, NUM_VARS - NUM_CCZ4_VARS> proca_variable_names = {
    "Xx_Re", "Xy_Re", "Xz_Re", "Xx_Im", "Xy_Im", "Xz_Im", "Ex_Re",   "Ey_Re",
    "Ez_Re", "Ex_Im", "Ey_Im", "Ez_Im", "Xphi_Re", "Xphi_Im", "Z_Re", "Z_Im"};

static const std::array<std::string, NUM_VARS> variable_names =
    ArrayTools::concatenate(ccz4_variable_names, proca_variable_names);
} // namespace UserVariables

#include "UserVariables.inc.hpp"

#endif /* USERVARIABLES_HPP */
