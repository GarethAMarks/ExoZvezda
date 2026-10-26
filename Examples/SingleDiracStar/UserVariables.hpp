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
    // the two Dirac spinors as 8 complex combinations (see Source/Matter/DiracField.hpp); the suffix lists the
    // directions in which each is odd for a mirror-symmetric configuration (parities 0 1 2 3 4 5 6 7 in this order)
    c_D_e_Re = NUM_CCZ4_VARS,
    c_D_e_Im,
    c_D_x_Re,
    c_D_x_Im,
    c_D_y_Re,
    c_D_y_Im,
    c_D_z_Re,
    c_D_z_Im,
    c_D_xy_Re,
    c_D_xy_Im,
    c_D_yz_Re,
    c_D_yz_Im,
    c_D_xz_Re,
    c_D_xz_Im,
    c_D_xyz_Re,
    c_D_xyz_Im,

    NUM_VARS
};

namespace UserVariables
{
static const std::array<std::string, NUM_VARS - NUM_CCZ4_VARS> dirac_variable_names = {
    "D_e_Re",  "D_e_Im",  "D_x_Re",  "D_x_Im",  "D_y_Re",  "D_y_Im",  "D_z_Re",   "D_z_Im",
    "D_xy_Re", "D_xy_Im", "D_yz_Re", "D_yz_Im", "D_xz_Re", "D_xz_Im", "D_xyz_Re", "D_xyz_Im"};

static const std::array<std::string, NUM_VARS> variable_names =
    ArrayTools::concatenate(ccz4_variable_names, dirac_variable_names);
} // namespace UserVariables

#include "UserVariables.inc.hpp"

#endif /* USERVARIABLES_HPP */
