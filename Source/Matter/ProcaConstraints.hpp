/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef PROCACONSTRAINTS_HPP_
#define PROCACONSTRAINTS_HPP_

#include "Cell.hpp"
#include "MatterCCZ4RHS.hpp"
#include "NewConstraints.hpp"
#include "ProcaField.hpp"

/// Hamiltonian and momentum constraints with the Proca field, plus the Proca diagnostics
/**
 * As MatterConstraints<ProcaField> (the Proca stress tensor needs no lapse or shift), plus, optionally (enum < 0 to
 * skip),
 *   N        = n chi^{-3/2}, the integrand of the particle number (sum it as c_N for the boson stars),
 *   mod_Xphi = |X_phi|, which equals the central amplitude A_central = f(0)/alpha(0) of the 1D solver at the centre of
 *              a star at rest (so max mod_Xphi tracks the central amplitude, like max mod_phi for boson stars),
 *   rho      = n^a n^b T_ab,
 *   n        = Im(E^i Xbar_i), the particle number density measured by the normal observers (n = -n_a j^a; the
 *              particle number is int n sqrt(gamma) d^3x),
 *   Gauss    = |D_i E^i + mu^2 X_phi|, the modulus of the (complex) Gauss constraint.
 **/
class ProcaConstraints : public Constraints
{
  public:
    template <class data_t> using AllVars = typename MatterCCZ4RHS<ProcaField>::template Vars<data_t>;

    ProcaConstraints(const ProcaField a_matter, double dx, double G_Newton, int a_c_Ham, const Interval &a_c_Moms,
                     int a_c_N = -1, int a_c_mod_Xphi = -1, int a_c_rho = -1, int a_c_n = -1, int a_c_Gauss = -1,
                     int a_c_Ham_abs_terms = -1, const Interval &a_c_Moms_abs_terms = Interval())
        : Constraints(dx, a_c_Ham, a_c_Moms, a_c_Ham_abs_terms, a_c_Moms_abs_terms, 0.0), my_matter(a_matter),
          m_G_Newton(G_Newton), m_c_N(a_c_N), m_c_mod_Xphi(a_c_mod_Xphi), m_c_rho(a_c_rho), m_c_n(a_c_n),
          m_c_Gauss(a_c_Gauss)
    {
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        const auto vars = current_cell.template load_vars<AllVars>();
        const auto d1 = m_deriv.template diff1<AllVars>(current_cell);
        const auto d2 = m_deriv.template diff2<Constraints::Diff2Vars>(current_cell);

        const auto h_UU = TensorAlgebra::compute_inverse_sym(vars.h);
        const auto chris = TensorAlgebra::compute_christoffel(d1.h, h_UU);

        Vars<data_t> out = constraint_equations(vars, d1, d2, h_UU, chris);

        const emtensor_t<data_t> emtensor = my_matter.compute_emtensor(vars, d1, h_UU, chris.ULL);
        out.Ham += -16.0 * M_PI * m_G_Newton * emtensor.rho;
        out.Ham_abs_terms += 16.0 * M_PI * m_G_Newton * abs(emtensor.rho);
        FOR(i)
        {
            out.Mom[i] += -8.0 * M_PI * m_G_Newton * emtensor.Si[i];
            out.Mom_abs_terms[i] += 8.0 * M_PI * m_G_Newton * abs(emtensor.Si[i]);
        }
        store_vars(out, current_cell);

        const data_t n = ProcaField::number_density(vars);
        if (m_c_N >= 0)
            current_cell.store_vars(n * pow(vars.chi, -1.5), m_c_N);
        if (m_c_mod_Xphi >= 0)
            current_cell.store_vars(sqrt(vars.Xphi_Re * vars.Xphi_Re + vars.Xphi_Im * vars.Xphi_Im), m_c_mod_Xphi);
        if (m_c_rho >= 0)
            current_cell.store_vars(emtensor.rho, m_c_rho);
        if (m_c_n >= 0)
            current_cell.store_vars(n, m_c_n);
        if (m_c_Gauss >= 0)
        {
            data_t C_Re, C_Im;
            my_matter.gauss_constraint(C_Re, C_Im, vars, d1);
            current_cell.store_vars(sqrt(C_Re * C_Re + C_Im * C_Im), m_c_Gauss);
        }
    }

  protected:
    ProcaField my_matter;
    double m_G_Newton;
    int m_c_N, m_c_mod_Xphi, m_c_rho, m_c_n, m_c_Gauss;
};

#endif /* PROCACONSTRAINTS_HPP_ */
