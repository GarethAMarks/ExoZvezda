/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIRACCONSTRAINTS_HPP_
#define DIRACCONSTRAINTS_HPP_

#include "Cell.hpp"
#include "DiracField.hpp"
#include "MatterCCZ4RHS.hpp"
#include "NewConstraints.hpp"

/// Hamiltonian and momentum constraints with the Dirac field, plus the Dirac diagnostics
/**
 * As MatterConstraints<matter_t>, but the lapse, shift and K_ij are loaded too: the energy density contains
 * n^a nabla_a psi, which is eliminated with the Dirac equation. Optionally also stores (enum < 0 to skip)
 *   N       = n chi^{-3/2}, the integrand of the particle number of both spinors (sum it as c_N for the boson stars),
 *   mod_psi = sqrt(2 pi n), which equals the central amplitude A_central = g(0) of the 1D solver at the centre of the
 *             initial star (so max mod_psi tracks the central amplitude, like max mod_phi for boson stars),
 *   rho     = n^a n^b T_ab,
 *   n       = psi_A^dag psi_A + psi_B^dag psi_B, the particle number density measured by the normal observers
 *             (n = -n_mu J^mu; the particle number is int n sqrt(gamma) d^3x),
 *   n_A,n_B = the number densities of the two spinor fields separately (n_A + n_B = n).
 **/
class DiracConstraints : public Constraints
{
  public:
    template <class data_t> using AllVars = typename MatterCCZ4RHS<DiracField>::template Vars<data_t>;

    DiracConstraints(const DiracField a_matter, double dx, double G_Newton, int a_c_Ham, const Interval &a_c_Moms,
                     int a_c_N = -1, int a_c_mod_psi = -1, int a_c_rho = -1, int a_c_n = -1, int a_c_n_A = -1,
                     int a_c_n_B = -1, int a_c_Ham_abs_terms = -1, const Interval &a_c_Moms_abs_terms = Interval())
        : Constraints(dx, a_c_Ham, a_c_Moms, a_c_Ham_abs_terms, a_c_Moms_abs_terms, 0.0), my_matter(a_matter),
          m_G_Newton(G_Newton), m_c_N(a_c_N), m_c_mod_psi(a_c_mod_psi), m_c_rho(a_c_rho),
          m_c_n(a_c_n), m_c_n_A(a_c_n_A), m_c_n_B(a_c_n_B)
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

        DiracField::Vars<data_t> rhs_D;
        emtensor_t<data_t> emtensor;
        data_t n;
        my_matter.compute(rhs_D, emtensor, n, vars, d1);

        out.Ham += -16.0 * M_PI * m_G_Newton * emtensor.rho;
        out.Ham_abs_terms += 16.0 * M_PI * m_G_Newton * abs(emtensor.rho);
        FOR(i)
        {
            out.Mom[i] += -8.0 * M_PI * m_G_Newton * emtensor.Si[i];
            out.Mom_abs_terms[i] += 8.0 * M_PI * m_G_Newton * abs(emtensor.Si[i]);
        }
        store_vars(out, current_cell);

        if (m_c_N >= 0)
            current_cell.store_vars(n * pow(vars.chi, -1.5), m_c_N);
        if (m_c_mod_psi >= 0)
            current_cell.store_vars(sqrt(2. * M_PI * n), m_c_mod_psi);
        if (m_c_rho >= 0)
            current_cell.store_vars(emtensor.rho, m_c_rho);
        if (m_c_n >= 0)
            current_cell.store_vars(n, m_c_n);
        if (m_c_n_A >= 0 || m_c_n_B >= 0)
        {
            data_t n_A, n_B;
            DiracField::spinor_number_densities(vars, n_A, n_B);
            if (m_c_n_A >= 0)
                current_cell.store_vars(n_A, m_c_n_A);
            if (m_c_n_B >= 0)
                current_cell.store_vars(n_B, m_c_n_B);
        }
    }

  protected:
    DiracField my_matter;
    double m_G_Newton;
    int m_c_N, m_c_mod_psi, m_c_rho, m_c_n, m_c_n_A, m_c_n_B;
};

#endif /* DIRACCONSTRAINTS_HPP_ */
