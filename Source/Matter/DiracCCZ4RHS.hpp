/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIRACCCZ4RHS_HPP_
#define DIRACCCZ4RHS_HPP_

#include "DiracField.hpp"
#include "MatterCCZ4RHS.hpp"

/// CCZ4 right-hand side with the Dirac field (two spinors, see DiracField.hpp)
/**
 * The same as MatterCCZ4RHS<DiracField> (which also works, through DiracField's compute_emtensor and add_matter_rhs),
 * but the spin connection, the spinor RHS and the stress tensor are computed once per cell instead of twice. The matter
 * terms are added exactly as in MatterCCZ4RHS::add_emtensor_rhs. Kreiss-Oliger dissipation is applied to all variables,
 * including the spinors (the first-order Dirac system needs it).
 **/
template <class gauge_t = MovingPunctureGauge, class deriv_t = FourthOrderDerivatives>
class DiracCCZ4RHS : public MatterCCZ4RHS<DiracField, gauge_t, deriv_t>
{
  public:
    using base_t = MatterCCZ4RHS<DiracField, gauge_t, deriv_t>;
    template <class data_t> using Vars = typename base_t::template Vars<data_t>;
    template <class data_t> using Diff2Vars = typename base_t::template Diff2Vars<data_t>;

    DiracCCZ4RHS(DiracField a_matter, CCZ4_params_t<typename gauge_t::params_t> a_params, double a_dx,
                 double a_sigma, int a_formulation = CCZ4RHS<>::USE_CCZ4, double a_G_Newton = 1.0)
        : base_t(a_matter, a_params, a_dx, a_sigma, a_formulation, a_G_Newton)
    {
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        const auto vars = current_cell.template load_vars<Vars>();
        const auto d1 = this->m_deriv.template diff1<Vars>(current_cell);
        const auto d2 = this->m_deriv.template diff2<Diff2Vars>(current_cell);
        const auto advec = this->m_deriv.template advection<Vars>(current_cell, vars.shift);

        Vars<data_t> rhs;
        this->rhs_equation(rhs, vars, d1, d2, advec);

        DiracField::Vars<data_t> rhs_D;
        emtensor_t<data_t> emtensor;
        data_t n;
        this->my_matter.compute(rhs_D, emtensor, n, vars, d1);
        for (int k = 0; k < 8; ++k)
        {
            rhs.D_Re[k] = rhs_D.D_Re[k] + advec.D_Re[k];
            rhs.D_Im[k] = rhs_D.D_Im[k] + advec.D_Im[k];
        }
        add_dirac_emtensor_rhs(rhs, vars, emtensor);

        this->m_deriv.add_dissipation(rhs, current_cell, this->m_sigma);
        current_cell.store_vars(rhs);
    }

  protected:
    //! as MatterCCZ4RHS::add_emtensor_rhs, for a given stress tensor
    template <class data_t>
    void add_dirac_emtensor_rhs(Vars<data_t> &matter_rhs, const Vars<data_t> &matter_vars,
                                const emtensor_t<data_t> &emtensor) const
    {
        using namespace TensorAlgebra;
        const auto h_UU = compute_inverse_sym(matter_vars.h);
        const double G = this->m_G_Newton;
        if (this->m_formulation == CCZ4RHS<>::USE_BSSN)
        {
            matter_rhs.K += 4.0 * M_PI * G * matter_vars.lapse * (emtensor.S + emtensor.rho);
            matter_rhs.Theta += 0.0;
        }
        else
        {
            matter_rhs.K += 4.0 * M_PI * G * matter_vars.lapse * (emtensor.S - 3 * emtensor.rho);
            matter_rhs.Theta += -8.0 * M_PI * G * matter_vars.lapse * emtensor.rho;
        }
        Tensor<2, data_t> Sij_TF = emtensor.Sij;
        make_trace_free(Sij_TF, matter_vars.h, h_UU);
        FOR(i, j)
        {
            matter_rhs.A[i][j] += -8.0 * M_PI * G * matter_vars.chi * matter_vars.lapse * Sij_TF[i][j];
        }
        FOR(i)
        {
            data_t matter_term_Gamma = 0.0;
            FOR(j) { matter_term_Gamma += -16.0 * M_PI * G * matter_vars.lapse * h_UU[i][j] * emtensor.Si[j]; }
            matter_rhs.Gamma[i] += matter_term_Gamma;
        }
        this->m_gauge.rhs_gauge_add_matter_terms(matter_rhs, matter_vars, h_UU, emtensor, G);
    }
};

#endif /* DIRACCCZ4RHS_HPP_ */
