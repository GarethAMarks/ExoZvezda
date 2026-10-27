/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef PROCAFIELD_HPP_
#define PROCAFIELD_HPP_

#include "CCZ4Geometry.hpp"
#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include <cmath>

/*
 A complex Proca field of mass mu coupled to CCZ4 in 3D: the Proca stars of Spherical_BS_Evolver (ProcaStar /
 ComplexVectorField, branch proca-stars) with the same conventions, signature (-,+,+,+), K_ij = -nabla_i n_j:

   L = -1/4 F_ab Fbar^ab - 1/2 mu^2 A_a Abar^a,   nabla_a F^{ab} = mu^2 A^b  (=> nabla_a A^a = 0).

 Variables (Zilhao, Witek & Cardoso 2015), real and imaginary parts of each:
   X_phi = -n^a A_a,   X_i = gamma_i^a A_a,   E^i = gamma^{ij} F_{ja} n^a,   Z (damps the Gauss constraint).
 Equations (F_ij = d_i X_j - d_j X_i, D_j(alpha F^{ij}) = (1/sqrt(gamma)) d_j(sqrt(gamma) alpha F^{ij})):
   d_t X_i   = L_beta X_i - alpha (E_i + D_i X_phi) - X_phi D_i alpha
   d_t E^i   = L_beta E^i + alpha K E^i + D_j(alpha F^{ij}) + alpha mu^2 X^i + alpha D^i Z
   d_t X_phi = L_beta X_phi + alpha K X_phi - alpha D_i X^i - X^i D_i alpha - alpha Z
   d_t Z     = L_beta Z + alpha (D_i E^i + mu^2 X_phi - kappa Z)
 Gauss constraint C = D_i E^i + mu^2 X_phi; with Z the constraints obey a damped wave equation (damping kappa).
 Sources (sums over the real and imaginary parts), B^2 = F_ij F^ij / 2:
   rho  = (E_i E^i + B^2)/2 + mu^2 (X_phi^2 + X_i X^i)/2,     S_i = F_ij E^j + mu^2 X_phi X_i,
   S_ij = -E_i E_j + F_ik F_jl gamma^kl + mu^2 X_i X_j + gamma_ij [E^2/2 - B^2/2 - mu^2 (X_k X^k - X_phi^2)/2].
 Number density measured by the normal observers n = -n_a j^a = Im(E^i Xbar_i) (j^a = Im(F^{ab} Abar_b)), positive for
 the stars of the 1D code; the particle number is int n sqrt(gamma) d^3x.

 Only the free field is evolved (as in the 1D code; the self-interacting potentials of ProcaStarSolution are for
 initial data and families only). The system is first order in space, so it needs Kreiss-Oliger dissipation, which
 MatterCCZ4RHS applies to all variables.

 Reflections: X_k and E^k are odd in x_k, everything else is even (GRChombo parities ODD_X/Y/Z and EVEN); no
 variables are swapped.

 Checked numerically in Examples/SingleProcaStar/derivations/check_proca3d.py (against the covariant 4D field
 equations and stress tensor, reflections, and the 1D equations for spherical data), and this header against that
 Python reference by Examples/SingleProcaStar/tests/test_proca_cpp.
*/

class ProcaField
{
  public:
    struct params_t
    {
        double mass;    //!< mu
        double damping; //!< kappa in d_t Z = alpha (C - kappa Z) + ...
    };

    explicit ProcaField(params_t a_params) : m_params(a_params) {}

    //! grid variables (enum entries c_Xx_Re ... c_Z_Im must be defined in UserVariables.hpp, the three components of
    //! each vector consecutive)
    template <class data_t> struct Vars
    {
        Tensor<1, data_t> X_Re, X_Im; //!< X_i
        Tensor<1, data_t> E_Re, E_Im; //!< E^i
        data_t Xphi_Re, Xphi_Im, Z_Re, Z_Im;

        template <typename mapping_function_t> void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools;
            define_enum_mapping(mapping_function, GRInterval<c_Xx_Re, c_Xz_Re>(), X_Re);
            define_enum_mapping(mapping_function, GRInterval<c_Xx_Im, c_Xz_Im>(), X_Im);
            define_enum_mapping(mapping_function, GRInterval<c_Ex_Re, c_Ez_Re>(), E_Re);
            define_enum_mapping(mapping_function, GRInterval<c_Ex_Im, c_Ez_Im>(), E_Im);
            define_enum_mapping(mapping_function, c_Xphi_Re, Xphi_Re);
            define_enum_mapping(mapping_function, c_Xphi_Im, Xphi_Im);
            define_enum_mapping(mapping_function, c_Z_Re, Z_Re);
            define_enum_mapping(mapping_function, c_Z_Im, Z_Im);
        }
    };

    //! second derivatives of X_i (for the curl term)
    template <class data_t> struct Diff2Vars
    {
        Tensor<1, data_t> X_Re, X_Im;

        template <typename mapping_function_t> void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools;
            define_enum_mapping(mapping_function, GRInterval<c_Xx_Re, c_Xz_Re>(), X_Re);
            define_enum_mapping(mapping_function, GRInterval<c_Xx_Im, c_Xz_Im>(), X_Im);
        }
    };

    //! the physical metric quantities needed by the Proca equations
    template <class data_t> struct metric_t
    {
        Tensor<2, data_t> g_LL, g_UU; //!< gamma_ij, gamma^ij
        Tensor<3, data_t> dg_UU;      //!< d_k gamma^ij, index order [i][j][k]
        Tensor<1, data_t> dlnsg;      //!< d_k ln sqrt(gamma)
    };

    template <class data_t, template <typename> class vars_t>
    static metric_t<data_t> metric(const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1)
    {
        metric_t<data_t> m;
        const auto h_UU = TensorAlgebra::compute_inverse_sym(vars.h);
        FOR(i, j)
        {
            m.g_LL[i][j] = vars.h[i][j] / vars.chi;
            m.g_UU[i][j] = vars.chi * h_UU[i][j];
        }
        FOR(k)
        {
            m.dlnsg[k] = -1.5 * d1.chi[k] / vars.chi;
            FOR(a, b) { m.dlnsg[k] += 0.5 * h_UU[a][b] * d1.h[a][b][k]; }
        }
        FOR(i, j, k)
        {
            m.dg_UU[i][j][k] = d1.chi[k] * h_UU[i][j];
            FOR(a, b) { m.dg_UU[i][j][k] += -vars.chi * h_UU[i][a] * d1.h[a][b][k] * h_UU[b][j]; }
        }
        return m;
    }

    //! stress tensor of the field (no lapse or shift needed)
    template <class data_t, template <typename> class vars_t>
    emtensor_t<data_t> compute_emtensor(const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
                                        const Tensor<2, data_t> &h_UU, const Tensor<3, data_t> &chris_ULL) const
    {
        const auto m = metric(vars, d1);
        emtensor_t<data_t> out;
        out.rho = 0.;
        FOR(i)
        {
            out.Si[i] = 0.;
            FOR(j) { out.Sij[i][j] = 0.; }
        }
        add_component_emtensor(out, m, vars.X_Re, vars.E_Re, vars.Xphi_Re, d1.X_Re);
        add_component_emtensor(out, m, vars.X_Im, vars.E_Im, vars.Xphi_Im, d1.X_Im);
        out.S = TensorAlgebra::compute_trace(out.Sij, m.g_UU);
        return out;
    }

    //! RHS of the Proca variables (vars_t contains the CCZ4 variables with gauge and Vars)
    template <class data_t, template <typename> class vars_t, template <typename> class diff2_vars_t,
              template <typename> class rhs_vars_t>
    void add_matter_rhs(rhs_vars_t<data_t> &total_rhs, const vars_t<data_t> &vars,
                        const vars_t<Tensor<1, data_t>> &d1, const diff2_vars_t<Tensor<2, data_t>> &d2,
                        const vars_t<data_t> &advec) const
    {
        const auto m = metric(vars, d1);
        component_rhs(total_rhs.X_Re, total_rhs.E_Re, total_rhs.Xphi_Re, total_rhs.Z_Re, vars.lapse, d1.lapse,
                      d1.shift, vars.K, m, vars.X_Re, vars.E_Re, vars.Xphi_Re, vars.Z_Re, d1.X_Re, d1.E_Re,
                      d1.Xphi_Re, d1.Z_Re, d2.X_Re);
        component_rhs(total_rhs.X_Im, total_rhs.E_Im, total_rhs.Xphi_Im, total_rhs.Z_Im, vars.lapse, d1.lapse,
                      d1.shift, vars.K, m, vars.X_Im, vars.E_Im, vars.Xphi_Im, vars.Z_Im, d1.X_Im, d1.E_Im,
                      d1.Xphi_Im, d1.Z_Im, d2.X_Im);
        FOR(i)
        {
            total_rhs.X_Re[i] += advec.X_Re[i];
            total_rhs.X_Im[i] += advec.X_Im[i];
            total_rhs.E_Re[i] += advec.E_Re[i];
            total_rhs.E_Im[i] += advec.E_Im[i];
        }
        total_rhs.Xphi_Re += advec.Xphi_Re;
        total_rhs.Xphi_Im += advec.Xphi_Im;
        total_rhs.Z_Re += advec.Z_Re;
        total_rhs.Z_Im += advec.Z_Im;
    }

    //! number density n = Im(E^i Xbar_i) measured by the normal observers
    template <class data_t, template <typename> class vars_t>
    static data_t number_density(const vars_t<data_t> &vars)
    {
        data_t n = 0.;
        FOR(i) { n += vars.E_Im[i] * vars.X_Re[i] - vars.E_Re[i] * vars.X_Im[i]; }
        return n;
    }

    //! Gauss constraint C = D_i E^i + mu^2 X_phi (real and imaginary parts)
    template <class data_t, template <typename> class vars_t>
    void gauss_constraint(data_t &C_Re, data_t &C_Im, const vars_t<data_t> &vars,
                          const vars_t<Tensor<1, data_t>> &d1) const
    {
        const auto m = metric(vars, d1);
        const double mu2 = m_params.mass * m_params.mass;
        C_Re = mu2 * vars.Xphi_Re;
        C_Im = mu2 * vars.Xphi_Im;
        FOR(i)
        {
            C_Re += d1.E_Re[i][i] + vars.E_Re[i] * m.dlnsg[i];
            C_Im += d1.E_Im[i][i] + vars.E_Im[i] * m.dlnsg[i];
        }
    }

  protected:
    params_t m_params;

    //! F_ab = d_a X_b - d_b X_a (GRChombo derivative index last: dX[b][a] = d_a X_b)
    template <class data_t, class d1_t> static Tensor<2, data_t> field_strength(const d1_t &dX)
    {
        Tensor<2, data_t> F;
        FOR(a, b) { F[a][b] = dX[b][a] - dX[a][b]; }
        return F;
    }

    template <class data_t, class d1_t>
    void add_component_emtensor(emtensor_t<data_t> &out, const metric_t<data_t> &m, const Tensor<1, data_t> &X,
                                const Tensor<1, data_t> &E, const data_t &P, const d1_t &dX) const
    {
        const double mu2 = m_params.mass * m_params.mass;
        const auto F = field_strength<data_t>(dX);
        Tensor<1, data_t> E_dn, X_up;
        FOR(i)
        {
            E_dn[i] = 0.;
            X_up[i] = 0.;
            FOR(j)
            {
                E_dn[i] += m.g_LL[i][j] * E[j];
                X_up[i] += m.g_UU[i][j] * X[j];
            }
        }
        Tensor<2, data_t> F_mix; // F_i^j = F_ik gamma^kj
        FOR(i, j)
        {
            F_mix[i][j] = 0.;
            FOR(k) { F_mix[i][j] += F[i][k] * m.g_UU[k][j]; }
        }
        data_t E2 = 0., B2 = 0., X2 = 0.;
        FOR(i)
        {
            E2 += E_dn[i] * E[i];
            X2 += X[i] * X_up[i];
            FOR(j) { B2 += 0.5 * F_mix[i][j] * F_mix[j][i] * (-1.); } // F_ij F^ij = -F_i^j F_j^i
        }
        const data_t iso = 0.5 * E2 - 0.5 * B2 - 0.5 * mu2 * (X2 - P * P);
        out.rho += 0.5 * (E2 + B2) + 0.5 * mu2 * (P * P + X2);
        FOR(i)
        {
            data_t FE = 0.;
            FOR(j) { FE += F[i][j] * E[j]; }
            out.Si[i] += FE + mu2 * P * X[i];
            FOR(j)
            {
                data_t FF = 0.;
                FOR(k) { FF += F_mix[i][k] * F[j][k]; }
                out.Sij[i][j] += -E_dn[i] * E_dn[j] + FF + mu2 * X[i] * X[j] + m.g_LL[i][j] * iso;
            }
        }
    }

    //! RHS of one real component (without the advection terms beta^k d_k)
    //! (d1_t: first derivatives of a vector or of the shift, [i][k] = d_k V^i; d2_t: [i][j][k] = d_j d_k X_i)
    template <class data_t, class d1_t, class d2_t>
    void component_rhs(Tensor<1, data_t> &rX, Tensor<1, data_t> &rE, data_t &rP, data_t &rZ, const data_t &alpha,
                       const Tensor<1, data_t> &dalpha, const d1_t &dbeta, const data_t &trK, const metric_t<data_t> &m,
                       const Tensor<1, data_t> &X, const Tensor<1, data_t> &E, const data_t &P, const data_t &Z,
                       const d1_t &dX, const d1_t &dE, const Tensor<1, data_t> &dP, const Tensor<1, data_t> &dZ,
                       const d2_t &ddX) const
    {
        const double mu2 = m_params.mass * m_params.mass;
        const double kappa = m_params.damping;

        Tensor<1, data_t> E_dn, X_up;
        FOR(i)
        {
            E_dn[i] = 0.;
            X_up[i] = 0.;
            FOR(j)
            {
                E_dn[i] += m.g_LL[i][j] * E[j];
                X_up[i] += m.g_UU[i][j] * X[j];
            }
        }
        data_t divX = 0., divE = 0.;
        FOR(i)
        {
            divX += X_up[i] * m.dlnsg[i];
            divE += dE[i][i] + E[i] * m.dlnsg[i];
            FOR(j) { divX += m.dg_UU[i][j][i] * X[j] + m.g_UU[i][j] * dX[j][i]; }
        }

        // D_j(alpha F^{ij}) = alpha (d_j F^{ij} + F^{ij} d_j ln sqrt(gamma)) + F^{ij} d_j alpha
        const auto F = field_strength<data_t>(dX);
        Tensor<2, data_t> gF, F_UU; // gF[i][b] = gamma^{ia} F_ab
        FOR(i, b)
        {
            gF[i][b] = 0.;
            FOR(a) { gF[i][b] += m.g_UU[i][a] * F[a][b]; }
        }
        FOR(i, j)
        {
            F_UU[i][j] = 0.;
            FOR(b) { F_UU[i][j] += gF[i][b] * m.g_UU[b][j]; }
        }
        Tensor<1, data_t> curl;
        FOR(i)
        {
            data_t divF = 0.;
            FOR(j)
            {
                divF += F_UU[i][j] * m.dlnsg[j];
                FOR(b)
                {
                    divF += gF[i][b] * m.dg_UU[b][j][j];
                    FOR(a)
                    {
                        const data_t dF_ab = ddX[b][a][j] - ddX[a][b][j]; // d_j F_ab
                        divF += (m.dg_UU[i][a][j] * F[a][b] + m.g_UU[i][a] * dF_ab) * m.g_UU[b][j];
                    }
                }
            }
            curl[i] = alpha * divF;
            FOR(j) { curl[i] += F_UU[i][j] * dalpha[j]; }
        }

        FOR(i)
        {
            rX[i] = -alpha * (E_dn[i] + dP[i]) - P * dalpha[i];
            rE[i] = alpha * trK * E[i] + alpha * mu2 * X_up[i] + curl[i];
            FOR(j)
            {
                rX[i] += X[j] * dbeta[j][i];  // X_j d_i beta^j
                rE[i] += -E[j] * dbeta[i][j]; // -E^j d_j beta^i
                rE[i] += alpha * m.g_UU[i][j] * dZ[j];
            }
        }
        rP = alpha * trK * P - alpha * divX - alpha * Z;
        FOR(i) { rP += -X_up[i] * dalpha[i]; }
        rZ = alpha * (divE + mu2 * P - kappa * Z);
    }
};

#endif /* PROCAFIELD_HPP_ */
