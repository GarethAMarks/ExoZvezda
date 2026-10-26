/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIRACFIELD_HPP_
#define DIRACFIELD_HPP_

#include "CCZ4Geometry.hpp"
#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp"
#include "VarsTools.hpp"
#include <array>
#include <cmath>
#include <complex>

/*
 Two Dirac spinors of mass mu coupled to CCZ4 in 3D: the Dirac stars of Spherical_BS_Evolver (DiracStar /
 ComplexSpinorField) and of 2DChombo (DiracFieldCartoon) with the same conventions, signature (-,+,+,+):

   (gamma^a nabla_a + mu) psi = 0,  {gamma^a, gamma^b} = 2 eta^{ab},  gamma^a = -i gamma_D^a (standard Dirac
   representation), psibar = psi^dag gamma_D^0,
   T_ab = (1/4)(X_ab + X_ba),  X_ab = psibar gamma_a nabla_b psi - (nabla_a psi)^dag gamma_D^0 gamma_b psi,
   summed over the two spinors psi_A and psi_B (for the spherical star, the j = 1/2 states with m_j = +1/2 and -1/2).

 Tetrad: time gauge e_0 = n and the Cholesky (Gram-Schmidt) triad, gamma_ij = L L^T with L lower triangular, co-triad
 theta^a_i = L_ia, triad e_a^i = (L^{-T})_ia. This is a smooth algebraic function of gamma_ij (no extra variables),
 is covariant under the coordinate reflections x, y, z -> -x, -y, -z (so reflective boundaries work), and equals the
 symmetric triad used in 1D/2D for conformally flat gamma_ij (so the initial spinor components are the same). Its
 derivatives are closed form: dL = L Phi(L^{-1} d gamma L^{-T}), Phi = lower triangle with half the diagonal.

 3+1 equation (Gamma_mu = (1/8) omega_{mu ab} [gamma^a, gamma^b], K_ij = -nabla_i n_j as in GRChombo):
   d_t psi = beta^k d_k psi - alpha Gamma_n psi + alpha gamma^0 gamma^b e_b^k (d_k psi + Gamma_k psi) + alpha mu gamma^0 psi,
   Gamma_k = (1/2) omega_{k0b} gamma^0 gamma^b + (1/4) omega_{kab} gamma^a gamma^b,   omega_{k0b} = K_kj e_b^j,
   Gamma_n = (1/2) n.omega_{0b} gamma^0 gamma^b + (1/4) n.omega_{ab} gamma^a gamma^b,  n.omega_{0b} = -e_b^j d_j alpha/alpha,
   n.omega_{ab} = [ e_a.(d_t e_b - L_beta e_b) - (a <-> b) ] / (2 alpha)   (rotation of the triad as gamma_ij evolves,
                  d_t gamma_ij = -2 alpha K_ij + L_beta gamma_ij),
   omega_{kab} = gamma_ij e_a^i (d_k e_b^j + Gamma^j_kl e_b^l).
 The time derivative in T_ab is eliminated with the equation itself, so rho, S_i, S_ij need only slice data.

 Grid variables. The two spinors (psi_A = A0..A3, psi_B = B0..B3, components upper-up, upper-down, lower-up,
 lower-down along z) are stored as 8 complex combinations, each with a definite parity under every coordinate
 reflection of a mirror-symmetric configuration (the suffix lists the directions in which the variable is odd):
   D_e   = (A0 + B1)/sqrt2     D_xy  = (A0 - B1)/sqrt2     D_yz = (A1 + B0)/sqrt2     D_xz = (A1 - B0)/sqrt2
   D_xyz = (A2 + B3)/sqrt2     D_z   = (A2 - B3)/sqrt2     D_x  = (A3 + B2)/sqrt2     D_y  = (A3 - B2)/sqrt2
 (the reflections act on (psi_A, psi_B) as -tau_x (x) M_x, tau_y (x) M_y, -tau_z (x) M_z, M_a = gamma_D^5 gamma_D^a,
 which leave the star invariant and are diagonal in this basis). For the star at rest
   D_e = g / sqrt(2 pi),  D_x = -i f n_x / sqrt(2 pi),  D_y = f n_y / sqrt(2 pi),  D_z = -i f n_z / sqrt(2 pi),
 the others vanish, with f, g the physical radial spinor functions of the 1D solver (g(0) = A_central).

 Everything here is checked numerically in Examples/SingleDiracStar/derivations/check_dirac3d.py (against the
 covariant 4D Dirac operator, reflections, and the 1D equations for spherical data), and this header against that
 Python reference by Examples/SingleDiracStar/tests/test_dirac_cpp.
*/

namespace DiracFieldHelpers
{
template <class data_t> struct cplx
{
    data_t re, im;
};
template <class data_t> inline cplx<data_t> operator+(const cplx<data_t> &a, const cplx<data_t> &b)
{
    return {a.re + b.re, a.im + b.im};
}
template <class data_t> inline cplx<data_t> operator*(const data_t &c, const cplx<data_t> &a)
{
    return {c * a.re, c * a.im};
}
template <class data_t> struct spinor
{
    cplx<data_t> c[4];
};
template <class data_t> inline spinor<data_t> zero_spinor()
{
    spinor<data_t> out;
    for (int i = 0; i < 4; ++i)
        out.c[i] = {0., 0.};
    return out;
}
template <class data_t> inline spinor<data_t> operator+(const spinor<data_t> &a, const spinor<data_t> &b)
{
    spinor<data_t> out;
    for (int i = 0; i < 4; ++i)
        out.c[i] = a.c[i] + b.c[i];
    return out;
}
template <class data_t> inline spinor<data_t> operator*(const data_t &s, const spinor<data_t> &a)
{
    spinor<data_t> out;
    for (int i = 0; i < 4; ++i)
        out.c[i] = s * a.c[i];
    return out;
}

using cmat = std::array<std::array<std::complex<double>, 4>, 4>;

template <class data_t> inline spinor<data_t> apply(const cmat &M, const spinor<data_t> &p)
{
    spinor<data_t> out = zero_spinor<data_t>();
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            const double mr = M[i][j].real(), mi = M[i][j].imag();
            if (mr != 0.)
            {
                out.c[i].re += mr * p.c[j].re;
                out.c[i].im += mr * p.c[j].im;
            }
            if (mi != 0.)
            {
                out.c[i].re -= mi * p.c[j].im;
                out.c[i].im += mi * p.c[j].re;
            }
        }
    return out;
}

//! Re < p, M q >
template <class data_t> inline data_t re_inner(const spinor<data_t> &p, const cmat &M, const spinor<data_t> &q)
{
    const spinor<data_t> Mq = apply(M, q);
    data_t out = 0.;
    for (int i = 0; i < 4; ++i)
        out += p.c[i].re * Mq.c[i].re + p.c[i].im * Mq.c[i].im;
    return out;
}

struct Constants
{
    cmat G0;       //!< gamma^0
    cmat G0G[3];   //!< gamma^0 gamma^b
    cmat GG[3][3]; //!< gamma^a gamma^b
    cmat BG[4];    //!< gamma_D^0 gamma_a (lower frame index)

    Constants()
    {
        using cd = std::complex<double>;
        const cd I(0., 1.);
        typedef std::array<std::array<cd, 2>, 2> m2;
        const m2 sig[3] = {{{{0., 1.}, {1., 0.}}}, {{{0., -I}, {I, 0.}}}, {{{1., 0.}, {0., -1.}}}};
        cmat gD[4], gam[4];
        for (int a = 0; a < 4; ++a)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    gD[a][i][j] = 0.;
        for (int i = 0; i < 2; ++i)
        {
            gD[0][i][i] = 1.;
            gD[0][i + 2][i + 2] = -1.;
        }
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                {
                    gD[k + 1][i][j + 2] = sig[k][i][j];
                    gD[k + 1][i + 2][j] = -sig[k][i][j];
                }
        for (int a = 0; a < 4; ++a)
            gam[a] = scale(-I, gD[a]);
        G0 = gam[0];
        for (int b = 0; b < 3; ++b)
        {
            G0G[b] = mult(gam[0], gam[b + 1]);
            for (int a = 0; a < 3; ++a)
                GG[a][b] = mult(gam[a + 1], gam[b + 1]);
        }
        BG[0] = mult(gD[0], scale(-1., gam[0]));
        for (int a = 1; a < 4; ++a)
            BG[a] = mult(gD[0], gam[a]);
    }
    static cmat mult(const cmat &A, const cmat &B)
    {
        cmat C;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
            {
                C[i][j] = 0.;
                for (int k = 0; k < 4; ++k)
                    C[i][j] += A[i][k] * B[k][j];
            }
        return C;
    }
    static cmat scale(std::complex<double> c, const cmat &A)
    {
        cmat C;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                C[i][j] = c * A[i][j];
        return C;
    }
};

inline const Constants &constants()
{
    static const Constants c;
    return c;
}

// D (8 combinations) <-> (psi_A, psi_B): D_k = sum_c s_kc P_c / sqrt2 over two components, see the header comment.
// Each entry: {component of D index -> (spinor index 0..7 = A0..A3, B0..B3, sign)} pairs, in the order
// D_e, D_x, D_y, D_z, D_xy, D_yz, D_xz, D_xyz
static const int dirac_comb_idx[8][2] = {{0, 5}, {3, 6}, {3, 6}, {2, 7}, {0, 5}, {1, 4}, {1, 4}, {2, 7}};
static const double dirac_comb_sgn[8][2] = {{1, 1}, {1, 1}, {1, -1}, {1, -1}, {1, -1}, {1, 1}, {1, -1}, {1, 1}};
} // namespace DiracFieldHelpers

class DiracField
{
  public:
    struct params_t
    {
        double mass; //!< fermion mass mu
    };

    explicit DiracField(params_t a_params) : m_params(a_params) {}

    //! the 8 complex grid variables (enum entries c_D_e_Re ... c_D_xyz_Im must be defined in UserVariables.hpp)
    template <class data_t> struct Vars
    {
        data_t D_Re[8]; //!< order: e, x, y, z, xy, yz, xz, xyz
        data_t D_Im[8];

        template <typename mapping_function_t> void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools;
            define_enum_mapping(mapping_function, c_D_e_Re, D_Re[0]);
            define_enum_mapping(mapping_function, c_D_e_Im, D_Im[0]);
            define_enum_mapping(mapping_function, c_D_x_Re, D_Re[1]);
            define_enum_mapping(mapping_function, c_D_x_Im, D_Im[1]);
            define_enum_mapping(mapping_function, c_D_y_Re, D_Re[2]);
            define_enum_mapping(mapping_function, c_D_y_Im, D_Im[2]);
            define_enum_mapping(mapping_function, c_D_z_Re, D_Re[3]);
            define_enum_mapping(mapping_function, c_D_z_Im, D_Im[3]);
            define_enum_mapping(mapping_function, c_D_xy_Re, D_Re[4]);
            define_enum_mapping(mapping_function, c_D_xy_Im, D_Im[4]);
            define_enum_mapping(mapping_function, c_D_yz_Re, D_Re[5]);
            define_enum_mapping(mapping_function, c_D_yz_Im, D_Im[5]);
            define_enum_mapping(mapping_function, c_D_xz_Re, D_Re[6]);
            define_enum_mapping(mapping_function, c_D_xz_Im, D_Im[6]);
            define_enum_mapping(mapping_function, c_D_xyz_Re, D_Re[7]);
            define_enum_mapping(mapping_function, c_D_xyz_Im, D_Im[7]);
        }
    };

    //! the Dirac field needs no second derivatives
    template <class data_t> struct Diff2Vars
    {
        template <typename mapping_function_t> void enum_mapping(mapping_function_t) {}
    };

    //! RHS of the grid variables without the advection term beta^k d_k (added by the caller from its upwinded
    //! derivatives), the 3+1 sources of both spinors and their number density n = psi_A^dag psi_A + psi_B^dag psi_B
    //! (particle number = int n sqrt(gamma) d^3x). vars_t must contain the CCZ4 variables with gauge and Vars.
    template <class data_t, template <typename> class vars_t>
    void compute(Vars<data_t> &rhs_no_advec, emtensor_t<data_t> &emtensor, data_t &n, const vars_t<data_t> &vars,
                 const vars_t<Tensor<1, data_t>> &d1) const;

    // ---- the generic GRChombo matter interface (MatterCCZ4RHS, EMTensor, ...). DiracCCZ4RHS and DiracConstraints
    //      call compute() once instead of twice.
    template <class data_t, template <typename> class vars_t>
    emtensor_t<data_t> compute_emtensor(const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
                                        const Tensor<2, data_t> &h_UU, const Tensor<3, data_t> &chris_ULL) const
    {
        Vars<data_t> rhs;
        emtensor_t<data_t> out;
        data_t n;
        compute(rhs, out, n, vars, d1);
        return out;
    }

    template <class data_t, template <typename> class vars_t, template <typename> class diff2_vars_t,
              template <typename> class rhs_vars_t>
    void add_matter_rhs(rhs_vars_t<data_t> &total_rhs, const vars_t<data_t> &vars,
                        const vars_t<Tensor<1, data_t>> &d1, const diff2_vars_t<Tensor<2, data_t>> &d2,
                        const vars_t<data_t> &advec) const
    {
        Vars<data_t> rhs;
        emtensor_t<data_t> em;
        data_t n;
        compute(rhs, em, n, vars, d1);
        for (int k = 0; k < 8; ++k)
        {
            total_rhs.D_Re[k] = rhs.D_Re[k] + advec.D_Re[k];
            total_rhs.D_Im[k] = rhs.D_Im[k] + advec.D_Im[k];
        }
    }

    //! number density of both spinors (the D basis is orthonormal)
    template <class data_t, template <typename> class vars_t>
    static data_t number_density(const vars_t<data_t> &vars)
    {
        data_t n = 0.;
        for (int k = 0; k < 8; ++k)
            n += vars.D_Re[k] * vars.D_Re[k] + vars.D_Im[k] * vars.D_Im[k];
        return n;
    }

    //! number densities of the two spinors separately, n_A = psi_A^dag psi_A and n_B = psi_B^dag psi_B
    //! (n_A + n_B = number_density). NB the x and y reflections swap psi_A and psi_B, so n_A and n_B are each
    //! other's mirror images across such planes (their sum is even, their difference odd).
    template <class data_t, template <typename> class vars_t>
    static void spinor_number_densities(const vars_t<data_t> &vars, data_t &n_A, data_t &n_B)
    {
        using namespace DiracFieldHelpers;
        const double r2 = 1. / sqrt(2.);
        data_t P_re[8], P_im[8];
        for (int c = 0; c < 8; ++c)
        {
            P_re[c] = 0.;
            P_im[c] = 0.;
        }
        for (int m = 0; m < 8; ++m)
            for (int s = 0; s < 2; ++s)
            {
                const int c = dirac_comb_idx[m][s];
                const double w = r2 * dirac_comb_sgn[m][s];
                P_re[c] += w * vars.D_Re[m];
                P_im[c] += w * vars.D_Im[m];
            }
        n_A = 0.;
        n_B = 0.;
        for (int c = 0; c < 4; ++c)
        {
            n_A += P_re[c] * P_re[c] + P_im[c] * P_im[c];
            n_B += P_re[c + 4] * P_re[c + 4] + P_im[c + 4] * P_im[c + 4];
        }
    }

  protected:
    params_t m_params;
};

template <class data_t, template <typename> class vars_t>
void DiracField::compute(Vars<data_t> &rhs_no_advec, emtensor_t<data_t> &emtensor, data_t &n,
                         const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1) const
{
    using namespace DiracFieldHelpers;
    const Constants &C = constants();
    const data_t alpha = vars.lapse;
    const data_t chi = vars.chi;
    const double mu = m_params.mass;

    // ---------------------------------------------------------------- metric data
    data_t g[3][3], dg[3][3][3], K[3][3], gd[3][3], gi[3][3];
    data_t beta[3], dbeta[3][3], dalpha[3]; // dbeta[k][i] = d_k beta^i
    for (int i = 0; i < 3; ++i)
    {
        beta[i] = vars.shift[i];
        dalpha[i] = d1.lapse[i];
        for (int j = 0; j < 3; ++j)
        {
            g[i][j] = vars.h[i][j] / chi;
            gi[i][j] = 0.;
            K[i][j] = (vars.A[i][j] + vars.h[i][j] * vars.K / 3.) / chi;
            for (int k = 0; k < 3; ++k)
                dg[k][i][j] = (d1.h[i][j][k] - vars.h[i][j] * d1.chi[k] / chi) / chi;
        }
    }
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 3; ++i)
            dbeta[k][i] = d1.shift[i][k];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            gd[i][j] = -2. * alpha * K[i][j];
            for (int k = 0; k < 3; ++k)
                gd[i][j] += beta[k] * dg[k][i][j] + g[k][j] * dbeta[i][k] + g[i][k] * dbeta[j][k];
        }
    {
        const auto h_UU = TensorAlgebra::compute_inverse_sym(vars.h);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                gi[i][j] = chi * h_UU[i][j];
    }
    data_t G3[3][3][3]; // Gamma^i_jk of gamma
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
            {
                data_t c = 0.;
                for (int l = 0; l < 3; ++l)
                    c += gi[i][l] * (dg[j][l][k] + dg[k][l][j] - dg[l][j][k]);
                G3[i][j][k] = 0.5 * c;
            }

    // ---------------------------------------------------------------- Cholesky triad and its derivatives
    data_t L[3][3], Li[3][3], E[3][3], th[3][3], dE[3][3][3], Edot[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            L[i][j] = 0.;
            Li[i][j] = 0.;
        }
    L[0][0] = sqrt(g[0][0]);
    L[1][0] = g[1][0] / L[0][0];
    L[2][0] = g[2][0] / L[0][0];
    L[1][1] = sqrt(g[1][1] - L[1][0] * L[1][0]);
    L[2][1] = (g[2][1] - L[2][0] * L[1][0]) / L[1][1];
    L[2][2] = sqrt(g[2][2] - L[2][0] * L[2][0] - L[2][1] * L[2][1]);
    Li[0][0] = 1. / L[0][0];
    Li[1][1] = 1. / L[1][1];
    Li[2][2] = 1. / L[2][2];
    Li[1][0] = -L[1][0] * Li[0][0] * Li[1][1];
    Li[2][1] = -L[2][1] * Li[1][1] * Li[2][2];
    Li[2][0] = -(L[2][0] * Li[0][0] + L[2][1] * Li[1][0]) * Li[2][2];
    for (int i = 0; i < 3; ++i)
        for (int a = 0; a < 3; ++a)
        {
            E[i][a] = Li[a][i];  // E = L^{-T}
            th[a][i] = L[i][a];  // theta^a_i = L_ia
        }
    for (int dir = 0; dir < 4; ++dir) // x, y, z, t
    {
        data_t dgm[3][3], S[3][3], dL[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                dgm[i][j] = (dir < 3) ? dg[dir][i][j] : gd[i][j];
        // S = L^{-1} dgamma L^{-T}
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                data_t c = 0.;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        c += Li[i][k] * dgm[k][l] * Li[j][l];
                S[i][j] = c;
            }
        // dL = L Phi(S)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                data_t c = 0.;
                for (int k = j; k < 3; ++k) // Phi(S)_kj nonzero for k >= j
                    c += L[i][k] * ((k == j) ? data_t(0.5 * S[k][j]) : S[k][j]);
                dL[i][j] = c;
            }
        // dE = -E dL^T E
        for (int i = 0; i < 3; ++i)
            for (int a = 0; a < 3; ++a)
            {
                data_t c = 0.;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        c -= E[i][k] * dL[l][k] * E[l][a];
                if (dir < 3)
                    dE[dir][i][a] = c;
                else
                    Edot[i][a] = c;
            }
    }

    // ---------------------------------------------------------------- spin connection
    data_t om[3][3][3], om0[3][3], an0[3], anab[3][3];
    for (int k = 0; k < 3; ++k)
        for (int b = 0; b < 3; ++b)
        {
            data_t nab[3];
            for (int j = 0; j < 3; ++j)
            {
                nab[j] = dE[k][j][b];
                for (int l = 0; l < 3; ++l)
                    nab[j] += G3[j][k][l] * E[l][b];
            }
            for (int a = 0; a < 3; ++a)
            {
                data_t c = 0.;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        c += g[i][j] * E[i][a] * nab[j];
                om[k][a][b] = c;
            }
            data_t c0 = 0.;
            for (int j = 0; j < 3; ++j)
                c0 += K[k][j] * E[j][b];
            om0[k][b] = c0;
        }
    for (int b = 0; b < 3; ++b)
    {
        data_t c = 0.;
        for (int j = 0; j < 3; ++j)
            c -= E[j][b] * dalpha[j];
        an0[b] = c / alpha;
    }
    {
        data_t V[3][3], W[3][3];
        for (int i = 0; i < 3; ++i)
            for (int b = 0; b < 3; ++b)
            {
                data_t c = Edot[i][b];
                for (int l = 0; l < 3; ++l)
                    c -= beta[l] * dE[l][i][b] - E[l][b] * dbeta[l][i];
                V[i][b] = c;
            }
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
            {
                data_t c = 0.;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        c += E[i][a] * g[i][j] * V[j][b];
                W[a][b] = c;
            }
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                anab[a][b] = (W[a][b] - W[b][a]) / (2. * alpha);
    }

    // ---------------------------------------------------------------- the two spinors from the grid variables
    spinor<data_t> psi[2], dpsi[2][3];
    {
        const double r2 = 1. / sqrt(2.);
        cplx<data_t> P[8], dP[3][8];
        for (int c = 0; c < 8; ++c)
        {
            P[c] = {0., 0.};
            for (int k = 0; k < 3; ++k)
                dP[k][c] = {0., 0.};
        }
        // P = DIRAC_BASIS D: each D_m contributes to two spinor components
        for (int m = 0; m < 8; ++m)
            for (int s = 0; s < 2; ++s)
            {
                const int c = dirac_comb_idx[m][s];
                const double w = r2 * dirac_comb_sgn[m][s];
                P[c].re += w * vars.D_Re[m];
                P[c].im += w * vars.D_Im[m];
                for (int k = 0; k < 3; ++k)
                {
                    dP[k][c].re += w * d1.D_Re[m][k];
                    dP[k][c].im += w * d1.D_Im[m][k];
                }
            }
        for (int f = 0; f < 2; ++f)
            for (int c = 0; c < 4; ++c)
            {
                psi[f].c[c] = P[4 * f + c];
                for (int k = 0; k < 3; ++k)
                    dpsi[f][k].c[c] = dP[k][4 * f + c];
            }
    }

    // ---------------------------------------------------------------- RHS and stress tensor of each spinor
    data_t T[4][4];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            T[a][b] = 0.;
    spinor<data_t> rhs[2];
    for (int f = 0; f < 2; ++f)
    {
        const spinor<data_t> &p = psi[f];
        spinor<data_t> G0Gp[3], GGp[3][3];
        for (int b = 0; b < 3; ++b)
        {
            G0Gp[b] = apply(C.G0G[b], p);
            for (int a = 0; a < 3; ++a)
                if (a != b)
                    GGp[a][b] = apply(C.GG[a][b], p);
        }
        spinor<data_t> Gk_p[3], Gn_p = zero_spinor<data_t>();
        for (int k = 0; k < 3; ++k)
        {
            Gk_p[k] = zero_spinor<data_t>();
            for (int b = 0; b < 3; ++b)
            {
                Gk_p[k] = Gk_p[k] + data_t(0.5 * om0[k][b]) * G0Gp[b];
                for (int a = 0; a < 3; ++a)
                    if (a != b)
                        Gk_p[k] = Gk_p[k] + data_t(0.25 * om[k][a][b]) * GGp[a][b];
            }
        }
        for (int b = 0; b < 3; ++b)
        {
            Gn_p = Gn_p + data_t(0.5 * an0[b]) * G0Gp[b];
            for (int a = 0; a < 3; ++a)
                if (a != b)
                    Gn_p = Gn_p + data_t(0.25 * anab[a][b]) * GGp[a][b];
        }
        spinor<data_t> nab[4];
        for (int b = 0; b < 3; ++b)
        {
            nab[b + 1] = zero_spinor<data_t>();
            for (int k = 0; k < 3; ++k)
                nab[b + 1] = nab[b + 1] + E[k][b] * (dpsi[f][k] + Gk_p[k]);
        }
        spinor<data_t> r = (-1. * alpha) * Gn_p + (alpha * mu) * apply(C.G0, p);
        for (int b = 0; b < 3; ++b)
            r = r + alpha * apply(C.G0G[b], nab[b + 1]);
        rhs[f] = r;
        nab[0] = (1. / alpha) * r + Gn_p; // n^mu nabla_mu psi (the advection term cancels)
        for (int a = 0; a < 4; ++a)
            for (int b = a; b < 4; ++b)
                T[a][b] += 0.25 * (re_inner(p, C.BG[a], nab[b]) - re_inner(nab[a], C.BG[b], p) +
                                   re_inner(p, C.BG[b], nab[a]) - re_inner(nab[b], C.BG[a], p));
    }
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < a; ++b)
            T[a][b] = T[b][a];

    // back to the grid combinations: D_m = sum over its two components
    {
        const double r2 = 1. / sqrt(2.);
        for (int m = 0; m < 8; ++m)
        {
            rhs_no_advec.D_Re[m] = 0.;
            rhs_no_advec.D_Im[m] = 0.;
            for (int s = 0; s < 2; ++s)
            {
                const int c = dirac_comb_idx[m][s];
                const double w = r2 * dirac_comb_sgn[m][s];
                rhs_no_advec.D_Re[m] += w * rhs[c / 4].c[c % 4].re;
                rhs_no_advec.D_Im[m] += w * rhs[c / 4].c[c % 4].im;
            }
        }
    }

    // ---------------------------------------------------------------- 3+1 sources (coordinate components)
    emtensor.rho = T[0][0];
    for (int i = 0; i < 3; ++i)
    {
        data_t c = 0.;
        for (int b = 0; b < 3; ++b)
            c -= th[b][i] * T[b + 1][0];
        emtensor.Si[i] = c;
        for (int j = 0; j < 3; ++j)
        {
            data_t s = 0.;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    s += th[a][i] * th[b][j] * T[a + 1][b + 1];
            emtensor.Sij[i][j] = s;
        }
    }
    emtensor.S = 0.;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            emtensor.S += gi[i][j] * emtensor.Sij[i][j];
    n = number_density(vars);
}

#endif /* DIRACFIELD_HPP_ */
