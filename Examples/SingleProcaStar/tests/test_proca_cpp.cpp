// Standalone test of Source/Matter/ProcaField.hpp (no Chombo). One point per line on stdin:
//   chi h[6] (xx xy xz yy yz zz) K A[6] lapse shift[3]
//   for k = x, y, z: d_k chi, d_k h[6], d_k lapse, d_k shift[3]
//   X_Re[3] X_Im[3] E_Re[3] E_Im[3] Xphi_Re Xphi_Im Z_Re Z_Im, then for k = x, y, z: d_k of these 16
//   for j, k = x, y, z: d_j d_k X_Re[3], d_j d_k X_Im[3];   then mu kappa
// prints the 16 RHS components without advection (in the order above), rho, S_i[3], S_ij[6], S, n, C_Re, C_Im
#include <cstdio>
#include <iostream>
#include "ProcaField.hpp"

template <class data_t> struct TestVars : ProcaField::Vars<data_t>
{
    data_t chi, K, lapse;
    Tensor<2, data_t> h, A;
    Tensor<1, data_t> shift;
};

template <class T> void read_sym(T &M)
{
    std::cin >> M[0][0] >> M[0][1] >> M[0][2] >> M[1][1] >> M[1][2] >> M[2][2];
    M[1][0] = M[0][1];
    M[2][0] = M[0][2];
    M[2][1] = M[1][2];
}


int main()
{
    TestVars<double> v, rhs, advec;
    TestVars<Tensor<1, double>> d1;
    ProcaField::Diff2Vars<Tensor<2, double>> d2;
    double mu, kappa;
    while (std::cin >> v.chi)
    {
        read_sym(v.h);
        std::cin >> v.K;
        read_sym(v.A);
        std::cin >> v.lapse >> v.shift[0] >> v.shift[1] >> v.shift[2];
        for (int k = 0; k < 3; ++k)
        {
            double h6[6];
            std::cin >> d1.chi[k];
            for (double &x : h6)
                std::cin >> x;
            const int ii[6] = {0, 0, 0, 1, 1, 2}, jj[6] = {0, 1, 2, 1, 2, 2};
            for (int q = 0; q < 6; ++q)
                d1.h[ii[q]][jj[q]][k] = d1.h[jj[q]][ii[q]][k] = h6[q];
            std::cin >> d1.lapse[k] >> d1.shift[0][k] >> d1.shift[1][k] >> d1.shift[2][k];
        }
        for (int i = 0; i < 3; ++i) std::cin >> v.X_Re[i];
        for (int i = 0; i < 3; ++i) std::cin >> v.X_Im[i];
        for (int i = 0; i < 3; ++i) std::cin >> v.E_Re[i];
        for (int i = 0; i < 3; ++i) std::cin >> v.E_Im[i];
        std::cin >> v.Xphi_Re >> v.Xphi_Im >> v.Z_Re >> v.Z_Im;
        for (int k = 0; k < 3; ++k)
        {
            for (int i = 0; i < 3; ++i) std::cin >> d1.X_Re[i][k];
            for (int i = 0; i < 3; ++i) std::cin >> d1.X_Im[i][k];
            for (int i = 0; i < 3; ++i) std::cin >> d1.E_Re[i][k];
            for (int i = 0; i < 3; ++i) std::cin >> d1.E_Im[i][k];
            std::cin >> d1.Xphi_Re[k] >> d1.Xphi_Im[k] >> d1.Z_Re[k] >> d1.Z_Im[k];
        }
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
            {
                for (int i = 0; i < 3; ++i) std::cin >> d2.X_Re[i][j][k];
                for (int i = 0; i < 3; ++i) std::cin >> d2.X_Im[i][j][k];
            }
        std::cin >> mu >> kappa;

        for (int i = 0; i < 3; ++i)
            advec.X_Re[i] = advec.X_Im[i] = advec.E_Re[i] = advec.E_Im[i] = 0.;
        advec.Xphi_Re = advec.Xphi_Im = advec.Z_Re = advec.Z_Im = 0.;

        ProcaField proca({mu, kappa});
        proca.add_matter_rhs(rhs, v, d1, d2, advec);
        Tensor<2, double> h_UU;
        Tensor<3, double> chris;
        const emtensor_t<double> em = proca.compute_emtensor(v, d1, h_UU, chris);
        double C_Re, C_Im;
        proca.gauss_constraint(C_Re, C_Im, v, d1);
        for (int i = 0; i < 3; ++i) printf("%.17e ", rhs.X_Re[i]);
        for (int i = 0; i < 3; ++i) printf("%.17e ", rhs.X_Im[i]);
        for (int i = 0; i < 3; ++i) printf("%.17e ", rhs.E_Re[i]);
        for (int i = 0; i < 3; ++i) printf("%.17e ", rhs.E_Im[i]);
        printf("%.17e %.17e %.17e %.17e ", rhs.Xphi_Re, rhs.Xphi_Im, rhs.Z_Re, rhs.Z_Im);
        printf("%.17e %.17e %.17e %.17e ", em.rho, em.Si[0], em.Si[1], em.Si[2]);
        printf("%.17e %.17e %.17e %.17e %.17e %.17e ", em.Sij[0][0], em.Sij[0][1], em.Sij[0][2], em.Sij[1][1],
               em.Sij[1][2], em.Sij[2][2]);
        printf("%.17e %.17e %.17e %.17e\n", em.S, ProcaField::number_density(v), C_Re, C_Im);
    }
    return 0;
}
