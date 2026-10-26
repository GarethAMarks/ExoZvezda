// Standalone test of Source/Matter/DiracField.hpp (no Chombo). One point per line on stdin:
//   chi h[6] (xx xy xz yy yz zz) K A[6] lapse shift[3]
//   for k = x, y, z: d_k chi, d_k h[6], d_k lapse, d_k shift[3]
//   D_Re[8] D_Im[8], then for k = x, y, z: d_k D_Re[8], d_k D_Im[8], then mu
// prints the 16 RHS components (no advection; Re, Im per variable), rho, S_i[3], S_ij[6], S, n
#include <cstdio>
#include <iostream>
#include "DiracField.hpp"

template <class data_t> struct TestVars : DiracField::Vars<data_t>
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
    TestVars<double> v;
    TestVars<Tensor<1, double>> d1;
    double mu;
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
        for (int m = 0; m < 8; ++m)
            std::cin >> v.D_Re[m];
        for (int m = 0; m < 8; ++m)
            std::cin >> v.D_Im[m];
        for (int k = 0; k < 3; ++k)
        {
            for (int m = 0; m < 8; ++m)
                std::cin >> d1.D_Re[m][k];
            for (int m = 0; m < 8; ++m)
                std::cin >> d1.D_Im[m][k];
        }
        std::cin >> mu;
        DiracField dirac({mu});
        DiracField::Vars<double> rhs;
        emtensor_t<double> em;
        double n;
        dirac.compute(rhs, em, n, v, d1);
        for (int m = 0; m < 8; ++m)
            printf("%.17e %.17e ", rhs.D_Re[m], rhs.D_Im[m]);
        printf("%.17e %.17e %.17e %.17e ", em.rho, em.Si[0], em.Si[1], em.Si[2]);
        printf("%.17e %.17e %.17e %.17e %.17e %.17e ", em.Sij[0][0], em.Sij[0][1], em.Sij[0][2], em.Sij[1][1],
               em.Sij[1][2], em.Sij[2][2]);
        printf("%.17e %.17e\n", em.S, n);
    }
    return 0;
}
