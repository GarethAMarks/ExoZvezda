/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef ASPHERICALPERTURBATION_HPP_
#define ASPHERICALPERTURBATION_HPP_

#include <cmath>
#include <string>

/*
 Aspherical perturbations of a star, with the conventions of G. A. Marks, "Stability and formation of solitonic boson
 stars", arXiv:2609.30021, Eq. (11). Every matter field variable is multiplied by

     F = exp(-2 delta cos(m phi)) exp(-2 epsilon cos(n phi')),

 where phi = atan2(x_b, x_a) is the azimuthal angle about the star centre with respect to a pair of orthogonal
 reference axes (x_a, x_b) = (axes[0], axes[1]). The second factor is optional: it is a second perturbation prepared in
 an orthogonal plane (axes2), giving a configuration with no symmetry axis left (the "two-axis" runs of the paper). It
 is only available in 3D. In the 2D (cartoon) codes the reference axes are the symmetry axis and the cylindrical radius,
 so phi is the polar angle from the symmetry axis and the perturbation is axisymmetric.

 The variables multiplied by F are, for the Dirac stars, all spinor components and, for the Proca stars, X_phi, X_i
 and E^i (for the scalar field of the paper, phi and Pi). The constraints are not re-solved, as in the paper, so the
 Hamiltonian constraint is violated at O(delta). F depends only on the angles about the star centre, so for a Proca star
 at rest, whose E^i is radial, x^i d_i F = 0 and the Gauss constraint D_i E^i + mu^2 X_phi = 0 still holds wherever F
 is smooth.

 Index 0 is reserved for the spherical perturbation of the 1D solvers (Gaussian, with the constraints re-solved), so in
 this generalised formalism m = 0 means "spherical" and a factor with index 0 here is the identity.

 NB: phi is undefined on the line through the centre normal to the (x_a, x_b) plane (in the axisymmetric 2D case, only
 at the centre), and F jumps there by up to exp(4 delta). Any field that does not vanish there (the scalar field of the
 paper, the Dirac g, the Proca X_phi) therefore becomes discontinuous at that line or point. This is inherent in the
 prescription of Eq. (11). Next to the line, the gradient energy of the perturbed fields grows like 1/rho^2 (rho the
 distance from the line), so the constraint violations there (including Gauss for the Proca field) are O(delta) or
 larger and grow with resolution.

 Smoothed version (smooth = true): cos(m phi) is replaced by cos(m phi) sin^m(theta) = Re[(x_a + i x_b)^m] / r^m,
 theta the angle from that line (sin theta = rho / r). This is the angular dependence of the sectoral harmonics
 Y_mm, and it vanishes on the line, so F is smooth everywhere except at the centre itself. There F remains direction
 dependent, a point discontinuity with finite gradient energy. In the plane of the reference axes (theta = pi/2) the
 smoothed and unsmoothed factors agree. In particular, on the cartoon plane of the 2D codes they are identical, and
 there the only singular point is the centre anyway.

 Chombo-free and shared by the Dirac and Proca stars of 2DChombo and ExoZvezda.
*/
struct AsphericalPerturbation
{
    bool active = false; //!< false disables the aspherical perturbations altogether (F = 1)
    int m = 0;            //!< azimuthal index of the first perturbation (0: none here)
    double delta = 0.;    //!< its magnitude
    int axes[2] = {0, 1}; //!< reference axes (x_a, x_b) of phi = atan2(x_b, x_a): 0 = x, 1 = y, 2 = z
    int n = 0;            //!< azimuthal index of the second perturbation (0: none; 3D only)
    double epsilon = 0.;  //!< its magnitude
    int axes2[2] = {0, 2};//!< its reference axes (must span a plane orthogonal to the first one)
    bool smooth = false;  //!< replace cos(m phi) by cos(m phi) sin^m(theta), smooth off the centre (see above)

    //! cos(m phi), or cos(m phi) sin^m(theta) if smooth, for the plane of the reference axes ax
    double angular(int k, const int *ax, const double *c) const
    {
        const double phi = atan2(c[ax[1]], c[ax[0]]);
        if (!smooth)
            return cos(k * phi);
        const double r = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        if (r == 0.)
            return 0.; // the direction-dependent centre: take the average over directions
        return cos(k * phi) * pow(sqrt(c[ax[0]] * c[ax[0]] + c[ax[1]] * c[ax[1]]) / r, k);
    }

    //! F at the position (x, y, z) relative to the star centre (z is ignored, and should be 0, in 2D)
    double factor(double x, double y, double z) const
    {
        if (!active)
            return 1.;
        const double c[3] = {x, y, z};
        double F = 1.;
        if (m != 0 && delta != 0.)
            F *= exp(-2. * delta * angular(m, axes, c));
        if (n != 0 && epsilon != 0.)
            F *= exp(-2. * epsilon * angular(n, axes2, c));
        return F;
    }

    //! whether F is even under the reflection x_dir -> -x_dir about the star centre. phi = atan2(x_b, x_a) goes to
    //! -phi if dir = b (fine) and to pi - phi if dir = a, which flips the sign of cos(m phi) for odd m (sin(theta) is
    //! even under all three reflections, so the same holds for the smoothed version).
    bool even_under_reflection(int dir) const
    {
        if (!active)
            return true;
        if (m != 0 && delta != 0. && axes[0] == dir && m % 2 != 0)
            return false;
        if (n != 0 && epsilon != 0. && axes2[0] == dir && n % 2 != 0)
            return false;
        return true;
    }

    //! an error message if the parameters are inconsistent, empty otherwise
    std::string check(int dim) const
    {
        if (!active)
            return "";
        auto valid = [dim](const int *ax) { return ax[0] >= 0 && ax[0] < dim && ax[1] >= 0 && ax[1] < dim && ax[0] != ax[1]; };
        if (m < 0 || n < 0)
            return "the azimuthal indices m and n must be non-negative";
        if (!valid(axes))
            return "the reference axes must be two different directions (0 = x, 1 = y, 2 = z)";
        if (n != 0)
        {
            if (dim < 3)
                return "the second (two-axis) perturbation needs 3D";
            if (!valid(axes2))
                return "the second reference axes must be two different directions (0 = x, 1 = y, 2 = z)";
            const int shared = (axes2[0] == axes[0] || axes2[0] == axes[1]) + (axes2[1] == axes[0] || axes2[1] == axes[1]);
            if (shared != 1)
                return "the two perturbations must be prepared in orthogonal planes (the axis pairs must share one axis)";
        }
        return "";
    }
};

#endif /* ASPHERICALPERTURBATION_HPP_ */
