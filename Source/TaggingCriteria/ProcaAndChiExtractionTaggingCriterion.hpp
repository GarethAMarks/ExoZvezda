/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef PROCAANDCHIEXTRACTIONTAGGINGCRITERION_HPP_
#define PROCAANDCHIEXTRACTIONTAGGINGCRITERION_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "FourthOrderDerivatives.hpp"
#include "SimulationParametersBase.hpp"
#include "Tensor.hpp"

/// Tagging based on derivatives of X_phi = -n^a A_a (the Proca analogue of the boson stars' phi) and of chi, as
/// ComplexPhiAndChiExtractionTaggingCriterion does for the complex scalar field, plus the extraction spheres
class ProcaAndChiExtractionTaggingCriterion
{
  protected:
    const double m_dx;
    const FourthOrderDerivatives m_deriv;
    const bool m_activate_extraction;
    const extraction_params_t m_params;
    const int m_level;
    const double m_threshold_Xphi;
    const double m_threshold_chi;

    template <class data_t> struct MatterVars
    {
        data_t Xphi_Re, Xphi_Im;
        template <typename mapping_function_t> void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_enum_mapping(mapping_function, c_Xphi_Re, Xphi_Re);
            VarsTools::define_enum_mapping(mapping_function, c_Xphi_Im, Xphi_Im);
        }
    };

    template <class data_t> struct Vars
    {
        data_t chi;
        template <typename mapping_function_t> void enum_mapping(mapping_function_t mapping_function)
        {
            VarsTools::define_enum_mapping(mapping_function, c_chi, chi);
        }
    };

  public:
    ProcaAndChiExtractionTaggingCriterion(const double a_dx, const int a_level, const extraction_params_t a_params,
                                          const double a_threshold_Xphi, const double a_threshold_chi,
                                          const bool activate_extraction = false)
        : m_dx(a_dx), m_deriv(a_dx), m_activate_extraction(activate_extraction), m_params(a_params),
          m_level(a_level), m_threshold_Xphi(a_threshold_Xphi), m_threshold_chi(a_threshold_chi){};

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        const auto d1 = m_deriv.template diff1<MatterVars>(current_cell);
        const auto d2 = m_deriv.template diff2<MatterVars>(current_cell);
        const auto d1chi = m_deriv.template diff1<Vars>(current_cell);
        const auto d2chi = m_deriv.template diff2<Vars>(current_cell);

        data_t d2_Xphi_ratio = 0.;
        data_t d2_chi_ratio = 0.;
        FOR2(idir, jdir)
        {
            data_t mod_d2 = d2.Xphi_Re[idir][jdir] * d2.Xphi_Re[idir][jdir] +
                            d2.Xphi_Im[idir][jdir] * d2.Xphi_Im[idir][jdir];
            data_t abs_d1d1 = abs(d1.Xphi_Re[idir] * d1.Xphi_Re[jdir] + d1.Xphi_Im[idir] * d1.Xphi_Im[jdir]);
            d2_Xphi_ratio += mod_d2 / (abs_d1d1 + 1e-5);
            d2_chi_ratio +=
                d2chi.chi[idir][jdir] * d2chi.chi[idir][jdir] / (1e-2 + abs(d1chi.chi[idir] * d1chi.chi[jdir]));
        }
        data_t criterion_Xphi = m_dx * sqrt(d2_Xphi_ratio) / m_threshold_Xphi;
        data_t criterion_chi = m_dx * sqrt(d2_chi_ratio) / m_threshold_chi;
        data_t criterion = simd_max(criterion_chi, criterion_Xphi);

        if (m_activate_extraction)
        {
            for (int iradius = 0; iradius < m_params.num_extraction_radii; ++iradius)
            {
                if (m_level < m_params.extraction_levels[iradius])
                {
                    const Coordinates<data_t> coords(current_cell, m_dx, m_params.extraction_center);
                    const data_t r = coords.get_radius();
                    auto regrid = simd_compare_lt(r, 1.2 * m_params.extraction_radii[iradius]);
                    criterion = simd_conditional(regrid, 100.0, criterion);
                }
            }
        }
        current_cell.store_vars(criterion, 0);
    }
};

#endif /* PROCAANDCHIEXTRACTIONTAGGINGCRITERION_HPP_ */
