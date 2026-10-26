/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "BinaryDiracStarLevel.hpp"
#include "BoxLoops.hpp"
#include "GammaCalculator.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "TraceARemoval.hpp"

// Dirac field
#include "DiracCCZ4RHS.hpp"
#include "DiracConstraints.hpp"
#include "DiracField.hpp"

// Tagging
#include "DiracAndChiExtractionTaggingCriterion.hpp"

// Initial data
#include "ComputePack.hpp"
#include "SetValue.hpp"
#include "BinaryDiracStar.hpp"

// Diagnostics
#include "ADMMass.hpp"
#include "ADMMassExtraction.hpp"
#include "AMRReductions.hpp"
#include "SmallDataIO.hpp"

void BinaryDiracStarLevel::specificAdvance()
{
    // Enforce the trace free A_ij condition and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()), m_state_new, m_state_new,
                   INCLUDE_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(m_dx, m_p.center, "NaNCheck in specific Advance"), m_state_new, m_state_new,
                       EXCLUDE_GHOST_CELLS, disable_simd());
}

void BinaryDiracStarLevel::initialData()
{
    CH_TIME("BinaryDiracStarLevel::initialData");
    if (m_verbosity)
        pout() << "BinaryDiracStarLevel::initialData " << m_level << endl;

    BinaryDiracStar dirac_binary(m_p.binary_params, m_dx, m_verbosity);
    dirac_binary.compute_1d_solution();

    if (m_level == 0)
    {
        for (const DiracStarSolution *s : {&dirac_binary.m_1d_sol, &dirac_binary.m_1d_sol2})
            pout() << "Dirac star has A_central " << s->A_central << " omega " << s->omega << " mass " << s->M
                   << " particle number " << s->noether_charge << " binding energy " << s->binding_energy
                   << " r_99 " << s->r_99 << endl;
    }

    BoxLoops::loop(make_compute_pack(SetValue(0.0), dirac_binary), m_state_new, m_state_new, INCLUDE_GHOST_CELLS,
                   disable_simd());

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new, EXCLUDE_GHOST_CELLS, disable_simd());
}

void BinaryDiracStarLevel::computeDiagnostics()
{
    DiracField dirac_field(m_p.dirac_params);
    BoxLoops::loop(DiracConstraints(dirac_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom1, c_Mom3), c_N,
                                    c_mod_psi, c_rho),
                   m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}

void BinaryDiracStarLevel::preCheckpointLevel()
{
    CH_TIME("BinaryDiracStarLevel::preCheckpointLevel");
    fillAllGhosts();
    computeDiagnostics();
}

void BinaryDiracStarLevel::prePlotLevel()
{
    CH_TIME("BinaryDiracStarLevel::prePlotLevel");
    fillAllGhosts();
    computeDiagnostics();
}

void BinaryDiracStarLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs, const double a_time)
{
    // Enforce trace free A_ij and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()), a_soln, a_soln, INCLUDE_GHOST_CELLS);

    // Calculate CCZ4 right hand side with the Dirac field
    DiracField dirac_field(m_p.dirac_params);
    DiracCCZ4RHS<MovingPunctureGauge, FourthOrderDerivatives> my_ccz4_dirac(dirac_field, m_p.ccz4_params, m_dx,
                                                                           m_p.sigma, m_p.formulation, m_p.G_Newton);
    BoxLoops::loop(my_ccz4_dirac, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
}

void BinaryDiracStarLevel::specificUpdateODE(GRLevelData &a_soln, const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce the trace free A_ij condition
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void BinaryDiracStarLevel::specificPostTimeStep()
{
    CH_TIME("BinaryDiracStarLevel::specificPostTimeStep");

    bool first_step = (m_time == 0.0);

    fillAllGhosts();

    BoxLoops::loop(ADMMass(m_p.center, m_dx), m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
    computeDiagnostics();

    if (m_p.activate_mass_extraction == 1 &&
        at_level_timestep_multiple(m_p.mass_extraction_params.min_extraction_level()))
    {
        if (m_verbosity)
            pout() << "BinaryDiracStarLevel::specificPostTimeStep: Extracting mass." << endl;

        // Now refresh the interpolator and do the interpolation
        if (m_level == m_p.mass_extraction_params.min_extraction_level())
        {
            m_gr_amr.m_interpolator->refresh();
            ADMMassExtraction mass_extraction(m_p.mass_extraction_params, m_dt, m_time, first_step,
                                              m_restart_time);
            mass_extraction.execute_query(m_gr_amr.m_interpolator);
        }
    }

    if (m_level == 0)
    {
        AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
        AMRReductions<VariableType::evolution> amr_reductions_ev(m_gr_amr);

        if (m_p.calculate_particle_number)
        {
            double particle_number = amr_reductions.sum(c_N);
            SmallDataIO N_file("ParticleNumber", m_dt, m_time, m_restart_time, SmallDataIO::APPEND, first_step);
            N_file.remove_duplicate_time_data();
            if (first_step)
                N_file.write_header_line({"N (both spinors)"});
            N_file.write_time_data_line({particle_number});
        }

        // maximum of sqrt(2 pi n): the central amplitude A_central of the 1D solver for a star at rest
        double mod_psi_max = amr_reductions.max(c_mod_psi);
        SmallDataIO mod_psi_max_file("mod_psi_max", m_dt, m_time, m_restart_time, SmallDataIO::APPEND, first_step);
        mod_psi_max_file.remove_duplicate_time_data();
        if (first_step)
            mod_psi_max_file.write_header_line({"max mod psi"});
        mod_psi_max_file.write_time_data_line({mod_psi_max});

        double min_chi = amr_reductions_ev.min(c_chi);
        SmallDataIO min_chi_file("min_chi", m_dt, m_time, m_restart_time, SmallDataIO::APPEND, first_step);
        min_chi_file.remove_duplicate_time_data();
        if (first_step)
            min_chi_file.write_header_line({"min chi"});
        min_chi_file.write_time_data_line({min_chi});

        double L2_Ham = amr_reductions.norm(c_Ham, 2, true);
        double L2_Mom = amr_reductions.norm(Interval(c_Mom1, c_Mom3), 2, true);
        double L1_Ham = amr_reductions.norm(c_Ham, 1, true);
        double L1_Mom = amr_reductions.norm(Interval(c_Mom1, c_Mom3), 1, true);
        SmallDataIO constraints_file("constraint_norms", m_dt, m_time, m_restart_time, SmallDataIO::APPEND,
                                     first_step);
        constraints_file.remove_duplicate_time_data();
        if (first_step)
            constraints_file.write_header_line({"L^2_Ham", "L^2_Mom", "L^1_Ham", "L^1_Mom"});
        constraints_file.write_time_data_line({L2_Ham, L2_Mom, L1_Ham, L1_Mom});
    }

#ifdef USE_AHFINDER
    if (m_p.AH_activate && m_level == m_p.AH_params.level_to_run)
        m_st_amr.m_ah_finder.solve(m_dt, m_time, m_restart_time);
#endif
}

void BinaryDiracStarLevel::computeTaggingCriterion(FArrayBox &tagging_criterion, const FArrayBox &current_state,
                                                   const FArrayBox &current_state_diagnostics)
{
    BoxLoops::loop(DiracAndChiExtractionTaggingCriterion(m_dx, m_level, m_p.extraction_params,
                                                         m_p.regrid_threshold_psi, m_p.regrid_threshold_chi,
                                                         m_p.activate_extraction),
                   current_state, tagging_criterion);
}
