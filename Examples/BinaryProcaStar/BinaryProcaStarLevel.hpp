/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef BINARYPROCASTARLEVEL_HPP_
#define BINARYPROCASTARLEVEL_HPP_

#include "DefaultLevelFactory.hpp"
#include "GRAMRLevel.hpp"
#include "STAMR.hpp"

/*!
     Evolves a Proca star binary (a complex Proca field, see Source/Matter/ProcaField.hpp) with the CCZ4 equations: the
     Proca analogue of the boson-star binary levels.
*/
class BinaryProcaStarLevel : public GRAMRLevel
{
    friend class DefaultLevelFactory<BinaryProcaStarLevel>;
    // Inherit the contructors from GRAMRLevel
    using GRAMRLevel::GRAMRLevel;

    STAMR &m_st_amr = dynamic_cast<STAMR &>(m_gr_amr);

    virtual void specificAdvance() override;
    virtual void initialData() override;
    virtual void preCheckpointLevel() override;
    virtual void prePlotLevel() override;
    virtual void specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs, const double a_time) override;
    virtual void specificUpdateODE(GRLevelData &a_soln, const GRLevelData &a_rhs, Real a_dt) override;
    virtual void computeTaggingCriterion(FArrayBox &tagging_criterion, const FArrayBox &current_state,
                                         const FArrayBox &current_state_diagnostics) override;
    virtual void specificPostTimeStep() override;

    //! constraints and the Proca diagnostics (N, mod_Xphi, rho, n, Gauss)
    void computeDiagnostics();
};

#endif /* BINARYPROCASTARLEVEL_HPP_ */
