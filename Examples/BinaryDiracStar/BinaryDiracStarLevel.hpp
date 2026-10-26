/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef BINARYDIRACSTARLEVEL_HPP_
#define BINARYDIRACSTARLEVEL_HPP_

#include "DefaultLevelFactory.hpp"
#include "GRAMRLevel.hpp"
#include "STAMR.hpp"

/*!
     Evolves a binary of Dirac stars (two Dirac spinors, see Source/Matter/DiracField.hpp) with the CCZ4 equations: the Dirac
     analogue of the BBS*Level classes.
*/
class BinaryDiracStarLevel : public GRAMRLevel
{
    friend class DefaultLevelFactory<BinaryDiracStarLevel>;
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

    //! constraints and the Dirac diagnostics (N, mod_psi, rho)
    void computeDiagnostics();
};

#endif /* BINARYDIRACSTARLEVEL_HPP_ */
