#include "EventSelector.h"

#include "AlertBanks.h"
#include "Cutflow.h"
#include "MatchResolver.h"

namespace alert::postpid {

bool EventSelector::hasRequiredEventBanks(const AlertBanks& banks, Cutflow& cutflow) const {
    if (!banks.hasMC()) {
        cutflow.increment("missing_mc_particle_bank");
        return false;
    }

    if (!banks.hasProjections()) {
        cutflow.increment("missing_alert_ai_projections_bank");
        return false;
    }

    return true;
}

bool EventSelector::hasAtLeastOneEligibleRow(
    const AlertBanks& banks,
    const MatchResolver& resolver,
    Cutflow& cutflow) const
{
    for (int proj_row = 0; proj_row < banks.projectionsRows(); ++proj_row) {
        CandidateRefs refs;
        if (resolver.resolveCandidate(proj_row, banks, refs)) {
            return true;
        }
    }

    cutflow.increment("no_row_level_eligible_candidates");
    return false;
}

bool EventSelector::isRowEligible(
    int proj_row,
    const AlertBanks& banks,
    const MatchResolver& resolver,
    CandidateRefs& refs,
    Cutflow& cutflow) const
{
    if (!resolver.resolveCandidate(proj_row, banks, refs)) {
        switch (refs.status) {
            case RowStatus::kInvalidTrackId:
                cutflow.increment("invalid_track_id");
                break;
            case RowStatus::kInvalidMatchedAtofHitId:
                cutflow.increment("invalid_matched_atof_hit_id");
                break;
            case RowStatus::kMissingKftrackMatch:
                cutflow.increment("missing_kftrack_match");
                break;
            case RowStatus::kMissingAtofHitMatch:
                cutflow.increment("missing_atof_hit_match");
                break;
            case RowStatus::kInvalidClusterId:
                cutflow.increment("invalid_cluster_id");
                break;
            case RowStatus::kMissingAtofClusterMatch:
                cutflow.increment("missing_atof_cluster_match");
                break;
            default:
                break;
        }
        return false;
    }

    return true;
}

}  // namespace alert::postpid