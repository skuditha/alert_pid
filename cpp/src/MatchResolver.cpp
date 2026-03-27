#include "MatchResolver.h"

#include "AlertBanks.h"

namespace alert::postpid {

void MatchResolver::buildIndices(const AlertBanks& banks) {
    kftrack_row_by_trackid_.clear();
    hit_row_by_id_.clear();
    cluster_row_by_id_.clear();

    for (int row = 0; row < banks.kftrackRows(); ++row) {
        kftrack_row_by_trackid_[banks.getKfTrackId(row)] = row;
    }

    for (int row = 0; row < banks.hitRows(); ++row) {
        hit_row_by_id_[banks.getHitId(row)] = row;
    }

    for (int row = 0; row < banks.clusterRows(); ++row) {
        cluster_row_by_id_[banks.getClusterId(row)] = row;
    }
}

bool MatchResolver::resolveCandidate(
    int proj_row,
    const AlertBanks& banks,
    CandidateRefs& out) const
{
    out = CandidateRefs{};
    out.proj_row = proj_row;

    out.track_id = banks.getProjectionTrackId(proj_row);
    if (out.track_id == -1) {
        out.status = RowStatus::kInvalidTrackId;
        return false;
    }

    out.matched_atof_hit_id = banks.getProjectionMatchedAtofHitId(proj_row);
    if (out.matched_atof_hit_id == -1) {
        out.status = RowStatus::kInvalidMatchedAtofHitId;
        return false;
    }

    auto kt = kftrack_row_by_trackid_.find(out.track_id);
    if (kt == kftrack_row_by_trackid_.end()) {
        out.status = RowStatus::kMissingKftrackMatch;
        return false;
    }
    out.kftrack_row = kt->second;

    auto ht = hit_row_by_id_.find(out.matched_atof_hit_id);
    if (ht == hit_row_by_id_.end()) {
        out.status = RowStatus::kMissingAtofHitMatch;
        return false;
    }
    out.hit_row = ht->second;

    out.cluster_id = banks.getHitClusterId(out.hit_row);
    if (out.cluster_id == -1) {
        out.status = RowStatus::kInvalidClusterId;
        return false;
    }

    auto ct = cluster_row_by_id_.find(out.cluster_id);
    if (ct == cluster_row_by_id_.end()) {
        out.status = RowStatus::kMissingAtofClusterMatch;
        return false;
    }
    out.cluster_row = ct->second;

    out.status = RowStatus::kScoredNoMaskedFeatures;
    return true;
}

}  // namespace alert::postpid