#pragma once

#include <unordered_map>

#include "Types.h"

namespace alert::postpid {

class AlertBanks;

class MatchResolver {
public:
    void buildIndices(const AlertBanks& banks);

    bool resolveCandidate(
        int proj_row,
        const AlertBanks& banks,
        CandidateRefs& out) const;

private:
    std::unordered_map<int, int> kftrack_row_by_trackid_;
    std::unordered_map<int, int> hit_row_by_id_;
    std::unordered_map<int, int> cluster_row_by_id_;
};

}  // namespace alert::postpid