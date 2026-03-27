#pragma once

#include "Types.h"

namespace alert::postpid {

class AlertBanks;
class MatchResolver;
class Cutflow;

class EventSelector {
public:
    bool hasRequiredEventBanks(const AlertBanks& banks, Cutflow& cutflow) const;

    bool hasAtLeastOneEligibleRow(
        const AlertBanks& banks,
        const MatchResolver& resolver,
        Cutflow& cutflow) const;

    bool isRowEligible(
        int proj_row,
        const AlertBanks& banks,
        const MatchResolver& resolver,
        CandidateRefs& refs,
        Cutflow& cutflow) const;
};

}  // namespace alert::postpid