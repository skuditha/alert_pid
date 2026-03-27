#pragma once

#include <map>
#include <string>

#include "Types.h"

namespace alert::postpid {

class AlertBanks;
class Cutflow;

class TruthLabeler {
public:
    TruthLabeler();

    bool loadLabelMap(const std::string& path);
    bool extractTruth(const AlertBanks& banks, TruthInfo& out, Cutflow& cutflow) const;

private:
    std::map<int, std::string> pid_to_class_;
    std::map<int, int> pid_to_index_;
};

}  // namespace alert::postpid