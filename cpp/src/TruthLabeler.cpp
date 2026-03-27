#include "TruthLabeler.h"

#include <fstream>
#include <regex>
#include <sstream>

#include "AlertBanks.h"
#include "Cutflow.h"

namespace alert::postpid {

TruthLabeler::TruthLabeler() = default;

bool TruthLabeler::loadLabelMap(const std::string& path) {
    pid_to_class_.clear();
    pid_to_index_.clear();

    std::ifstream in(path);
    if (!in) {
        return false;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    std::regex entry_regex(
        R"REGEX("([^"]+)"\s*:\s*\{[^{}]*"pid"\s*:\s*(-?\d+)\s*,\s*"index"\s*:\s*(\d+)[^{}]*\})REGEX");

    auto begin = std::sregex_iterator(text.begin(), text.end(), entry_regex);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        const std::smatch& m = *it;
        const std::string class_name = m[1].str();
        const int pid = std::stoi(m[2].str());
        const int index = std::stoi(m[3].str());

        pid_to_class_[pid] = class_name;
        pid_to_index_[pid] = index;
    }

    if (pid_to_class_.empty()) {
        pid_to_class_[2212] = "proton";
        pid_to_class_[45]   = "deuteron";
        pid_to_class_[46]   = "triton";
        pid_to_class_[49]   = "helium3";
        pid_to_class_[47]   = "helium4";

        pid_to_index_[2212] = 0;
        pid_to_index_[45]   = 1;
        pid_to_index_[46]   = 2;
        pid_to_index_[49]   = 3;
        pid_to_index_[47]   = 4;
    }

    return true;
}

bool TruthLabeler::extractTruth(const AlertBanks& banks, TruthInfo& out, Cutflow& cutflow) const {
    out = TruthInfo{};

    if (!banks.hasMC()) {
        cutflow.increment("missing_mc_particle_bank");
        return false;
    }

    int supported_count = 0;
    int chosen_pid = 0;

    for (int row = 0; row < banks.mcRows(); ++row) {
        const int pid = banks.getMcPid(row);
        if (pid_to_index_.count(pid) > 0) {
            ++supported_count;
            chosen_pid = pid;
        }
    }

    if (supported_count == 0) {
        cutflow.increment("no_supported_truth_particle");
        return false;
    }

    if (supported_count > 1) {
        cutflow.increment("multiple_supported_truth_particles");
        return false;
    }

    out.valid = true;
    out.pid = chosen_pid;
    out.class_index = pid_to_index_.at(chosen_pid);
    out.class_name = pid_to_class_.at(chosen_pid);
    return true;
}

}  // namespace alert::postpid