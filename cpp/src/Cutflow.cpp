#include "Cutflow.h"

#include <iostream>

namespace alert::postpid {

void Cutflow::increment(const std::string& key, int64_t delta) {
    counters_[key] += delta;
}

int64_t Cutflow::get(const std::string& key) const {
    auto it = counters_.find(key);
    if (it == counters_.end()) {
        return 0;
    }
    return it->second;
}

const std::map<std::string, int64_t>& Cutflow::counters() const {
    return counters_;
}

void Cutflow::printSummary() const {
    std::cout << "\n=== Cutflow Summary ===\n";
    for (const auto& [key, value] : counters_) {
        std::cout << key << ": " << value << "\n";
    }
    std::cout << "=======================\n";
}

}  // namespace alert::postpid