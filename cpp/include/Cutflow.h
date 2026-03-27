#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace alert::postpid {

class Cutflow {
public:
    void increment(const std::string& key, int64_t delta = 1);
    int64_t get(const std::string& key) const;
    const std::map<std::string, int64_t>& counters() const;

    void printSummary() const;

private:
    std::map<std::string, int64_t> counters_;
};

}  // namespace alert::postpid