#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Cutflow.h"
#include "Types.h"

namespace alert::postpid {

class DataWriter {
public:
    DataWriter();
    ~DataWriter();

    bool open(const std::string& output_path, std::size_t n_rows);
    bool writeRow(std::size_t index, const FeatureRow& features, const OutputRowMeta& meta);
    bool writeMetadata(
        const std::vector<std::string>& input_files,
        const Cutflow& cutflow,
        const std::string& label_map_version,
        const std::string& feature_contract_version);
    bool close();

private:
    class Impl;
    Impl* impl_ = nullptr;
};

}  // namespace alert::postpid