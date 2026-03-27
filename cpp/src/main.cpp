#include <iostream>
#include <string>
#include <vector>

#include "AlertBanks.h"
#include "Cutflow.h"
#include "DataWriter.h"
#include "EventSelector.h"
#include "FeatureBuilder.h"
#include "MatchResolver.h"
#include "TruthLabeler.h"
#include "Types.h"
#include "RowQualityEvaluator.h"

// Include real HIPO headers in implementation build.
#include "reader.h"
#include "event.h"

using namespace alert::postpid;

namespace {

struct Config {
    std::vector<std::string> input_files;
    std::string output_h5;
    std::string label_map_path;
};

bool ParseArgs(int argc, char** argv, Config& cfg) {
    if (argc < 4) {
        std::cerr << "Usage: main_extract --label-map label_map.json --output out.h5 input1.hipo [input2.hipo ...]\n";
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--label-map" && i + 1 < argc) {
            cfg.label_map_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_h5 = argv[++i];
        } else {
            cfg.input_files.push_back(arg);
        }
    }

    return !cfg.label_map_path.empty() &&
           !cfg.output_h5.empty() &&
           !cfg.input_files.empty();
}

std::size_t CountEligibleRows(
    const Config& cfg,
    TruthLabeler& truth_labeler,
    Cutflow& cutflow)
{
    std::size_t total_rows = 0;

    for (const auto& path : cfg.input_files) {
        hipo::reader reader;
        reader.open(path.c_str());

        AlertBanks banks;
        if (!banks.initialize(reader)) {
            std::cerr << "Failed to initialize banks for file: " << path << "\n";
            continue;
        }

        hipo::event event;
        int64_t event_index = 0;

        while (reader.next()) {
            reader.read(event);
            ++event_index;
            cutflow.increment("total_events_seen");

            if (!banks.loadEvent(event)) {
                cutflow.increment("event_load_failure");
                continue;
            }

            TruthInfo truth;
            if (!truth_labeler.extractTruth(banks, truth, cutflow)) {
                continue;
            }

            EventSelector selector;
            if (!selector.hasRequiredEventBanks(banks, cutflow)) {
                continue;
            }

            MatchResolver resolver;
            resolver.buildIndices(banks);

            if (!selector.hasAtLeastOneEligibleRow(banks, resolver, cutflow)) {
                continue;
            }

            FeatureBuilder feature_builder;
            RowQualityEvaluator quality_evaluator;

            for (int proj_row = 0; proj_row < banks.projectionsRows(); ++proj_row) {
                CandidateRefs refs;
                if (!selector.isRowEligible(proj_row, banks, resolver, refs, cutflow)) {
                    continue;
                }

                FeatureRow features = feature_builder.build(banks, refs);

                if (!quality_evaluator.isRowQualityAcceptable(features, refs, cutflow)) {
                    continue;
                }

                ++total_rows;
            }
        }
    }

    return total_rows;
}

bool WriteDataset(
    const Config& cfg,
    TruthLabeler& truth_labeler,
    DataWriter& writer,
    Cutflow& cutflow)
{
    std::size_t row_index = 0;

    for (const auto& path : cfg.input_files) {
        hipo::reader reader;
        reader.open(path.c_str());

        AlertBanks banks;
        if (!banks.initialize(reader)) {
            std::cerr << "Failed to initialize banks for file: " << path << "\n";
            continue;
        }

        hipo::event event;
        int64_t event_index = 0;

        while (reader.next()) {
            reader.read(event);
            ++event_index;

            if (!banks.loadEvent(event)) {
                continue;
            }

            TruthInfo truth;
            if (!truth_labeler.extractTruth(banks, truth, cutflow)) {
                continue;
            }

            EventSelector selector;
            if (!selector.hasRequiredEventBanks(banks, cutflow)) {
                continue;
            }

            MatchResolver resolver;
            resolver.buildIndices(banks);

            if (!selector.hasAtLeastOneEligibleRow(banks, resolver, cutflow)) {
                continue;
            }

            FeatureBuilder feature_builder;
            RowQualityEvaluator quality_evaluator;

            for (int proj_row = 0; proj_row < banks.projectionsRows(); ++proj_row) {
                CandidateRefs refs;
                if (!selector.isRowEligible(proj_row, banks, resolver, refs, cutflow)) {
                    continue;
                }

                FeatureRow features = feature_builder.build(banks, refs);

                if (!quality_evaluator.isRowQualityAcceptable(features, refs, cutflow)) {
                    continue;
                }

                OutputRowMeta meta;
                meta.event_index = event_index;
                meta.run_number = -1;
                meta.track_id = refs.track_id;
                meta.matched_atof_hit_id = refs.matched_atof_hit_id;
                meta.cluster_id = refs.cluster_id;
                meta.truth_pid = truth.pid;
                meta.class_index = truth.class_index;
                meta.status = features.has_any_masked_feature
                                  ? static_cast<int32_t>(RowStatus::kScoredWithMaskedFeatures)
                                  : static_cast<int32_t>(RowStatus::kScoredNoMaskedFeatures);
                meta.has_any_masked_feature = features.has_any_masked_feature ? 1 : 0;

                if (!writer.writeRow(row_index, features, meta)) {
                    std::cerr << "Failed writing row " << row_index << "\n";
                    return false;
                }

                ++row_index;
            }
        }
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (!ParseArgs(argc, argv, cfg)) {
        return 1;
    }

    TruthLabeler truth_labeler;
    if (!truth_labeler.loadLabelMap(cfg.label_map_path)) {
        std::cerr << "Failed to load label map: " << cfg.label_map_path << "\n";
        return 1;
    }

    Cutflow cutflow_count_pass;
    std::size_t n_rows = CountEligibleRows(cfg, truth_labeler, cutflow_count_pass);

    std::cout << "Eligible output rows: " << n_rows << "\n";
    if (n_rows == 0) {
        std::cerr << "No eligible rows found.\n";
        return 2;
    }

    DataWriter writer;
    if (!writer.open(cfg.output_h5, n_rows)) {
        std::cerr << "Failed to open output HDF5 file: " << cfg.output_h5 << "\n";
        return 1;
    }

    Cutflow cutflow_write_pass;
    if (!WriteDataset(cfg, truth_labeler, writer, cutflow_write_pass)) {
        std::cerr << "Dataset writing failed.\n";
        writer.close();
        return 1;
    }

    if (!writer.writeMetadata(
            cfg.input_files,
            cutflow_write_pass,
            "v1",
            "postpid_feature_contract_v1")) {
        std::cerr << "Failed to write metadata.\n";
        writer.close();
        return 1;
    }

    if (!writer.close()) {
        std::cerr << "Failed to close HDF5 output cleanly.\n";
        return 1;
    }

    cutflow_write_pass.printSummary();
    return 0;
}