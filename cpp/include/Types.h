#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace alert::postpid {

constexpr int kNumFeatures = 38;

enum class RowStatus : int32_t {
    kScoredNoMaskedFeatures = 1,
    kScoredWithMaskedFeatures = 2,

    // Structural eligibility failures
    kMissingProjections = -1,
    kInvalidTrackId = -2,
    kInvalidMatchedAtofHitId = -3,
    kMissingKftrackMatch = -4,
    kMissingAtofHitMatch = -5,
    kInvalidClusterId = -6,
    kMissingAtofClusterMatch = -7,

    // Row quality failures
    kRowQualityNonPositiveP = -20,
    kRowQualityPTooLarge = -21,
    kRowQualityNonPositiveTofTime = -22,
    kRowQualityNonPositivePathlength = -23,
    kRowQualityNonPositiveBeta = -24,
    kRowQualityBetaTooLarge = -25,
    kRowQualityNonFiniteCritical = -26
};

struct TruthInfo {
    bool valid = false;
    int pid = 0;
    int class_index = -1;
    std::string class_name;
};

struct CandidateRefs {
    int proj_row = -1;
    int kftrack_row = -1;
    int hit_row = -1;
    int cluster_row = -1;

    int track_id = -1;
    int matched_atof_hit_id = -1;
    int cluster_id = -1;

    RowStatus status = RowStatus::kScoredNoMaskedFeatures;
};

struct FeatureRow {
    std::array<float, kNumFeatures> values{};
    std::array<uint8_t, kNumFeatures> masks{};
    bool has_any_masked_feature = false;
};

struct OutputRowMeta {
    int64_t event_index = -1;
    int32_t run_number = -1;
    int32_t track_id = -1;
    int32_t matched_atof_hit_id = -1;
    int32_t cluster_id = -1;
    int32_t truth_pid = 0;
    int32_t class_index = -1;
    int32_t status = 0;
    uint8_t has_any_masked_feature = 0;
};

inline const std::array<const char*, kNumFeatures> kFeatureNames = {{
    "px",
    "py",
    "pz",
    "p",
    "pt",
    "theta",
    "phi",
    "vx",
    "vy",
    "vz",
    "vr",
    "v3",
    "n_hits",
    "sum_adc",
    "path",
    "dEdx",
    "dedx_recomputed",
    "p_drift",
    "sum_residuals",
    "residual_per_hit",
    "adc_per_hit",
    "tof_time",
    "pathlength",
    "cluster_x",
    "cluster_y",
    "cluster_z",
    "cluster_energy",
    "n_bar",
    "n_wedge",
    "beta",
    "m2",
    "log_p",
    "log_pt",
    "log_sum_adc",
    "log_path",
    "log_dEdx",
    "log_dedx_recomputed",
    "log_cluster_energy"
}};

}  // namespace alert::postpid