#include "RowQualityEvaluator.h"

#include <cmath>

namespace alert::postpid {
namespace {

// Canonical feature indices from Types.h / feature contract.
constexpr int kIdxP = 3;
constexpr int kIdxTofTime = 21;
constexpr int kIdxPathlength = 22;
constexpr int kIdxBeta = 29;
constexpr int kIdxSumADC = 13;
constexpr int kIdxPath = 14;
constexpr int kIdxVr = 10;
constexpr int kIdxV3 = 11;

}  // namespace

RowQualityEvaluator::RowQualityEvaluator()
    : cfg_{}  // uses default member initializers
{}

RowQualityEvaluator::RowQualityEvaluator(Config cfg)
    : cfg_(cfg)
{}

bool RowQualityEvaluator::isFinite(float x) {
    return std::isfinite(x);
}

bool RowQualityEvaluator::isRowQualityAcceptable(
    const FeatureRow& features,
    CandidateRefs& refs,
    Cutflow& cutflow) const
{
    const float p = features.values[kIdxP];
    const float tof_time = features.values[kIdxTofTime];
    const float pathlength = features.values[kIdxPathlength];
    const float beta = features.values[kIdxBeta];
    const int sum_adc = features.values[kIdxSumADC];
    const float path = features.values[kIdxPath];
    const float vr = features.values[kIdxVr];
    const float v3 = features.values[kIdxV3];

    // A. Any nonfinite in critical quantities
    if (!isFinite(p) || !isFinite(tof_time) || !isFinite(pathlength) || !isFinite(beta)) {
        refs.status = RowStatus::kRowQualityNonFiniteCritical;
        cutflow.increment("row_quality_nonfinite_critical");
        return false;
    }

    // B. Physics sanity / catastrophic outliers
    if (p <= 0.0f) {
        refs.status = RowStatus::kRowQualityNonPositiveP;
        cutflow.increment("row_quality_nonpositive_p");
        return false;
    }

    if (p > cfg_.max_p_mevc) {
        refs.status = RowStatus::kRowQualityPTooLarge;
        cutflow.increment("row_quality_p_too_large");
        return false;
    }

    if (tof_time <= 0.0f) {
        refs.status = RowStatus::kRowQualityNonPositiveTofTime;
        cutflow.increment("row_quality_nonpositive_tof_time");
        return false;
    }

    if (pathlength <= 0.0f) {
        refs.status = RowStatus::kRowQualityNonPositivePathlength;
        cutflow.increment("row_quality_nonpositive_pathlength");
        return false;
    }

    if (beta <= 0.0f) {
        refs.status = RowStatus::kRowQualityNonPositiveBeta;
        cutflow.increment("row_quality_nonpositive_beta");
        return false;
    }

    if (beta > cfg_.max_beta) {
        refs.status = RowStatus::kRowQualityBetaTooLarge;
        cutflow.increment("row_quality_beta_too_large");
        return false;
    }

    if (sum_adc > cfg_.max_sum_adc) {
        refs.status = RowStatus::kRowQualitySumADCTooLarge;
        cutflow.increment("row_quality_sum_adc_too_large");
        return false;
    }

    if (path > cfg_.max_path) {
        refs.status = RowStatus::kRowQualityPathTooLarge;
        cutflow.increment("row_quality_path_too_large");
        return false;
    }

    if (vr > cfg_.max_vr) {
        refs.status = RowStatus::kRowQualityVrTooLarge;
        cutflow.increment("row_quality_vr_too_large");
        return false;
    }

    if (v3 > cfg_.max_v3) {
        refs.status = RowStatus::kRowQualityV3TooLarge;
        cutflow.increment("row_quality_v3_too_large");
        return false;
    }

    if (pathlength > cfg_.max_pathlength) {
        refs.status = RowStatus::kRowQualityPathLengthTooLarge;
        cutflow.increment("row_quality_pathlength_too_large");
        return false;
    }

    if (tof_time > cfg_.max_tof_time) {
        refs.status = RowStatus::kRowQualityTOFTimeTooLarge;
        cutflow.increment("row_quality_tof_time_too_large");
        return false;
    }

    return true;
}

}  // namespace alert::postpid