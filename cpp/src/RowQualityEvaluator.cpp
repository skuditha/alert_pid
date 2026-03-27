#include "RowQualityEvaluator.h"

#include <cmath>

namespace alert::postpid {
namespace {

// Canonical feature indices from Types.h / feature contract.
constexpr int kIdxP = 3;
constexpr int kIdxTofTime = 21;
constexpr int kIdxPathlength = 22;
constexpr int kIdxBeta = 29;

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

    return true;
}

}  // namespace alert::postpid