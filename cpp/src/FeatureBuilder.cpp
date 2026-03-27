#include "FeatureBuilder.h"

#include <cmath>

#include "AlertBanks.h"

namespace alert::postpid {
namespace {
constexpr float kC_cm_per_ns = 29.9792458f;
constexpr float kZeroTol = 1e-12f;
}  // namespace

FeatureBuilder::FeatureBuilder() = default;

void FeatureBuilder::setValid(float value, float& out_value, uint8_t& out_mask) {
    out_value = value;
    out_mask = 1;
}

void FeatureBuilder::setInvalid(float& out_value, uint8_t& out_mask) {
    out_value = 0.0f;
    out_mask = 0;
}

float FeatureBuilder::safeLog(float x, bool& valid) {
    if (x > 0.0f) {
        valid = true;
        return std::log(x);
    }
    valid = false;
    return 0.0f;
}

float FeatureBuilder::computeMomentum(float px, float py, float pz) {
    return std::sqrt(px * px + py * py + pz * pz);
}

float FeatureBuilder::computePt(float px, float py) {
    return std::sqrt(px * px + py * py);
}

float FeatureBuilder::computeTheta(float p, float pz, bool& valid) {
    if (p > kZeroTol) {
        valid = true;
        float c = pz / p;
        if (c > 1.0f) c = 1.0f;
        if (c < -1.0f) c = -1.0f;
        return std::acos(c);
    }
    valid = false;
    return 0.0f;
}

float FeatureBuilder::computePhi(float px, float py, bool& valid) {
    if (std::abs(px) > kZeroTol || std::abs(py) > kZeroTol) {
        valid = true;
        return std::atan2(py, px);
    }
    valid = false;
    return 0.0f;
}

float FeatureBuilder::computeVr(float vx, float vy) {
    return std::sqrt(vx * vx + vy * vy);
}

float FeatureBuilder::computeV3(float vx, float vy, float vz) {
    return std::sqrt(vx * vx + vy * vy + vz * vz);
}

float FeatureBuilder::computeDedxRecomputed(float sum_adc, float path, bool& valid) {
    if (path > kZeroTol) {
        valid = true;
        return sum_adc / path;
    }
    valid = false;
    return 0.0f;
}

float FeatureBuilder::computeResidualPerHit(float sum_residuals, float n_hits, bool& valid) {
    if (n_hits > kZeroTol) {
        valid = true;
        return sum_residuals / n_hits;
    }
    valid = false;
    return 0.0f;
}

float FeatureBuilder::computeAdcPerHit(float sum_adc, float n_hits, bool& valid) {
    if (n_hits > kZeroTol) {
        valid = true;
        return sum_adc / n_hits;
    }
    valid = false;
    return 0.0f;
}

float FeatureBuilder::computeBeta(float pathlength, float tof_time, bool& valid) {
    if (pathlength > kZeroTol && tof_time > kZeroTol) {
        valid = true;
        return pathlength / (kC_cm_per_ns * tof_time);
    }
    valid = false;
    return 0.0f;
}

float FeatureBuilder::computeM2(float p, float beta, bool& valid) {
    if (p > kZeroTol && beta > kZeroTol && beta < 1.0f) {
        valid = true;
        const float invb2 = 1.0f / (beta * beta);
        return p * p * (invb2 - 1.0f);
    }
    valid = false;
    return 0.0f;
}

FeatureRow FeatureBuilder::build(const AlertBanks& banks, const CandidateRefs& refs) const {
    FeatureRow row{};

    const int kf = refs.kftrack_row;
    const int cl = refs.cluster_row;

    const float px = banks.getKfPx(kf);
    const float py = banks.getKfPy(kf);
    const float pz = banks.getKfPz(kf);
    const float vx = banks.getKfX(kf);
    const float vy = banks.getKfY(kf);
    const float vz = banks.getKfZ(kf);
    const float n_hits = banks.getKfNHits(kf);
    const float sum_adc = banks.getKfSumAdc(kf);
    const float path = banks.getKfPath(kf);
    const float dedx = banks.getKfDEdx(kf);
    const float p_drift = banks.getKfPDrift(kf);
    const float sum_residuals = banks.getKfSumResiduals(kf);

    const float tof_time = banks.getClusterTime(cl);
    const float pathlength = banks.getClusterPathLength(cl);
    const float cluster_x = banks.getClusterX(cl);
    const float cluster_y = banks.getClusterY(cl);
    const float cluster_z = banks.getClusterZ(cl);
    const float cluster_energy = banks.getClusterEnergy(cl);
    const float n_bar = banks.getClusterNBar(cl);
    const float n_wedge = banks.getClusterNWedge(cl);

    const float p = computeMomentum(px, py, pz);
    const float pt = computePt(px, py);

    bool valid = false;
    const float theta = computeTheta(p, pz, valid);
    const uint8_t theta_mask = valid ? 1 : 0;

    const float phi = computePhi(px, py, valid);
    const uint8_t phi_mask = valid ? 1 : 0;

    const float vr = computeVr(vx, vy);
    const float v3 = computeV3(vx, vy, vz);

    const float dedx_recomputed = computeDedxRecomputed(sum_adc, path, valid);
    const uint8_t dedx_recomputed_mask = valid ? 1 : 0;

    const float residual_per_hit = computeResidualPerHit(sum_residuals, n_hits, valid);
    const uint8_t residual_per_hit_mask = valid ? 1 : 0;

    const float adc_per_hit = computeAdcPerHit(sum_adc, n_hits, valid);
    const uint8_t adc_per_hit_mask = valid ? 1 : 0;

    const float beta = computeBeta(pathlength, tof_time, valid);
    const uint8_t beta_mask = valid ? 1 : 0;

    const float m2 = computeM2(p, beta, valid);
    const uint8_t m2_mask = valid ? 1 : 0;

    bool log_valid = false;
    const float log_p = safeLog(p, log_valid);
    const uint8_t log_p_mask = log_valid ? 1 : 0;

    const float log_pt = safeLog(pt, log_valid);
    const uint8_t log_pt_mask = log_valid ? 1 : 0;

    const float log_sum_adc = safeLog(sum_adc, log_valid);
    const uint8_t log_sum_adc_mask = log_valid ? 1 : 0;

    const float log_path = safeLog(path, log_valid);
    const uint8_t log_path_mask = log_valid ? 1 : 0;

    const float log_dedx = safeLog(dedx, log_valid);
    const uint8_t log_dedx_mask = log_valid ? 1 : 0;

    const float log_dedx_recomputed = safeLog(dedx_recomputed, log_valid);
    const uint8_t log_dedx_recomputed_mask = log_valid ? 1 : 0;

    const float log_cluster_energy = safeLog(cluster_energy, log_valid);
    const uint8_t log_cluster_energy_mask = log_valid ? 1 : 0;

    auto set = [&](int idx, float value, uint8_t mask = 1) {
        row.values[idx] = (mask ? value : 0.0f);
        row.masks[idx] = mask;
        if (!mask) {
            row.has_any_masked_feature = true;
        }
    };

    set(0, px);
    set(1, py);
    set(2, pz);
    set(3, p);
    set(4, pt);
    set(5, theta, theta_mask);
    set(6, phi, phi_mask);
    set(7, vx);
    set(8, vy);
    set(9, vz);
    set(10, vr);
    set(11, v3);
    set(12, n_hits);
    set(13, sum_adc);
    set(14, path);
    set(15, dedx);
    set(16, dedx_recomputed, dedx_recomputed_mask);
    set(17, p_drift);
    set(18, sum_residuals);
    set(19, residual_per_hit, residual_per_hit_mask);
    set(20, adc_per_hit, adc_per_hit_mask);
    set(21, tof_time);
    set(22, pathlength);
    set(23, cluster_x);
    set(24, cluster_y);
    set(25, cluster_z);
    set(26, cluster_energy);
    set(27, n_bar);
    set(28, n_wedge);
    set(29, beta, beta_mask);
    set(30, m2, m2_mask);
    set(31, log_p, log_p_mask);
    set(32, log_pt, log_pt_mask);
    set(33, log_sum_adc, log_sum_adc_mask);
    set(34, log_path, log_path_mask);
    set(35, log_dedx, log_dedx_mask);
    set(36, log_dedx_recomputed, log_dedx_recomputed_mask);
    set(37, log_cluster_energy, log_cluster_energy_mask);

    return row;
}

}  // namespace alert::postpid