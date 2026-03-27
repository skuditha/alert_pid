#pragma once

#include "Types.h"

namespace alert::postpid {

class AlertBanks;

class FeatureBuilder {
public:
    FeatureBuilder();

    FeatureRow build(const AlertBanks& banks, const CandidateRefs& refs) const;

private:
    static void setValid(float value, float& out_value, uint8_t& out_mask);
    static void setInvalid(float& out_value, uint8_t& out_mask);

    static float safeLog(float x, bool& valid);
    static float computeMomentum(float px, float py, float pz);
    static float computePt(float px, float py);
    static float computeTheta(float p, float pz, bool& valid);
    static float computePhi(float px, float py, bool& valid);
    static float computeVr(float vx, float vy);
    static float computeV3(float vx, float vy, float vz);
    static float computeDedxRecomputed(float sum_adc, float path, bool& valid);
    static float computeResidualPerHit(float sum_residuals, float n_hits, bool& valid);
    static float computeAdcPerHit(float sum_adc, float n_hits, bool& valid);
    static float computeBeta(float pathlength, float tof_time, bool& valid);
    static float computeM2(float p, float beta, bool& valid);
};

}  // namespace alert::postpid