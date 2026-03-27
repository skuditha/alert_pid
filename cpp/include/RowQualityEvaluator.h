#pragma once

#include "Cutflow.h"
#include "Types.h"

namespace alert::postpid {

class RowQualityEvaluator {
public:
    struct Config {
        float max_p_mevc = 2.0e3f;
        float max_beta = 1.2f;
    };

   RowQualityEvaluator();                 // default
    explicit RowQualityEvaluator(Config cfg);  // custom

    bool isRowQualityAcceptable(
        const FeatureRow& features,
        CandidateRefs& refs,
        Cutflow& cutflow) const;

private:
    Config cfg_;

    static bool isFinite(float x);
};

}  // namespace alert::postpid