#include "AlertBanks.h"

#include <memory>
#include <stdexcept>
#include <string>

// Match these includes to the working sample .cc files on your system.
#include "bank.h"
#include "dictionary.h"
#include "event.h"
#include "reader.h"
//#include "schema.h"

namespace alert::postpid {

class AlertBanks::Impl {
public:
    std::unique_ptr<hipo::dictionary> dict;

    std::unique_ptr<hipo::bank> mc;
    std::unique_ptr<hipo::bank> projections;
    std::unique_ptr<hipo::bank> kftrack;
    std::unique_ptr<hipo::bank> hits;
    std::unique_ptr<hipo::bank> clusters;
};

AlertBanks::AlertBanks() : impl_(std::make_unique<Impl>()) {}
AlertBanks::~AlertBanks() = default;

bool AlertBanks::initialize(hipo::reader& reader) {
    impl_->dict = std::make_unique<hipo::dictionary>();
    reader.readDictionary(*impl_->dict);

    try {
        impl_->mc = std::make_unique<hipo::bank>(impl_->dict->getSchema("MC::Particle"));
        impl_->projections = std::make_unique<hipo::bank>(impl_->dict->getSchema("ALERT::ai:projections"));
        impl_->kftrack = std::make_unique<hipo::bank>(impl_->dict->getSchema("AHDC::kftrack"));
        impl_->hits = std::make_unique<hipo::bank>(impl_->dict->getSchema("ATOF::hits"));
        impl_->clusters = std::make_unique<hipo::bank>(impl_->dict->getSchema("ATOF::clusters"));
    } catch (...) {
        return false;
    }

    return true;
}

bool AlertBanks::loadEvent(hipo::event& event) {
    try {
        event.read(*impl_->mc);
        event.read(*impl_->projections);
        event.read(*impl_->kftrack);
        event.read(*impl_->hits);
        event.read(*impl_->clusters);
    } catch (...) {
        return false;
    }
    return true;
}

int AlertBanks::mcRows() const { return impl_->mc ? impl_->mc->getRows() : 0; }
int AlertBanks::projectionsRows() const { return impl_->projections ? impl_->projections->getRows() : 0; }
int AlertBanks::kftrackRows() const { return impl_->kftrack ? impl_->kftrack->getRows() : 0; }
int AlertBanks::hitRows() const { return impl_->hits ? impl_->hits->getRows() : 0; }
int AlertBanks::clusterRows() const { return impl_->clusters ? impl_->clusters->getRows() : 0; }

bool AlertBanks::hasMC() const { return mcRows() > 0; }
bool AlertBanks::hasProjections() const { return projectionsRows() > 0; }
bool AlertBanks::hasKftrack() const { return kftrackRows() > 0; }
bool AlertBanks::hasHits() const { return hitRows() > 0; }
bool AlertBanks::hasClusters() const { return clusterRows() > 0; }

int AlertBanks::getMcPid(int row) const {
    return impl_->mc->getInt("pid", row);
}

int AlertBanks::getProjectionTrackId(int row) const {
    return impl_->projections->getInt("trackID", row);
}

int AlertBanks::getProjectionMatchedAtofHitId(int row) const {
    return impl_->projections->getInt("matched_atof_hit_id", row);
}

int AlertBanks::getKfTrackId(int row) const {
    return impl_->kftrack->getInt("trackid", row);
}

float AlertBanks::getKfPx(int row) const { return impl_->kftrack->getFloat("px", row); }
float AlertBanks::getKfPy(int row) const { return impl_->kftrack->getFloat("py", row); }
float AlertBanks::getKfPz(int row) const { return impl_->kftrack->getFloat("pz", row); }

float AlertBanks::getKfX(int row) const { return impl_->kftrack->getFloat("x", row); }
float AlertBanks::getKfY(int row) const { return impl_->kftrack->getFloat("y", row); }
float AlertBanks::getKfZ(int row) const { return impl_->kftrack->getFloat("z", row); }

float AlertBanks::getKfNHits(int row) const { return impl_->kftrack->getFloat("n_hits", row); }
float AlertBanks::getKfSumAdc(int row) const { return impl_->kftrack->getFloat("sum_adc", row); }
float AlertBanks::getKfPath(int row) const { return impl_->kftrack->getFloat("path", row); }
float AlertBanks::getKfDEdx(int row) const { return impl_->kftrack->getFloat("dEdx", row); }
float AlertBanks::getKfPDrift(int row) const { return impl_->kftrack->getFloat("p_drift", row); }
float AlertBanks::getKfSumResiduals(int row) const { return impl_->kftrack->getFloat("sum_residuals", row); }

int AlertBanks::getHitId(int row) const {
    return impl_->hits->getInt("id", row);
}

int AlertBanks::getHitClusterId(int row) const {
    return impl_->hits->getInt("clusterid", row);
}

int AlertBanks::getClusterId(int row) const {
    return impl_->clusters->getInt("id", row);
}

float AlertBanks::getClusterTime(int row) const {
    return impl_->clusters->getFloat("time", row);
}

float AlertBanks::getClusterX(int row) const {
    return impl_->clusters->getFloat("x", row);
}

float AlertBanks::getClusterY(int row) const {
    return impl_->clusters->getFloat("y", row);
}

float AlertBanks::getClusterZ(int row) const {
    return impl_->clusters->getFloat("z", row);
}

float AlertBanks::getClusterEnergy(int row) const {
    return impl_->clusters->getFloat("energy", row);
}

float AlertBanks::getClusterPathLength(int row) const {
    return impl_->clusters->getFloat("pathlength", row);
}

float AlertBanks::getClusterNBar(int row) const {
    return impl_->clusters->getFloat("n_bar", row);
}

float AlertBanks::getClusterNWedge(int row) const {
    return impl_->clusters->getFloat("n_wedge", row);
}

}  // namespace alert::postpid