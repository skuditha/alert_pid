#pragma once

#include <memory>
#include <string>

namespace hipo {
class reader;
class dictionary;
class event;
class bank;
}

namespace alert::postpid {

class AlertBanks {
public:
    AlertBanks();
    ~AlertBanks();

    bool initialize(hipo::reader& reader);
    bool loadEvent(hipo::event& event);

    int mcRows() const;
    int projectionsRows() const;
    int kftrackRows() const;
    int hitRows() const;
    int clusterRows() const;

    bool hasMC() const;
    bool hasProjections() const;
    bool hasKftrack() const;
    bool hasHits() const;
    bool hasClusters() const;

    int getMcPid(int row) const;

    int getProjectionTrackId(int row) const;
    int getProjectionMatchedAtofHitId(int row) const;

    int getKfTrackId(int row) const;
    float getKfPx(int row) const;
    float getKfPy(int row) const;
    float getKfPz(int row) const;
    float getKfX(int row) const;
    float getKfY(int row) const;
    float getKfZ(int row) const;
    float getKfNHits(int row) const;
    float getKfSumAdc(int row) const;
    float getKfPath(int row) const;
    float getKfDEdx(int row) const;
    float getKfPDrift(int row) const;
    float getKfSumResiduals(int row) const;

    int getHitId(int row) const;
    int getHitClusterId(int row) const;

    int getClusterId(int row) const;
    float getClusterTime(int row) const;
    float getClusterX(int row) const;
    float getClusterY(int row) const;
    float getClusterZ(int row) const;
    float getClusterEnergy(int row) const;
    float getClusterPathLength(int row) const;
    float getClusterNBar(int row) const;
    float getClusterNWedge(int row) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace alert::postpid