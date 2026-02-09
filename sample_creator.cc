#include <iostream>
#include <vector>

// HIPO
#include "reader.h"
#include "event.h"
#include "bank.h"

// ROOT
#include "TFile.h"
#include "TTree.h"

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " input.hipo output.root" << std::endl;
        return 1;
    }

    const char* inputFile  = argv[1];
    const char* outputFile = argv[2];

    // ==========================
    // Open HIPO reader
    // ==========================
    hipo::reader reader;
    reader.open(inputFile);

    hipo::dictionary factory;
    reader.readDictionary(factory);

    hipo::event event;

    // ==========================
    // Define banks
    // ==========================
    hipo::bank b_kftrack;
    hipo::bank b_atof;
    hipo::bank b_mc;

    b_kftrack = factory.getSchema("AHDC::track");
    b_atof    = factory.getSchema("ATOF::clusters");
    b_mc      = factory.getSchema("MC::Particle");

    // ==========================
    // Create ROOT output
    // ==========================
    TFile* fout = new TFile(outputFile, "RECREATE");
    TTree* tree = new TTree("pidTree", "ALERT AI PID training tree");

    // ==========================
    // AHDC::kftrack variables
    // ==========================
    float kf_x, kf_y, kf_z;
    float kf_px, kf_py, kf_pz;
    int   kf_sum_adc, kf_nhits;
    float kf_path, kf_dedx, kf_pdrift;
    float kf_chi2, kf_sum_residuals;

    // ==========================
    // ATOF::clusters variables
    // ==========================
    int   atof_nbar, atof_nwedge;
    float atof_time;
    float atof_x, atof_y, atof_z;
    float atof_energy;
    float atof_pathlength, atof_inpathlength;

    // ==========================
    // MC::Particle variables
    // ==========================
    int   mc_pid;
    float mc_px, mc_py, mc_pz;
    float mc_vx, mc_vy, mc_vz, mc_vt;

    // ==========================
    // Tree branches
    // ==========================
    tree->Branch("kf_x", &kf_x);
    tree->Branch("kf_y", &kf_y);
    tree->Branch("kf_z", &kf_z);
    tree->Branch("kf_px", &kf_px);
    tree->Branch("kf_py", &kf_py);
    tree->Branch("kf_pz", &kf_pz);
    tree->Branch("kf_nhits", &kf_nhits);
    tree->Branch("kf_sum_adc", &kf_sum_adc);
    tree->Branch("kf_path", &kf_path);
    tree->Branch("kf_dedx", &kf_dedx);
    tree->Branch("kf_pdrift", &kf_pdrift);
    tree->Branch("kf_chi2", &kf_chi2);
    tree->Branch("kf_sum_residuals", &kf_sum_residuals);

    tree->Branch("atof_nbar", &atof_nbar);
    tree->Branch("atof_nwedge", &atof_nwedge);
    tree->Branch("atof_time", &atof_time);
    tree->Branch("atof_x", &atof_x);
    tree->Branch("atof_y", &atof_y);
    tree->Branch("atof_z", &atof_z);
    tree->Branch("atof_energy", &atof_energy);
    tree->Branch("atof_pathlength", &atof_pathlength);
    tree->Branch("atof_inpathlength", &atof_inpathlength);

    tree->Branch("mc_pid", &mc_pid);
    tree->Branch("mc_px", &mc_px);
    tree->Branch("mc_py", &mc_py);
    tree->Branch("mc_pz", &mc_pz);
    tree->Branch("mc_vx", &mc_vx);
    tree->Branch("mc_vy", &mc_vy);
    tree->Branch("mc_vz", &mc_vz);
    tree->Branch("mc_vt", &mc_vt);

    // ==========================
    // Event loop
    // ==========================
    while (reader.next(event)) {

        event.getStructure(b_kftrack);
        event.getStructure(b_atof);
        event.getStructure(b_mc);

        // Require at least one entry
        if (b_kftrack.getRows() < 1 || b_atof.getRows() < 1 || b_mc.getRows() < 1)
            continue;

        // ---- Take first KF track ----
        kf_x   = b_kftrack.getFloat("x", 0);
        kf_y   = b_kftrack.getFloat("y", 0);
        kf_z   = b_kftrack.getFloat("z", 0);
        kf_px  = b_kftrack.getFloat("px", 0);
        kf_py  = b_kftrack.getFloat("py", 0);
        kf_pz  = b_kftrack.getFloat("pz", 0);

        kf_nhits          = b_kftrack.getInt("n_hits", 0);
        kf_sum_adc        = b_kftrack.getInt("sum_adc", 0);
        kf_path           = b_kftrack.getFloat("path", 0);
        kf_dedx           = b_kftrack.getFloat("dEdx", 0);
        kf_pdrift         = b_kftrack.getFloat("p_drift", 0);
        kf_chi2           = b_kftrack.getFloat("chi2", 0);
        kf_sum_residuals  = b_kftrack.getFloat("sum_residuals", 0);

        // ---- First ATOF cluster ----
        atof_nbar          = b_atof.getInt("n_bar", 0);
        atof_nwedge        = b_atof.getInt("n_wedge", 0);
        atof_time          = b_atof.getFloat("time", 0);
        atof_x             = b_atof.getFloat("x", 0);
        atof_y             = b_atof.getFloat("y", 0);
        atof_z             = b_atof.getFloat("z", 0);
        atof_energy        = b_atof.getFloat("energy", 0);
        atof_pathlength    = b_atof.getFloat("pathlength", 0);
        atof_inpathlength  = b_atof.getFloat("inpathlength", 0);

        // ---- MC truth (primary particle) ----
        mc_pid = b_mc.getInt("pid", 0);
        mc_px  = b_mc.getFloat("px", 0);
        mc_py  = b_mc.getFloat("py", 0);
        mc_pz  = b_mc.getFloat("pz", 0);
        mc_vx  = b_mc.getFloat("vx", 0);
        mc_vy  = b_mc.getFloat("vy", 0);
        mc_vz  = b_mc.getFloat("vz", 0);
        mc_vt  = b_mc.getFloat("vt", 0);

        tree->Fill();
    }

    fout->Write();
    fout->Close();

    std::cout << "Wrote ROOT file: " << outputFile << std::endl;

    return 0;
}
