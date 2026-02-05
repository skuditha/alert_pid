import uproot
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
ROOT_FILE = "data/training_sample.root"
TREE_NAME = "pidTree"

PID_LIST = [2212, 45, 46, 47, 49]

FEATURES = [
    # AHDC
    "kf_x", "kf_y", "kf_z",
    "kf_px", "kf_py", "kf_pz",
    "kf_nhits", "kf_sum_adc",
    "kf_path", "kf_dedx", "kf_pdrift",
    "kf_chi2", "kf_sum_residuals",

    # ATOF
    "atof_nbar", "atof_nwedge",
    "atof_time",
    "atof_x", "atof_y", "atof_z",
    "atof_energy",
    "atof_pathlength", "atof_inpathlength",
]

# Manually specified ranges (EDIT AS NEEDED)
RANGES = {
    "kf_x": (-2.0, 2.0),
    "kf_y": (-1.5, 1.5),
    "kf_z": (-300, 300),
    "kf_px": (-2000, 2000),
    "kf_py": (-2000, 2000),
    "kf_pz": (-5000, 5000),
    "kf_nhits": (5, 15),
    "kf_sum_adc": (0, 30000),
    "kf_path": (0, 250),
    "kf_dedx": (0, 350),
    "kf_pdrift": (0, 10000),
    "kf_chi2": (0, 30),
    "kf_sum_residuals": (-15, 5),
    "atof_nbar": (0, 3),
    "atof_nwedge": (0, 4),
    "atof_time": (120, 130),
    "atof_x": (-100, 100),
    "atof_y": (-100, 100),
    "atof_z": (-320, 320),
    "atof_energy": (0, 30),
    "atof_pathlength": (0, 350),
    "atof_inpathlength": (0, 24),
}

NBINS = 100

# ==========================
# Load ROOT tree
# ==========================
with uproot.open(ROOT_FILE) as f:
    tree = f[TREE_NAME]
    data = tree.arrays(FEATURES + ["mc_pid"], library="np")

mc_pid = data["mc_pid"]

# ==========================
# Plotting loop
# ==========================
for feature in FEATURES:

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    axes = axes.flatten()

    xmin, xmax = RANGES.get(feature, (np.min(data[feature]), np.max(data[feature])))

    # ---- PID-specific plots ----
    for i, pid in enumerate(PID_LIST):
        mask = (mc_pid == pid)
        values = data[feature][mask]

        axes[i].hist(values, bins=NBINS, range=(xmin, xmax))
        axes[i].set_title(f"{feature} | PID = {pid}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Counts")

    # ---- All PIDs ----
    axes[5].hist(data[feature], bins=NBINS, range=(xmin, xmax))
    axes[5].set_title(f"{feature} | All PIDs")
    axes[5].set_xlabel(feature)
    axes[5].set_ylabel("Counts")

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"plots/{feature}.png")
    plt.close()

# ==========================
# End of script
# ==========================
