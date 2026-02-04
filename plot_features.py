import uproot
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Configuration
# ==========================
ROOT_FILE = "data/sample.root"
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
    "kf_dedx": (0, 10),
    "kf_chi2": (0, 50),
    "atof_time": (0, 50),
    "atof_energy": (0, 20),
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
    plt.show()
    plt.tight_layout()
    plt.savefig(f"plots/{feature}.png")
    plt.close()

# ==========================
# End of script
# ==========================
