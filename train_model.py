#Importing all the necessary libraries
import ROOT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = ROOT.RDataFrame("pidTree","/work/clas12b/users/skuditha/ALERT/alert_pid/data/training_sample.root")
npy = df.AsNumpy()
df2 = pd.DataFrame(npy)

q_map = {
    2212: 1,
    45:   1,
    46:   1,
    47:   2,
    49:   2
}

m_map = {
    2212: 1,
    45:   2,
    46:   3,
    47:   4,
    49:   3
}

df2["Q"] = df2["mc_pid"].map(q_map)
df2["M"] = df2["mc_pid"].map(m_map)
#df2['pid'] = df2['pid'].astype(str)
df2.loc[df2["mc_pid"] == 2212, "mc_pid"] = 112212
df2.loc[df2["mc_pid"] == 45, "mc_pid"] = 120045
df2.loc[df2["mc_pid"] == 46, "mc_pid"] = 130046
df2.loc[df2["mc_pid"] == 47, "mc_pid"] = 240047
df2.loc[df2["mc_pid"] == 49, "mc_pid"] = 230049

df3 = df2[(df2['kf_chi2'] < 30)
& ((df2['kf_pz'] > -5000) & (df2['kf_pz'] < 5000))
& (df2['atof_energy'] < 30)]
filtered_df = df3.dropna()

# Find minimum group size
min_count = filtered_df['mc_pid'].value_counts().min()

# Get top min_count rows sorted by track_pred for each pid group
result_df = (
    filtered_df.groupby("mc_pid", group_keys=False)
    .apply(lambda x: x.sort_values("kf_chi2", ascending=True).head(min_count))
)

truth_cols = [
    'mc_pid','mc_px','mc_py','mc_pz','mc_vx','mc_vy','mc_vz','mc_vt','Q','M'
]
X = result_df.drop(columns=truth_cols)
y_pid = result_df['mc_pid']
y_Q = result_df['Q']
y_M = result_df['M']

X_train_pid, X_test_pid, y_train_pid, y_test_pid = train_test_split(
    X, y_pid, test_size=0.2, random_state=42, stratify=y_pid
)
X_train_Q, X_test_Q, y_train_Q, y_test_Q = train_test_split(
    X, y_Q, test_size=0.2, random_state=42, stratify=y_Q
)
X_train_M, X_test_M, y_train_M, y_test_M = train_test_split(
    X, y_M, test_size=0.2, random_state=42, stratify=y_M
)

pid_model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=200,
    max_leaf_nodes=None
)

pid_model.fit(X_train_pid, y_train_pid)

Q_model = HistGradientBoostingClassifier(max_depth=3)
Q_model.fit(X_train_Q, y_train_Q)

M_model = HistGradientBoostingClassifier(max_depth=4)
M_model.fit(X_train_M, y_train_M)

y_predict_pid = pid_model.predict(X_test_pid)
y_predict_Q = Q_model.predict(X_test_Q)
y_predict_M = M_model.predict(X_test_M)

print("PID:")
print(classification_report(y_test_pid, y_predict_pid))

print("Q:")
print(classification_report(y_test_Q, y_predict_Q))

print("M:")
print(classification_report(y_test_M, y_predict_M))

cm_pid = confusion_matrix(y_test_pid, y_predict_pid,normalize="true")
sns.heatmap(cm_pid, annot=True, fmt='.1%',xticklabels = ['proton','deuteron','tritium','3He','4He'],yticklabels = ['proton','deuteron','tritium','3He','4He'])
#plt.show()
plt.savefig(f"plots/pid.png")
plt.close()

cm_Q = confusion_matrix(y_test_Q, y_predict_Q,normalize="true")
sns.heatmap(cm_Q, annot=True, fmt='.1%',xticklabels = ['+1q','+2q'],yticklabels = ['+1q','+2q'])
#plt.show()
plt.savefig(f"plots/Q.png")
plt.close()

cm_M = confusion_matrix(y_test_M, y_predict_M,normalize="true")
sns.heatmap(cm_M, annot=True, fmt='.1%',xticklabels = ['1','2','3','4'],yticklabels = ['1','2','3','4'])
#plt.show()
plt.savefig(f"plots/M.png")
plt.close()
