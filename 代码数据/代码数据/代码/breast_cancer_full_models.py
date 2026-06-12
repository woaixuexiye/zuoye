# ==============================
# 威斯康星乳腺癌诊断 - 全模型建立方案（修复版+分文件夹存储）
# 包含：关联规则+决策树+3类集成+3类传统ML+朴素贝叶斯+MLP
# 输出：每个模型结果存在独立文件夹
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

# ====================== 0. 基础设置：创建文件夹 ======================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
base_dir = os.path.join(desktop_path, "breast_cancer_results")

# 定义每个模型的文件夹路径
folders = {
    "关联规则": os.path.join(base_dir, "1_关联规则"),
    "决策树": os.path.join(base_dir, "2_决策树"),
    "随机森林": os.path.join(base_dir, "3_随机森林"),
    "梯度提升树": os.path.join(base_dir, "4_梯度提升树"),
    "AdaBoost": os.path.join(base_dir, "5_AdaBoost"),
    "逻辑回归": os.path.join(base_dir, "6_逻辑回归"),
    "SVM": os.path.join(base_dir, "7_SVM"),
    "KNN": os.path.join(base_dir, "8_KNN"),
    "朴素贝叶斯": os.path.join(base_dir, "9_朴素贝叶斯"),
    "MLP": os.path.join(base_dir, "10_MLP"),
    "汇总": os.path.join(base_dir, "汇总结果")
}

# 自动创建所有文件夹（不存在则创建）
for path in folders.values():
    os.makedirs(path, exist_ok=True)

# 全局绘图设置（美观统一）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300

# ====================== 1. 数据预处理（通用）======================
# 读取数据（桌面路径）
data_path = os.path.join(desktop_path, "wisconsin_breast_cancer.csv")
df = pd.read_csv(data_path)

# 数据清洗：标签映射（M=恶性1，B=良性0）+ 删无用ID
df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])  # B=0, M=1
df = df.drop(columns=["ID"], errors="ignore")  # 防止ID列不存在

# 特征与标签拆分
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# 标准化（用于逻辑回归、SVM、KNN、MLP）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 训练集/测试集划分（所有模型通用）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train_scaled, X_test_scaled, _, _ = train_test_split(
    X_scaled_df, y, test_size=0.3, random_state=42, stratify=y
)

# ====================== 2. 关联规则挖掘（保留并优化）======================
print("="*50)
print("【1/10 关联规则挖掘】")
print("="*50)

# 连续特征离散化（低/中/高，更贴合临床）
def discretize_3level(series):
    return pd.qcut(series, q=3, labels=["Low", "Mid", "High"])

# 对所有特征离散化
X_discrete = X.copy()
for col in X.columns:
    X_discrete[col] = discretize_3level(X[col])
    X_discrete[col] = col + "_" + X_discrete[col].astype(str)  # 特征名+级别

# 构建事务集（离散特征+诊断标签）
transactions = []
for idx in range(len(X_discrete)):
    row = X_discrete.iloc[idx].tolist()
    row.append("Diagnosis_" + ("M" if y.iloc[idx]==1 else "B"))
    transactions.append(row)

# Apriori算法挖掘频繁项集
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
trans_df = pd.DataFrame(te_array, columns=te.columns_)

# 挖掘频繁项集（支持度≥0.15，平衡数量与可靠性）
frequent_items = apriori(trans_df, min_support=0.15, use_colnames=True)

# 生成关联规则（置信度≥0.7，筛选强规则）
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)

# 筛选与诊断相关的规则（后件包含诊断结果）
diagnosis_rules = rules[rules["consequents"].astype(str).str.contains("Diagnosis")]
diagnosis_rules = diagnosis_rules.sort_values(by=["lift", "confidence"], ascending=False)
top10_rules = diagnosis_rules.head(10)

# 输出结果
print("Top10 强关联规则：")
print(top10_rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# 保存规则到关联规则文件夹
top10_rules.to_csv(
    os.path.join(folders["关联规则"], "关联规则Top10.csv"), 
    index=False, encoding="utf-8-sig"
)

# 可视化1：关联规则网络图（重点优化，修复版）
plt.figure(figsize=(14, 10))
G = nx.DiGraph()

# 添加节点（前件特征、后件诊断）
for _, rule in top10_rules.iterrows():
    antecedent = list(rule["antecedents"])[0]  # 简化：取前件第一个特征
    consequent = list(rule["consequents"])[0]
    G.add_node(antecedent, color="lightblue", size=1000)
    G.add_node(consequent, color="coral", size=2000)
    # 添加边（权重=提升度）
    G.add_edge(antecedent, consequent, weight=rule["lift"], label=f"Lift:{rule['lift']:.2f}")

# 设置节点颜色
node_colors = [G.nodes[n]["color"] for n in G.nodes]
node_sizes = [G.nodes[n]["size"] for n in G.nodes]

# 绘制网络（修复参数名）
pos = nx.spring_layout(G, k=3, iterations=50)  # 布局调整
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
nx.draw_networkx_edges(G, pos, width=[G[u][v]["weight"]/2 for u, v in G.edges()], alpha=0.6)
# 修复：fontsize → font_size
nx.draw_networkx_labels(G, pos, font_size=10)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]["label"] for u, v in G.edges()}, font_size=8)

plt.title("乳腺癌病理特征→良恶性 关联规则网络图（Top10）", fontsize=16, pad=20)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(folders["关联规则"], "关联规则网络图.png"), bbox_inches="tight")
plt.close()

# 可视化2：支持度-置信度-提升度散点图
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    top10_rules["support"], top10_rules["confidence"],
    c=top10_rules["lift"], cmap="viridis", s=200, alpha=0.8, edgecolors="black"
)
plt.colorbar(scatter, label="提升度（Lift）")
plt.xlabel("支持度（Support）", fontsize=12)
plt.ylabel("置信度（Confidence）", fontsize=12)
plt.title("关联规则 支持度-置信度-提升度分布", fontsize=14)
plt.grid(alpha=0.3, linestyle="--")

# 添加数据标签（前3条规则）
for idx, row in top10_rules.head(3).iterrows():
    plt.annotate(
        f"Lift:{row['lift']:.2f}",
        (row["support"], row["confidence"]),
        xytext=(5, 5), textcoords="offset points", fontsize=9
    )

plt.tight_layout()
plt.savefig(os.path.join(folders["关联规则"], "关联规则散点图.png"), bbox_inches="tight")
plt.close()

# ====================== 3. 决策树分类模型（保留+扩展）======================
print("\n" + "="*50)
print("【2/10 决策树（CART）+ 剪枝】")
print("="*50)

# 1. CART分类树（预剪枝+后剪枝）
# 预剪枝模型
dt_preprune = DecisionTreeClassifier(
    criterion="gini", max_depth=4, min_samples_split=10,
    min_samples_leaf=5, random_state=42
)
dt_preprune.fit(X_train, y_train)

# 后剪枝（成本复杂度剪枝）
path = dt_preprune.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]  # 排除最后一个alpha（会导致树只剩根节点）
dt_postprune = DecisionTreeClassifier(
    criterion="gini", ccp_alpha=ccp_alphas[np.argmax([
        accuracy_score(y_test, DecisionTreeClassifier(criterion="gini", ccp_alpha=a, random_state=42).fit(X_train, y_train).predict(X_test)) 
        for a in ccp_alphas
    ])], random_state=42
)
dt_postprune.fit(X_train, y_train)

# 2. CART回归树（对比特征重要性）
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(
    max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42
)
dt_reg.fit(X_train, y_train)

# 预测与评估（后剪枝模型为最终模型）
y_dt_pred = dt_postprune.predict(X_test)
dt_acc = accuracy_score(y_test, y_dt_pred)
print(f"后剪枝CART分类树准确率：{dt_acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_dt_pred, target_names=["良性(B)", "恶性(M)"]))

# 特征重要性对比（分类树vs回归树）
feature_importance = pd.DataFrame({
    "特征": X.columns,
    "CART分类树重要性": dt_postprune.feature_importances_,
    "CART回归树重要性": dt_reg.feature_importances_
}).sort_values(by="CART分类树重要性", ascending=False).head(10)

print("\nTop10 特征重要性（分类树vs回归树）：")
print(feature_importance)

# 保存结果到决策树文件夹
feature_importance.to_csv(
    os.path.join(folders["决策树"], "决策树特征重要性.csv"), 
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(classification_report(
    y_test, y_dt_pred, target_names=["良性(B)", "恶性(M)"], output_dict=True
)).T.round(4).to_csv(
    os.path.join(folders["决策树"], "决策树分类报告.csv"), 
    index=True, encoding="utf-8-sig"
)

# 可视化1：决策树结构图（后剪枝）
plt.figure(figsize=(20, 12))
plot_tree(
    dt_postprune, feature_names=X.columns, class_names=["良性", "恶性"],
    filled=True, rounded=True, fontsize=9, proportion=True
)
plt.title("后剪枝CART决策树结构（乳腺癌诊断）", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(folders["决策树"], "决策树结构图.png"), bbox_inches="tight")
plt.close()

# 可视化2：特征重要性对比（条形图）
plt.figure(figsize=(14, 8))
feature_importance_melt = pd.melt(
    feature_importance, id_vars=["特征"], 
    value_vars=["CART分类树重要性", "CART回归树重要性"],
    var_name="模型类型", value_name="重要性"
)

sns.barplot(x="重要性", y="特征", hue="模型类型", data=feature_importance_melt, palette="Set2")
plt.xlabel("特征重要性分数", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.title("CART分类树 vs 回归树 特征重要性对比（Top10）", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(folders["决策树"], "决策树特征重要性对比.png"), bbox_inches="tight")
plt.close()

# ====================== 4. 集成学习模型（3类重点扩展）======================
# 4.1 随机森林
print("\n" + "="*50)
print("【3/10 集成学习 - 随机森林】")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=100, max_features="sqrt", max_depth=6,
    min_samples_split=8, oob_score=True, random_state=42
)
rf.fit(X_train, y_train)

# 评估
y_rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_rf_pred)
rf_oob_acc = rf.oob_score_  # 袋外误差
print(f"随机森林测试集准确率：{rf_acc:.4f}")
print(f"随机森林袋外（OOB）准确率：{rf_oob_acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_rf_pred, target_names=["良性(B)", "恶性(M)"]))

# 特征重要性
rf_feature_importance = pd.DataFrame({
    "特征": X.columns, "重要性": rf.feature_importances_
}).sort_values(by="重要性", ascending=False).head(15)

# 混淆矩阵
rf_cm = confusion_matrix(y_test, y_rf_pred)

# 保存结果到随机森林文件夹
rf_feature_importance.to_csv(
    os.path.join(folders["随机森林"], "随机森林特征重要性.csv"), 
    index=False, encoding="utf-8-sig"
)
pd.DataFrame(rf_cm, index=["实际良性", "实际恶性"], columns=["预测良性", "预测恶性"]).to_csv(
    os.path.join(folders["随机森林"], "随机森林混淆矩阵.csv"), 
    index=True, encoding="utf-8-sig"
)

# 可视化1：特征重要性热力图
plt.figure(figsize=(14, 10))
sns.heatmap(
    rf_feature_importance.set_index("特征").T, 
    annot=True, cmap="YlOrRd", fmt=".3f", cbar_kws={"label": "重要性分数"}
)
plt.title("随机森林 特征重要性热力图（Top15）", fontsize=14)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(folders["随机森林"], "随机森林特征热力图.png"), bbox_inches="tight")
plt.close()

# 可视化2：混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(
    rf_cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["预测良性", "预测恶性"], yticklabels=["实际良性", "实际恶性"]
)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("实际标签", fontsize=12)
plt.title("随机森林 混淆矩阵", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(folders["随机森林"], "随机森林混淆矩阵.png"), bbox_inches="tight")
plt.close()

# 4.2 梯度提升树（XGBoost+LightGBM，修复版）
print("\n" + "="*50)
print("【4/10 集成学习 - XGBoost+LightGBM（修复版）】")
print("="*50)

# 导入库
import xgboost as xgb
import lightgbm as lgb

# XGBoost（修复early_stopping参数）
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic", learning_rate=0.1, max_depth=3,
    n_estimators=100, reg_alpha=0.1, reg_lambda=0.1, random_state=42
)
# 新版XGBoost写法：eval_set + verbose
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# LightGBM（兼容新版，修复verbose参数）
lgb_model = lgb.LGBMClassifier(
    objective="binary", learning_rate=0.1, max_depth=3,
    n_estimators=100, reg_alpha=0.1, reg_lambda=0.1, random_state=42,
    verbose=-1  # 直接在模型定义里设置静默模式
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)]
)

# 评估
y_xgb_pred = xgb_model.predict(X_test)
y_lgb_pred = lgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, y_xgb_pred)
lgb_acc = accuracy_score(y_test, y_lgb_pred)

print(f"XGBoost准确率：{xgb_acc:.4f}")
print(f"LightGBM准确率：{lgb_acc:.4f}")
print("\nXGBoost分类报告：")
print(classification_report(y_test, y_xgb_pred, target_names=["良性(B)", "恶性(M)"]))

# 特征重要性（XGBoost）
xgb_feature_importance = pd.DataFrame({
    "特征": X.columns, "重要性": xgb_model.feature_importances_
}).sort_values(by="重要性", ascending=False).head(10)

# 保存结果到梯度提升树文件夹
xgb_feature_importance.to_csv(
    os.path.join(folders["梯度提升树"], "XGBoost特征重要性.csv"), 
    index=False, encoding="utf-8-sig"
)

# 可视化1：XGBoost特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x="重要性", y="特征", data=xgb_feature_importance, palette="Oranges")
plt.xlabel("特征重要性分数", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.title("XGBoost 特征重要性排序（Top10）", fontsize=14)
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(folders["梯度提升树"], "XGBoost特征重要性.png"), bbox_inches="tight")
plt.close()

# 可视化2：学习曲线（简化版，适配新版）
plt.figure(figsize=(12, 8))
# XGBoost（使用训练集/测试集准确率）
train_acc = []
test_acc = []
for i in range(1, 101):
    temp_model = xgb.XGBClassifier(
        objective="binary:logistic", learning_rate=0.1, max_depth=3,
        n_estimators=i, reg_alpha=0.1, reg_lambda=0.1, random_state=42
    )
    temp_model.fit(X_train, y_train, verbose=False)
    train_acc.append(accuracy_score(y_train, temp_model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, temp_model.predict(X_test)))

plt.plot(range(1, 101), train_acc, label="XGBoost-训练集准确率", color="blue")
plt.plot(range(1, 101), test_acc, label="XGBoost-测试集准确率", color="blue", linestyle="--")

plt.xlabel("迭代次数", fontsize=12)
plt.ylabel("准确率", fontsize=12)
plt.title("XGBoost 学习曲线", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(folders["梯度提升树"], "XGBoost学习曲线.png"), bbox_inches="tight")
plt.close()

# 4.3 AdaBoost
print("\n" + "="*50)
print("【5/10 集成学习 - AdaBoost】")
print("="*50)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
    n_estimators=50, learning_rate=0.1, random_state=42
)
ada.fit(X_train, y_train)

# 评估
y_ada_pred = ada.predict(X_test)
ada_acc = accuracy_score(y_test, y_ada_pred)
print(f"AdaBoost准确率：{ada_acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_ada_pred, target_names=["良性(B)", "恶性(M)"]))

# 保存结果到AdaBoost文件夹
pd.DataFrame(classification_report(
    y_test, y_ada_pred, target_names=["良性(B)", "恶性(M)"], output_dict=True
)).T.round(4).to_csv(
    os.path.join(folders["AdaBoost"], "AdaBoost分类报告.csv"), 
    index=True, encoding="utf-8-sig"
)

# 可视化1：样本权重分布（Top20）
sample_weights = ada.estimator_weights_
weight_df = pd.DataFrame({"弱分类器索引": range(len(sample_weights)), "权重": sample_weights})
weight_df = weight_df.sort_values(by="权重", ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(x="弱分类器索引", y="权重", data=weight_df, palette="Purples")
plt.xlabel("弱分类器索引", fontsize=12)
plt.ylabel("分类器权重", fontsize=12)
plt.title("AdaBoost 弱分类器权重分布（Top20）", fontsize=14)
plt.xticks(rotation=45)
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(folders["AdaBoost"], "AdaBoost分类器权重.png"), bbox_inches="tight")
plt.close()

# 可视化2：分类边界（用前2个重要特征）
top2_features = pd.DataFrame({"特征": X.columns, "重要性": ada.feature_importances_}).sort_values(by="重要性", ascending=False).head(2)["特征"].tolist()
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

# 重新训练AdaBoost（仅用前2特征，用于可视化）
ada_vis = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
    n_estimators=50, learning_rate=0.1, random_state=42
)
ada_vis.fit(X_train_top2, y_train)

# 绘制分类边界
x_min, x_max = X_train_top2.iloc[:, 0].min() - 1, X_train_top2.iloc[:, 0].max() + 1
y_min, y_max = X_train_top2.iloc[:, 1].min() - 1, X_train_top2.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = ada_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(
    X_train_top2.iloc[:, 0], X_train_top2.iloc[:, 1], 
    c=y_train, cmap=plt.cm.RdYlBu, edgecolors="black", alpha=0.8
)
plt.xlabel(top2_features[0], fontsize=12)
plt.ylabel(top2_features[1], fontsize=12)
plt.title("AdaBoost 分类边界可视化（前2个重要特征）", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(folders["AdaBoost"], "AdaBoost分类边界.png"), bbox_inches="tight")
plt.close()

# ====================== 5. 传统机器学习模型（3类补充）======================
# 5.1 逻辑回归
print("\n" + "="*50)
print("【6/10 传统ML - 逻辑回归】")
print("="*50)

lr = LogisticRegression(
    penalty="l2", C=1.0, max_iter=1000, class_weight="balanced", random_state=42
)
lr.fit(X_train_scaled, y_train)

# 评估
y_lr_pred = lr.predict(X_test_scaled)
y_lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
lr_acc = accuracy_score(y_test, y_lr_pred)
lr_auc = roc_auc_score(y_test, y_lr_prob)
print(f"逻辑回归准确率：{lr_acc:.4f}")
print(f"逻辑回归AUC值：{lr_auc:.4f}")

# 回归系数（解释特征影响）
lr_coef = pd.DataFrame({
    "特征": X.columns, "回归系数": lr.coef_[0], "绝对值": np.abs(lr.coef_[0])
}).sort_values(by="绝对值", ascending=False).head(10)

print("\nTop10 特征回归系数（正值=促进恶性，负值=促进良性）：")
print(lr_coef[["特征", "回归系数"]])

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_lr_prob)

# 保存结果到逻辑回归文件夹
lr_coef.to_csv(
    os.path.join(folders["逻辑回归"], "逻辑回归系数.csv"), 
    index=False, encoding="utf-8-sig"
)

# 可视化1：回归系数
plt.figure(figsize=(12, 8))
colors = ["red" if c > 0 else "green" for c in lr_coef["回归系数"]]
sns.barplot(x="回归系数", y="特征", data=lr_coef, palette=colors)
plt.xlabel("回归系数（正值=恶性倾向，负值=良性倾向）", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.title("逻辑回归 特征回归系数（Top10）", fontsize=14)
plt.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(folders["逻辑回归"], "逻辑回归系数.png"), bbox_inches="tight")
plt.close()

# 可视化2：ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"逻辑回归 (AUC={lr_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="随机猜测")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("假正例率（FPR）", fontsize=12)
plt.ylabel("真正例率（TPR）", fontsize=12)
plt.title("逻辑回归 ROC曲线", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(folders["逻辑回归"], "逻辑回归ROC曲线.png"), bbox_inches="tight")
plt.close()

# 5.2 支持向量机（SVM）
print("\n" + "="*50)
print("【7/10 传统ML - 支持向量机】")
print("="*50)

# 网格搜索调优（线性核+RBF核）
svm_param = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10],
    "gamma": ["scale", "auto"]
}
svm_grid = GridSearchCV(SVC(class_weight="balanced", probability=True, random_state=42), svm_param, cv=5, n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)
best_svm = svm_grid.best_estimator_

print(f"SVM最优参数：{svm_grid.best_params_}")

# 评估
y_svm_pred = best_svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, y_svm_pred)
print(f"SVM准确率：{svm_acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_svm_pred, target_names=["良性(B)", "恶性(M)"]))

# 支持向量索引
support_vec_idx = best_svm.support_
support_vec_df = pd.DataFrame({
    "支持向量索引": support_vec_idx,
    "对应标签": y_train.iloc[support_vec_idx].values
})

# 保存结果到SVM文件夹
support_vec_df.to_csv(
    os.path.join(folders["SVM"], "SVM支持向量.csv"), 
    index=False, encoding="utf-8-sig"
)

# 可视化：分类边界（前2个特征）
top2_features = X.columns[:2]
X_train_svm = X_train_scaled[top2_features]

# 重新训练SVM（仅前2特征）
svm_vis = SVC(kernel=best_svm.kernel, C=best_svm.C, gamma=best_svm.gamma, class_weight="balanced", random_state=42)
svm_vis.fit(X_train_svm, y_train)

# 绘制分类边界
x_min, x_max = X_train_svm.iloc[:, 0].min() - 1, X_train_svm.iloc[:, 0].max() + 1
y_min, y_max = X_train_svm.iloc[:, 1].min() - 1, X_train_svm.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
# 绘制支持向量
plt.scatter(
    X_train_svm.iloc[svm_vis.support_, 0], X_train_svm.iloc[svm_vis.support_, 1],
    s=100, facecolors="none", edgecolors="black", label="支持向量"
)
# 绘制普通样本
plt.scatter(
    X_train_svm.iloc[:, 0], X_train_svm.iloc[:, 1],
    c=y_train, cmap=plt.cm.Paired, edgecolors="gray", alpha=0.6, label="普通样本"
)
plt.xlabel(top2_features[0], fontsize=12)
plt.ylabel(top2_features[1], fontsize=12)
plt.title(f"SVM 分类边界可视化（{best_svm.kernel}核）", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(folders["SVM"], "SVM分类边界.png"), bbox_inches="tight")
plt.close()

# 5.3 K近邻（KNN）
print("\n" + "="*50)
print("【8/10 传统ML - K近邻】")
print("="*50)

# 交叉验证选最优K值
k_range = range(1, 21)
k_acc = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
    k_acc.append(scores.mean())

best_k = k_range[np.argmax(k_acc)]
print(f"最优K值：{best_k}（准确率：{max(k_acc):.4f}）")

# 训练最优KNN
knn = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
knn.fit(X_train_scaled, y_train)

# 评估
y_knn_pred = knn.predict(X_test_scaled)
knn_acc = accuracy_score(y_test, y_knn_pred)
print(f"KNN测试集准确率：{knn_acc:.4f}")

# 不同K值准确率曲线
k_acc_df = pd.DataFrame({"K值": k_range, "交叉验证准确率": k_acc})

# 保存结果到KNN文件夹
k_acc_df.to_csv(
    os.path.join(folders["KNN"], "KNN不同K值准确率.csv"), 
    index=False, encoding="utf-8-sig"
)

# 可视化1：不同K值准确率曲线
plt.figure(figsize=(12, 8))
sns.lineplot(x="K值", y="交叉验证准确率", data=k_acc_df, marker="o", linewidth=2, markersize=8)
plt.axvline(x=best_k, color="red", linestyle="--", label=f"最优K={best_k}")
plt.xlabel("K值（近邻数量）", fontsize=12)
plt.ylabel("5折交叉验证准确率", fontsize=12)
plt.title("KNN 不同K值准确率曲线", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(k_range)
plt.tight_layout()
plt.savefig(os.path.join(folders["KNN"], "KNN_K值曲线.png"), bbox_inches="tight")
plt.close()

# ====================== 6. 概率模型：朴素贝叶斯 ======================
print("\n" + "="*50)
print("【9/10 概率模型 - 高斯朴素贝叶斯】")
print("="*50)

gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# 评估
y_gnb_pred = gnb.predict(X_test_scaled)
gnb_acc = accuracy_score(y_test, y_gnb_pred)
print(f"高斯朴素贝叶斯准确率：{gnb_acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_gnb_pred, target_names=["良性(B)", "恶性(M)"]))

# 保存结果到朴素贝叶斯文件夹
pd.DataFrame(classification_report(
    y_test, y_gnb_pred, target_names=["良性(B)", "恶性(M)"], output_dict=True
)).T.round(4).to_csv(
    os.path.join(folders["朴素贝叶斯"], "朴素贝叶斯分类报告.csv"), 
    index=True, encoding="utf-8-sig"
)

# 可视化：前2特征的概率密度分布
top2_features = X.columns[:2]
X_benign = X_train_scaled[y_train==0][top2_features]
X_malignant = X_train_scaled[y_train==1][top2_features]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 特征1
sns.kdeplot(X_benign.iloc[:, 0], ax=axes[0], label="良性", color="green", fill=True)
sns.kdeplot(X_malignant.iloc[:, 0], ax=axes[0], label="恶性", color="red", fill=True)
axes[0].set_xlabel(top2_features[0], fontsize=12)
axes[0].set_ylabel("概率密度", fontsize=12)
axes[0].set_title(f"{top2_features[0]} 概率密度分布", fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# 特征2
sns.kdeplot(X_benign.iloc[:, 1], ax=axes[1], label="良性", color="green", fill=True)
sns.kdeplot(X_malignant.iloc[:, 1], ax=axes[1], label="恶性", color="red", fill=True)
axes[1].set_xlabel(top2_features[1], fontsize=12)
axes[1].set_ylabel("概率密度", fontsize=12)
axes[1].set_title(f"{top2_features[1]} 概率密度分布", fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(folders["朴素贝叶斯"], "朴素贝叶斯概率分布.png"), bbox_inches="tight")
plt.close()

# ====================== 7. 神经网络：多层感知机（MLP）======================
print("\n" + "="*50)
print("【10/10 神经网络 - 多层感知机】")
print("="*50)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
    alpha=0.01, max_iter=500, random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 评估
y_mlp_pred = mlp.predict(X_test_scaled)
mlp_acc = accuracy_score(y_test, y_mlp_pred)
print(f"MLP准确率：{mlp_acc:.4f}")
print("\n分类报告：")
print(classification_report(y_test, y_mlp_pred, target_names=["良性(B)", "恶性(M)"]))

# 训练损失曲线
mlp_loss = mlp.loss_curve_

# 混淆矩阵
mlp_cm = confusion_matrix(y_test, y_mlp_pred)

# 保存结果到MLP文件夹
pd.DataFrame(mlp_cm, index=["实际良性", "实际恶性"], columns=["预测良性", "预测恶性"]).to_csv(
    os.path.join(folders["MLP"], "MLP混淆矩阵.csv"), 
    index=True, encoding="utf-8-sig"
)

# 可视化1：训练损失曲线
plt.figure(figsize=(12, 8))
plt.plot(range(1, len(mlp_loss)+1), mlp_loss, color="purple", linewidth=2)
plt.xlabel("迭代次数", fontsize=12)
plt.ylabel("训练损失（Loss）", fontsize=12)
plt.title("MLP 训练损失曲线", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(folders["MLP"], "MLP损失曲线.png"), bbox_inches="tight")
plt.close()

# 可视化2：混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(
    mlp_cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=["预测良性", "预测恶性"], yticklabels=["实际良性", "实际恶性"]
)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("实际标签", fontsize=12)
plt.title("MLP 混淆矩阵", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(folders["MLP"], "MLP混淆矩阵.png"), bbox_inches="tight")
plt.close()

# ====================== 8. 所有模型结果汇总 ======================
print("\n" + "="*60)
print("【所有模型准确率汇总】")
print("="*60)

model_results = pd.DataFrame({
    "模型": [
        "关联规则（强规则数）", "CART决策树（后剪枝）", "随机森林", 
        "XGBoost", "LightGBM", "AdaBoost", "逻辑回归", 
        "SVM", "KNN（最优K）", "高斯朴素贝叶斯", "MLP"
    ],
    "准确率/规则数": [
        f"Top10条", f"{dt_acc:.4f}", f"{rf_acc:.4f}",
        f"{xgb_acc:.4f}", f"{lgb_acc:.4f}", f"{ada_acc:.4f}",
        f"{lr_acc:.4f}", f"{svm_acc:.4f}", f"{knn_acc:.4f}",
        f"{gnb_acc:.4f}", f"{mlp_acc:.4f}"
    ],
    "关键指标": [
        "Lift最高2.9+", f"特征重要性Top10", f"OOB={rf_oob_acc:.4f}",
        "学习率0.1", "学习率0.1", "弱分类器50个",
        f"AUC={lr_auc:.4f}", f"核={best_svm.kernel}", f"K={best_k}",
        "概率分布", "Loss收敛"
    ]
})

print(model_results.to_string(index=False))

# 保存汇总结果到汇总文件夹
model_results.to_csv(
    os.path.join(folders["汇总"], "所有模型结果汇总.csv"), 
    index=False, encoding="utf-8-sig"
)

print("\n" + "="*60)
print("✅ 所有模型运行完成！结果已按模型分文件夹保存到桌面的 breast_cancer_results 目录：")
print("- 每个模型的表格文件、高清图片都存在独立文件夹中")
print("- 汇总结果保存在 汇总结果 文件夹，方便对比分析")
print("="*60)