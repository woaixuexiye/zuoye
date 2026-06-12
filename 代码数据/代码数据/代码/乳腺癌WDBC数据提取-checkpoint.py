import pandas as pd
import os
from pathlib import Path

# 读取桌面数据
desktop = str(Path.home() / "Desktop")
file = os.path.join(desktop, "wdbc.data")

# 列名
cols = [
    "ID","diagnosis",
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
]

df = pd.read_csv(file, header=None, names=cols)

# 1. 前5行
print("="*50)
print("前5行数据")
print(df.head())

# 2. 数据基本信息
print("\n" + "="*50)
print("数据基本信息")
df.info()

# 3. 良恶性数量
print("\n" + "="*50)
print("良性B、恶性M数量")
print(df["diagnosis"].value_counts())

# 4. 特征统计
print("\n" + "="*50)
print("所有特征统计指标")
print(df.iloc[:,2:].describe().round(4))

# 【新增】导出全部数据到桌面的CSV文件
output_file = os.path.join(desktop, "乳腺癌完整数据.csv")
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✅ 全部数据已导出到桌面：{output_file}")