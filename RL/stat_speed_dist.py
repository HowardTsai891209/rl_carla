import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 讀取 expert_labels.csv
csv_path = 'expert_labels.csv'
df = pd.read_csv(csv_path)

# 速度分布統計
bins = np.arange(0, 310, 10)  # 0, 10, ..., 300
labels = [f'{i}~{i+9}' for i in bins[:-1]]
df['speed_bin'] = pd.cut(df['speed'], bins=bins, labels=labels, right=False)

# 統計每個區間數量
speed_dist = df['speed_bin'].value_counts().sort_index()

print('Speed distribution (每10km/h一區間):')
for label, count in speed_dist.items():
    print(f'{label}: {count}')

# 每個速度區間下的控制均值、標準差
print('\n每個速度區間下的 steer/throttle/brake 統計:')
grouped = df.groupby('speed_bin')[['steer', 'throttle', 'brake']].agg(['mean', 'std', 'min', 'max', 'count'])
print(grouped)

# 條件分布分析：大轉彎時的 brake/speed
turning = df[np.abs(df['steer']) > 0.5]
print('\n大轉彎(steer>0.5)時 speed/brake 分布:')
print(turning[['speed', 'brake']].describe())

# 剎車時的 speed/steer 分布
braking = df[df['brake'] > 0.2]
print('\n剎車(brake>0.2)時 speed/steer 分布:')
print(braking[['speed', 'steer']].describe())

# 可視化
plt.figure(figsize=(10, 6))
sns.jointplot(x='speed', y='steer', data=df, kind='hex', gridsize=40)
plt.suptitle('Speed vs Steer')
plt.tight_layout()
plt.savefig('speed_vs_steer.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.jointplot(x='speed', y='brake', data=df, kind='hex', gridsize=40)
plt.suptitle('Speed vs Brake')
plt.tight_layout()
plt.savefig('speed_vs_brake.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.jointplot(x='speed', y='throttle', data=df, kind='hex', gridsize=40)
plt.suptitle('Speed vs Throttle')
plt.tight_layout()
plt.savefig('speed_vs_throttle.png')
plt.close()

print('\n已輸出 speed_vs_steer.png, speed_vs_brake.png, speed_vs_throttle.png 可視化圖檔。')
