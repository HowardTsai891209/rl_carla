import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 取得該 Python 腳本所在的資料夾
script_dir = os.path.dirname(os.path.abspath(__file__))

# 設定 log 資料夾與輸出資料夾
log_root = os.path.join(script_dir, "logs/1749455415")
output_dir = os.path.join(script_dir, "visualize data")
os.makedirs(output_dir, exist_ok=True)

# 要視覺化的 scalar 名稱
scalar_tags = [
    "rollout/ep_len_mean",
    "rollout/ep_rew_mean",
    "train/entropy_loss",
    "train/explained_variance",
    "train/value_loss",
    "train/policy_gradient_loss",
    "train/loss"
]

# 搜集所有 PPO_Iter 資料夾（依 Iter 數排序）
subdirs = sorted(
    [d for d in os.listdir(log_root) if d.startswith("PPO_Iter")],
    key=lambda x: int(x.split("Iter")[1].split("_")[0])
)

# 用 dict 儲存所有 scalar 的數據
scalar_data = {tag: {"steps": [], "values": []} for tag in scalar_tags}

# 讀取 tfevents 檔案並提取資料
for subdir in subdirs:
    event_path = os.path.join(log_root, subdir)
    files = os.listdir(event_path)
    event_file = next((f for f in files if f.startswith("events.out.tfevents")), None)
    if not event_file:
        continue

    event_file_path = os.path.join(event_path, event_file)
    ea = EventAccumulator(event_file_path)
    ea.Reload()

    available_tags = ea.Tags()["scalars"]

    for tag in scalar_tags:
        if tag not in available_tags:
            print(f"[!] scalar '{tag}' not found in {event_file_path}")
            continue

        events = ea.Scalars(tag)
        scalar_data[tag]["steps"].extend([e.step for e in events])
        scalar_data[tag]["values"].extend([e.value for e in events])

# 繪圖並儲存至 visualize data 資料夾
for tag in scalar_tags:
    steps = scalar_data[tag]["steps"]
    values = scalar_data[tag]["values"]

    if not steps or not values:
        print(f"[!] No data found for '{tag}', skipping plot.")
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, label=tag, color="blue")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(tag.replace("_", " ").title())
    plt.grid(True)
    plt.tight_layout()

    # 將 tag 中的 "/" 換成 "_" 以用作合法檔名
    filename = tag.replace("/", "_") + ".png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()

print(f"所有圖已儲存到資料夾：{output_dir}")
