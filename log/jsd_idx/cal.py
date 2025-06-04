import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# === 设置全局字体 ===
plt.rcParams.update({
    'font.size': 18,
    # 'font.family': 'Comic Sans MS',
})

# === 固定任务顺序 ===
task_order = ['vqav2', 'textvqa', 'vizwiz', 'gqa', 'okvqa', 'docvqa']
log_files = [f'log_{task}.log' for task in task_order]

# === 使用 pastel 淡色色系 ===
palette = plt.get_cmap('Pastel1')
predefined_labels = [0, 1, 2, 3, 4, 14, 15, 17, 19, 21, 22]
label_color_map = {label: palette(i % palette.N) for i, label in enumerate(predefined_labels)}
label_color_map["Other"] = "lightgray"

def extract_jsd_distribution(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    jsd_values = re.findall(r'jsd=\s*([\d\.]+)', text)
    jsd_values = [int(float(val)) for val in jsd_values]
    return Counter(jsd_values)

def simplify_counter(counter, top_n=6):
    most_common = counter.most_common(top_n)
    simplified = {k: v for k, v in most_common}
    other = sum(counter.values()) - sum(simplified.values())
    if other > 0:
        simplified["Other"] = other
    return simplified

def plot_single_pie(ax, counter, task_name, top_n=4):
    counter = simplify_counter(counter, top_n=top_n)
    total = sum(counter.values())

    sizes = []
    labels = []
    colors = []
    explode = []
    autopct_labels = []

    for k, v in counter.items():
        percentage = v / total
        sizes.append(v)
        labels.append(str(k))  # 简单显示标签内容
        colors.append(label_color_map.get(k, "gray"))
        explode.append(0.02 if percentage <= 0.05 else 0)  # 小比例的稍微突出
        autopct_labels.append(f"{percentage:.1%}" if percentage > 0.01 else "")

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        startangle=90,
        colors=colors,
        explode=explode,
        pctdistance=0.8,
        labeldistance=1.1,
        autopct='%1.1f%%'  # ✅ 添加这一行，确保返回三个值
    )

    # 设置比例文字（内部或外部）
    for i, a in enumerate(autotexts):
        if sizes[i] / total <= 0.05:
            # 小于 5%，标签引出图外
            a.set_position((1.2 * a.get_position()[0], 1.2 * a.get_position()[1]))
        a.set_fontsize(12)
        a.set_text(autopct_labels[i])

    ax.set_title(f'{task_name}')
    ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))


def main():
    task_order = ['vqav2', 'gqa', 'textvqa', 'okvqa', 'vizwiz', 'docvqa']
    log_files = [f'log_{task}.log' for task in task_order]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten()

    for idx, filename in enumerate(log_files):
        task_name = task_order[idx]
        if os.path.exists(filename):
            counter = extract_jsd_distribution(filename)
            plot_single_pie(axes[idx], counter, task_name)
        else:
            axes[idx].set_title(f"{task_name}: missing", fontsize=16)
            axes[idx].axis('off')

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("all_tasks_jsd_pie.pdf")
    plt.show()

if __name__ == "__main__":
    main()
