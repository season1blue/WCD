import os
import re
import pandas as pd

def parse_runtime_seconds_from_line(line):
    match = re.search(r'\[(\d+):(\d+)', line)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None

def parse_log_forgiving(file_path):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    results = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_layer = None
    last_runtime_sec = None

    for line in lines:
        # 记录新的 target_layer_idx（可能重复出现）
        if 'target_layer_idx' in line:
            match = re.search(r'(?:--)?target_layer_idx\s*=?\s*(\d+)', line)
            if match:
                current_layer = int(match.group(1))

        # 累积 Processing 行中最后一次出现的时间（用于最近运行时间）
        if 'Processing:' in line:
            time_sec = parse_runtime_seconds_from_line(line)
            if time_sec is not None:
                last_runtime_sec = time_sec

        # 遇到 Acc 行就记录一次数据
        if 'Acc:' in line:
            acc_match = re.search(r'Acc:\s*([\d\.]+)', line)
            if acc_match and current_layer is not None and last_runtime_sec is not None:
                acc = float(acc_match.group(1))
                results.append({
                    'log_file': dataset_name,
                    'target_layer_idx': current_layer,
                    'runtime_sec': last_runtime_sec,
                    'acc': acc
                })
                # 不清空 current_layer 和 runtime_sec，以支持多次记录相同层

    return results

def parse_all_logs_in_directory(directory='.'):
    all_results = []
    for fname in os.listdir(directory):
        if fname.endswith('.log'):
            full_path = os.path.join(directory, fname)
            all_results.extend(parse_log_forgiving(full_path))
    return all_results

# 主流程：读取、处理、导出
if __name__ == '__main__':
    all_log_results = parse_all_logs_in_directory()
    df_all = pd.DataFrame(all_log_results)

    # 输出结果
    print(df_all)
    df_all.to_csv("parsed_log_results.csv", index=False)
