import os
import shutil
import re

# ======= 配置部分 =======
source_dir = r"D:\SCALM\SCALM-ALL\DAppSCAN-main\DAppSCAN-source\contracts"
target_dir = r"D:\SCALM\SCALM-ALL\SCALM\extracted_SWCs"

# ======= 初始化 =======
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

pattern = re.compile(r"//\s*SWC-", re.IGNORECASE)
count_total = 0
count_with_swc = 0
copied_files = []

# ======= 遍历所有 .sol 文件 =======
for root, _, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".sol"):
            count_total += 1
            file_path = os.path.join(root, file)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 检查是否包含 SWC 注释
                if pattern.search(content):
                    count_with_swc += 1
                    # 构建目标路径（保持文件夹结构可选）
                    rel_path = os.path.relpath(file_path, source_dir)
                    dest_path = os.path.join(target_dir, rel_path)

                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(dest_path)

            except Exception as e:
                print(f"⚠️  读取 {file_path} 出错: {e}")

# ======= 统计结果 =======
print("\n✅ 提取完成！")
print(f"扫描到的 .sol 文件总数: {count_total}")
print(f"包含 SWC 注释的文件数: {count_with_swc}")
print(f"已复制至: {target_dir}")

# 保存列表记录
log_path = os.path.join(target_dir, "swc_extracted_list.txt")
with open(log_path, 'w', encoding='utf-8') as log_file:
    for path in copied_files:
        log_file.write(path + "\n")

print(f"📄 已保存提取文件清单: {log_path}")
