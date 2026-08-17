csv_path = "/mnt/petrelfs/zhiyuxu/JavisDiT/JavisBench/JavisBench1.csv"

with open(csv_path, 'rb') as f:
    data = f.read()

# 找到所有 0xa1 的位置
positions = []
for i, byte in enumerate(data):
    if byte == 0xa1:
        # 打印该位置前后各50字节的上下文
        start = max(0, i-50)
        end = min(len(data), i+50)
        context = data[start:end]
        positions.append((i, context))
        if len(positions) >= 20:  # 最多找20处
            break

print(f"文件总大小: {len(data)} 字节")
print(f"找到 {len(positions)} 处 0xa1 字节\n")

for pos, context in positions:
    # 尝试用不同编码解码上下文
    print(f"\n--- 位置 {pos} (0x{pos:X}) ---")
    print(f"  原始字节: {context}")
    
    # 用 latin-1 看（能看到"字符"）
    try:
        latin_str = context.decode('latin-1')
        print(f"  latin-1: {repr(latin_str)}")
    except:
        pass
    
    # 用 utf-8 + errors='replace' 看
    try:
        utf8_str = context.decode('utf-8', errors='replace')
        print(f"  utf-8(replace): {repr(utf8_str)}")
    except:
        pass
    
    # 定位所在行
    line_start = data.rfind(b'\n', 0, pos) + 1
    line_end = data.find(b'\n', pos)
    if line_end == -1:
        line_end = len(data)
    line_num = data[:pos].count(b'\n') + 1
    print(f"  所在行号: 第 {line_num} 行")
    print(f"  该行内容(latin-1): {data[line_start:line_end].decode('latin-1', errors='replace')[:200]}")