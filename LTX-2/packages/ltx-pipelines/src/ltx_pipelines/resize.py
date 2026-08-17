#!/usr/bin/env python3
"""
批量将视频缩放为 240p，并复制对应的 WAV 文件（支持多线程并行处理）。

用法：
    python resize_videos_to_240p.py -i /path/to/videos -o /path/to/output
    python resize_videos_to_240p.py -i /path/to/videos -o /path/to/output --workers 8
    python resize_videos_to_240p.py -i /path/to/videos -o /path/to/output --ffmpeg /custom/ffmpeg
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 默认 ffmpeg 路径（根据你的环境修改）
DEFAULT_FFMPEG = "/mnt/petrelfs/xuzhiyu/miniconda3/bin/ffmpeg"

# 支持的视频扩展名（不区分大小写）
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}

# 线程安全的打印锁
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs)

def find_video_files(input_dir: Path) -> List[Path]:
    """递归查找所有视频文件"""
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_dir.rglob(f"*{ext}"))
        video_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(video_files))

def get_wav_file(video_path: Path) -> Path:
    """根据视频路径返回同名的 WAV 文件路径（仅改扩展名）"""
    return video_path.with_suffix(".wav")

def resize_video_to_240p(
    input_video: Path,
    output_video: Path,
    ffmpeg_path: str,
    crf: int = 18,
    preset: str = "fast",
) -> bool:
    """
    使用 ffmpeg 将视频缩放为 240p（高度 240，宽度自动偶数），不保留音频。
    返回是否成功。
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-i", str(input_video),
        "-vf", "scale=-2:240",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-an",
        "-y",
        str(output_video),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False

def copy_wav_file(input_wav: Path, output_wav: Path) -> bool:
    """复制 WAV 文件到目标路径"""
    if not input_wav.exists():
        return False
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(input_wav, "rb") as src, open(output_wav, "wb") as dst:
            dst.write(src.read())
        return True
    except Exception:
        return False

def process_single_task(
    idx: int,
    total: int,
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    ffmpeg_path: str,
    crf: int,
    preset: str,
    no_audio: bool,
) -> Tuple[bool, bool, str]:
    """
    处理单个视频：缩放视频 + 复制 WAV。
    返回 (视频成功, WAV成功, 相对路径字符串)
    """
    rel_path = video_path.relative_to(input_dir)
    output_video = output_dir / rel_path.with_suffix(".mp4")
    
    safe_print(f"[{idx}/{total}] 开始处理: {rel_path}")
    
    # 缩放视频
    video_ok = resize_video_to_240p(video_path, output_video, ffmpeg_path, crf, preset)
    if not video_ok:
        safe_print(f"[{idx}/{total}] ✗ 视频缩放失败: {rel_path}")
        return False, False, str(rel_path)
    
    safe_print(f"[{idx}/{total}] ✓ 视频缩放成功: {rel_path}")
    
    # 复制 WAV
    wav_ok = False
    if not no_audio:
        input_wav = get_wav_file(video_path)
        output_wav = output_dir / rel_path.with_suffix(".wav")
        if copy_wav_file(input_wav, output_wav):
            wav_ok = True
            safe_print(f"[{idx}/{total}] ✓ 已复制 WAV: {input_wav.name}")
        else:
            safe_print(f"[{idx}/{total}] ⚠ 未找到 WAV 文件或复制失败: {input_wav.name}")
    
    return video_ok, wav_ok, str(rel_path)

def main():
    parser = argparse.ArgumentParser(description="将视频缩放为 240p 并复制同名 WAV 文件（多线程）")
    parser.add_argument("-i", "--input_dir", required=True, help="输入文件夹路径")
    parser.add_argument("-o", "--output_dir", required=True, help="输出文件夹路径")
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG, help=f"ffmpeg 可执行文件路径（默认: {DEFAULT_FFMPEG}）")
    parser.add_argument("--crf", type=int, default=18, help="视频编码 CRF 质量（默认 18，越小质量越好）")
    parser.add_argument("--preset", default="fast", help="编码预设（如 ultrafast, fast, medium, slow）")
    parser.add_argument("--no-audio", action="store_true", help="是否不复制 WAV 文件（仅处理视频）")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数（默认 4）")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"错误：输入目录不存在: {input_dir}")
        sys.exit(1)

    ffmpeg_path = args.ffmpeg
    if not os.path.isfile(ffmpeg_path):
        print(f"错误：ffmpeg 不存在于 {ffmpeg_path}")
        print("请确认路径正确，或使用 --ffmpeg 指定正确的 ffmpeg")
        sys.exit(1)

    video_files = find_video_files(input_dir)
    if not video_files:
        print(f"在 {input_dir} 中未找到任何视频文件（扩展名: {', '.join(VIDEO_EXTENSIONS)}）")
        sys.exit(0)

    total = len(video_files)
    print(f"找到 {total} 个视频文件")
    print(f"输出目录: {output_dir}")
    print(f"并行线程数: {args.workers}")
    print()

    success_video = 0
    success_wav = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_single_task,
                idx, total,
                video_path,
                input_dir,
                output_dir,
                ffmpeg_path,
                args.crf,
                args.preset,
                args.no_audio,
            ): video_path
            for idx, video_path in enumerate(video_files, 1)
        }
        
        for future in as_completed(futures):
            video_ok, wav_ok, rel_path = future.result()
            if video_ok:
                success_video += 1
            if wav_ok:
                success_wav += 1

    print()
    print("===== 处理完成 =====")
    print(f"成功缩放视频: {success_video}/{total}")
    if not args.no_audio:
        print(f"成功复制 WAV: {success_wav}/{total} (仅统计存在 WAV 的文件)")

if __name__ == "__main__":
    main()