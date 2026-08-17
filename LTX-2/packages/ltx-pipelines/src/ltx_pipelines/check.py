#!/usr/bin/env python3
"""
检查缩放后的视频文件夹：
- 视频是否存在、非空
- 视频高度是否为 240p（使用 ffprobe）
- 对应的 .wav 文件是否存在（仅当原视频旁有 .wav 时需要，可根据需要调整）

默认只打印有问题的文件。

用法：
    python check_resized_videos.py -i /path/to/output_folder
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

DEFAULT_FFPROBE = "/mnt/petrelfs/xuzhiyu/miniconda3/bin/ffprobe"

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def find_mp4_files(folder: Path) -> list[Path]:
    """递归查找文件夹下所有 .mp4 文件"""
    return sorted(folder.rglob("*.mp4"))


def check_video(video_path: Path, ffprobe_path: str, expected_height: int = 240) -> tuple[bool, str]:
    """
    检查单个视频：
    返回 (是否通过, 详细信息)
    """
    if not video_path.exists():
        return False, "文件不存在"
    if video_path.stat().st_size == 0:
        return False, "文件大小为 0"

    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=height",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        height_str = result.stdout.strip()
        if not height_str:
            return False, "未检测到视频流"
        height = int(height_str)
        if height != expected_height:
            return False, f"高度为 {height}，期望 {expected_height}"
        return True, f"高度正确 ({height})"
    except subprocess.CalledProcessError as e:
        return False, f"ffprobe 错误: {e.stderr.strip()}"
    except ValueError:
        return False, f"无法解析高度: {height_str}"
    except Exception as e:
        return False, f"未知错误: {e}"


def check_wav(video_path: Path) -> tuple[bool, str]:
    """检查对应 .wav 文件是否存在（仅检查存在性，不检查内容）"""
    wav_path = video_path.with_suffix(".wav")
    if wav_path.exists():
        size = wav_path.stat().st_size
        if size == 0:
            return False, "WAV 文件存在但大小为 0"
        return True, f"WAV 存在 ({size} bytes)"
    else:
        return False, "WAV 文件不存在"


def main():
    parser = argparse.ArgumentParser(description="检查缩放后的视频文件夹（默认只打印有问题的文件）")
    parser.add_argument("-i", "--input_dir", required=True, help="要检查的输出文件夹路径（包含缩放后的 .mp4 和 .wav）")
    parser.add_argument("--ffprobe", default=DEFAULT_FFPROBE, help=f"ffprobe 路径（默认: {DEFAULT_FFPROBE}）")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数（默认 4）")
    parser.add_argument("--check-wav", action="store_true", default=True, help="是否检查对应的 .wav 文件（默认启用）")
    parser.add_argument("--no-wav", dest="check_wav", action="store_false", help="不检查 WAV 文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印所有文件（包括正常的）")
    args = parser.parse_args()

    folder = Path(args.input_dir).resolve()
    if not folder.is_dir():
        print(f"错误：目录不存在 {folder}")
        sys.exit(1)

    ffprobe_path = args.ffprobe
    if not Path(ffprobe_path).is_file():
        print(f"错误：ffprobe 不存在于 {ffprobe_path}")
        sys.exit(1)

    mp4_files = find_mp4_files(folder)
    if not mp4_files:
        print(f"在 {folder} 中没有找到任何 .mp4 文件")
        sys.exit(0)

    total = len(mp4_files)
    print(f"找到 {total} 个视频文件")
    print(f"使用 ffprobe: {ffprobe_path}")
    print(f"并行线程数: {args.workers}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for idx, video_path in enumerate(mp4_files, 1):
            future = executor.submit(check_video, video_path, ffprobe_path)
            if args.check_wav:
                wav_future = executor.submit(check_wav, video_path)
                futures[future] = (idx, video_path, wav_future)
            else:
                futures[future] = (idx, video_path, None)

        for future in as_completed(futures):
            idx, video_path, wav_future = futures[future]
            video_ok, video_msg = future.result()
            wav_ok, wav_msg = (True, "未检查") if wav_future is None else wav_future.result()

            # 判断是否有问题
            has_problem = (not video_ok) or (args.check_wav and not wav_ok)
            if args.verbose or has_problem:
                status = "✓" if video_ok else "✗"
                safe_print(f"[{idx}/{total}] {status} {video_path.relative_to(folder)} : {video_msg}")
                if args.check_wav:
                    wav_status = "✓" if wav_ok else "⚠"
                    safe_print(f"        {wav_status} WAV: {wav_msg}")

            results.append({
                "path": video_path,
                "video_ok": video_ok,
                "video_msg": video_msg,
                "wav_ok": wav_ok,
                "wav_msg": wav_msg,
            })

    # 汇总报告
    bad_videos = [r for r in results if not r["video_ok"]]
    bad_wavs = [r for r in results if not r["wav_ok"]] if args.check_wav else []

    print("\n===== 检查完成 =====")
    print(f"视频正常: {total - len(bad_videos)}/{total}")
    if args.check_wav:
        print(f"WAV 正常: {total - len(bad_wavs)}/{total} (仅检查存在性)")

    if bad_videos:
        print("\n有问题的视频文件：")
        for r in bad_videos:
            print(f"  {r['path'].relative_to(folder)} : {r['video_msg']}")

    if args.check_wav and bad_wavs:
        print("\n缺失或为空的 WAV 文件：")
        for r in bad_wavs:
            print(f"  {r['path'].relative_to(folder)} : {r['wav_msg']}")

    if not bad_videos and (not args.check_wav or not bad_wavs):
        print("\n所有文件均正常！")

if __name__ == "__main__":
    main()