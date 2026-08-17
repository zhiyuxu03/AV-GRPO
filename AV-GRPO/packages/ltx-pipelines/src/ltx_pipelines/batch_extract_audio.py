import os
import subprocess
from pathlib import Path
import glob

def extract_audio_from_videos(input_dir, output_dir=None, sample_rate=16000):
    """
    批量从视频文件中提取音频，使用固定的ffmpeg路径
    """
    
    # 固定的ffmpeg路径
    ffmpeg_path = "/mnt/petrelfs/xuzhiyu/miniconda3/bin/ffmpeg"
    
    if not os.path.isfile(ffmpeg_path):
        print(f"错误: ffmpeg不存在于 {ffmpeg_path}")
        return
    
    print(f"使用ffmpeg: {ffmpeg_path}")
    
    # 设置输入和输出目录
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的视频格式
    video_extensions = ['*.mp4', '*.MP4', '*.avi', '*.mov', '*.mkv']
    
    # 查找所有视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_dir.glob(ext))
    
    if not video_files:
        print(f"在 {input_dir} 中没有找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 逐个处理视频文件
    success_count = 0
    for i, video_path in enumerate(video_files, 1):
        # 生成输出音频文件名（同名不同后缀）
        audio_path = output_dir / f"{video_path.stem}.wav"
        
        print(f"[{i}/{len(video_files)}] 处理: {video_path.name}")
        
        # 构建ffmpeg命令
        cmd = [
            ffmpeg_path,
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '2',
            '-y',
            str(audio_path)
        ]
        
        try:
            # 执行ffmpeg命令
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ 提取成功: {audio_path.name}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 提取失败: {e.stderr[:200]}...")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    print(f"\n完成！成功提取 {success_count}/{len(video_files)} 个音频文件")
    print(f"音频保存在: {output_dir}")

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python batch_extract_audio_final.py <视频文件夹路径> [输出文件夹路径]")
        print("示例: python batch_extract_audio_final.py /mnt/petrelfs/zhiyuxu/LTX-2/JavisBench")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 执行提取
    extract_audio_from_videos(input_dir, output_dir, sample_rate=16000)