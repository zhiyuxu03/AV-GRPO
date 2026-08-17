import torch
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
# 确保可以导入项目内的模块，根据你的实际目录结构调整
# 假设该脚本放在项目根目录下，或者与 eval 文件夹同级
# project_root = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, project_root)

from eval.javisbench.src.metrics import calc_video_quality_score

def compute_video_quality(video_path: str, prompt: str, device=None):
    """
    计算单个视频的 visual quality 和 motion quality
    
    Args:
        video_path: 视频文件路径
        prompt: 对应的文本描述
        device: 计算设备，默认自动选择 cuda 或 cpu
    
    Returns:
        tuple: (visual_quality, motion_quality) 浮点数
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 该函数接受列表形式，返回两个 tensor
    visual_scores, motion_scores = calc_video_quality_score(
        [video_path], [prompt], device
    )
    
    # 提取标量值
    vq = visual_scores[0].item() if isinstance(visual_scores, torch.Tensor) else visual_scores[0]
    mq = motion_scores[0].item() if isinstance(motion_scores, torch.Tensor) else motion_scores[0]
    
    return vq, mq

if __name__ == "__main__":
    # ========== 在这里直接修改你的输入视频和提示 ==========
    VIDEO_PATH = "/mnt/petrelfs/zhiyuxu/AV-GRPO/outputs/ltx2_lora/samples/ref_step_000033_rank7_00.mp4"   # 替换为你的视频路径
    PROMPT = "woman and bamboo"  # 替换为对应的文本
    
    # ========== 计算并输出 ==========
    try:
        visual_quality, motion_quality = compute_video_quality(VIDEO_PATH, PROMPT)
        print(f"Visual Quality: {visual_quality:.4f}")
        print(f"Motion Quality: {motion_quality:.4f}")
    except Exception as e:
        print(f"计算出错: {e}")