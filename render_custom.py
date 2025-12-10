import os
import sys
import yaml
import torch
import numpy as np
import subprocess
from argparse import ArgumentParser
from utils.general_utils import safe_state, parse_cfg
from utils.image_utils import save_rgba
from utils.graphics_utils import getWorld2View2

# ================= GPU 设置 =================
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
# ============================================

def get_rotation_matrix(rx, ry, rz):
    """
    根据欧拉角（角度制）生成旋转矩阵
    rx: Pitch (绕X轴)
    ry: Yaw   (绕Y轴)
    rz: Roll  (绕Z轴)
    """
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)

    mat_x = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    mat_y = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    mat_z = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    return np.dot(mat_z, np.dot(mat_y, mat_x))

def render_custom_view(dataset, pipe, iteration, ape_code, explicit):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0:
            pipe.add_prefilter = False
        else:
            pipe.add_prefilter = True
            
        print("Loading Scene and Gaussians...")
        scene_modules = __import__('scene')
        renderer_modules = __import__('gaussian_renderer')
        
        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(scene_modules, model_config['name'])(**model_config['kwargs'])
        scene = scene_modules.Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit)
        gaussians.eval()

        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        aerial_views = [c for c in all_cameras if c.image_type == "aerial"]
        
        if len(aerial_views) == 0:
            print("Error: No aerial views found!")
            return

        view = aerial_views[0] 
        print(f"\nUsing Base Camera: {view.image_name}")

        # =========================================================================================
        #                                   自定义视角修改区 (位置 + 旋转)
        # =========================================================================================
        
        # 兼容 NumPy 和 Tensor
        if isinstance(view.R, torch.Tensor):
            R_orig = view.R.cpu().numpy()
        else:
            R_orig = view.R.copy() 
            
        if isinstance(view.T, torch.Tensor):
            T_orig = view.T.cpu().numpy()
        else:
            T_orig = view.T.copy()
        
        center_orig = -np.dot(R_orig.T, T_orig)
        print(f"Original Center: {center_orig}")

        # -----------------------------------------------------------
        # [修改 1] 位置偏移
        # -----------------------------------------------------------
        move_x = 0.0   
        move_y = 0.0   
        move_z = 0.0   
        
        new_center = center_orig + np.array([move_x, move_y, move_z], dtype=np.float32)

        # -----------------------------------------------------------
        # [修改 2] 旋转偏移
        # -----------------------------------------------------------
        delta_pitch = 0.0 
        delta_yaw   = 0.0 
        delta_roll  = 0.0 

        R_delta = get_rotation_matrix(delta_pitch, delta_yaw, delta_roll)
        R_new = np.dot(R_delta, R_orig) 

        # -----------------------------------------------------------
        # [应用] 更新相机
        # -----------------------------------------------------------
        T_new = -np.dot(R_new, new_center)

        view.R = R_new
        view.T = T_new
        
        view_matrix = getWorld2View2(view.R, view.T, view.trans, view.scale) 
        view.world_view_transform = torch.tensor(view_matrix).transpose(0, 1).cuda() 
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        print(f"New Center     : {new_center}")
        print(f"Rotation Delta : Pitch={delta_pitch}, Yaw={delta_yaw}, Roll={delta_roll}")

        # =========================================================================================
        #                                   渲染与保存 (修复BUG部分)
        # =========================================================================================

        print(f"Rendering custom view...")
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_pkg = getattr(renderer_modules, 'render')(view, gaussians, pipe, background)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        
        # [关键修复]：手动添加 Alpha 通道 (RGB -> RGBA)
        # 检查当前是否只有3通道，如果是，添加全1的Alpha通道
        if rendering.shape[0] == 3:
            alpha = torch.ones((1, rendering.shape[1], rendering.shape[2]), device=rendering.device)
            rendering = torch.cat([rendering, alpha], dim=0)

        output_dir = os.path.join(dataset.model_path, "custom_renders")
        os.makedirs(output_dir, exist_ok=True)
        
        fname = f"render_x{move_x}_y{move_y}_z{move_z}_p{delta_pitch}_y{delta_yaw}_r{delta_roll}.png"
        save_path = os.path.join(output_dir, fname)
        
        # 现在 rendering 已经是 4 通道了，save_rgba 不会报错
        save_rgba(rendering, save_path)
        
        print(f"\n[Done] Result saved to: {save_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Custom Rendering Script")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path

    print(f"Rendering Custom View for: {args.model_path}")
    safe_state(args.quiet)

    render_custom_view(lp, pp, args.iteration, args.ape, args.explicit)