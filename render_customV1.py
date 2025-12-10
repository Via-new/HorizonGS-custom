import os
import sys
import yaml
import torch
import numpy as np
import subprocess
import socket
import cv2  # 需要安装: pip install opencv-python
import struct
from argparse import ArgumentParser
from utils.general_utils import safe_state, parse_cfg
# from utils.image_utils import save_rgba # 不需要保存本地了
from utils.graphics_utils import getWorld2View2

# ================= GPU 设置 =================
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
# ============================================

def get_rotation_matrix(rx, ry, rz):
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

def send_image_to_client(conn, rendering_tensor):
    """
    将Tensor图像转换并发送给客户端
    rendering_tensor: (C, H, W) RGB, value 0-1
    """
    try:
        # 1. Tensor (GPU) -> Numpy (CPU)
        # 形状从 (C, H, W) 变为 (H, W, C)
        img_np = rendering_tensor.detach().permute(1, 2, 0).cpu().numpy()
        
        # 2. 只有3通道时无需处理Alpha，直接由0-1 float转为0-255 uint8
        # 如果是4通道，client端cv2.imdecode也能处理，但JPEG不支持Alpha，这里转为BGR以防万一
        if img_np.shape[2] == 4:
            img_np = img_np[:, :, :3] # 丢弃Alpha通道，为了JPEG编码
            
        img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        
        # 3. RGB -> BGR (因为OpenCV使用BGR)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        
        # 4. 编码为JPEG (质量90，平衡速度和画质)
        # 使用JPEG可以极大减少网络带宽占用，提高帧率
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        
        if not result:
            print("Image encode failed!")
            return

        data = encimg.tobytes()
        size = len(data)

        # 5. 发送协议：先发4字节长度(Big Endian)，再发数据
        # 对应客户端: int.from_bytes(conn.recv(4), 'big')
        conn.sendall(size.to_bytes(4, byteorder='big'))
        conn.sendall(data)
        
        print(f"Sent frame size: {size/1024:.2f} KB")
        
    except Exception as e:
        print(f"Sending failed: {e}")

def render_custom_view(conn, dataset, pipe, iteration, ape_code, explicit):
    """
    conn: Socket连接对象
    """
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

        # ==================== 自定义视角修改区 ====================
        
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

        # [修改 1] 位置偏移
        move_x = 0.0   
        move_y = 0.0   
        move_z = 0.0   
        new_center = center_orig + np.array([move_x, move_y, move_z], dtype=np.float32)

        # [修改 2] 旋转偏移
        delta_pitch = 0.0 
        delta_yaw   = 0.0 
        delta_roll  = 0.0 

        R_delta = get_rotation_matrix(delta_pitch, delta_yaw, delta_roll)
        R_new = np.dot(R_delta, R_orig) 
        T_new = -np.dot(R_new, new_center)

        view.R = R_new
        view.T = T_new
        
        view_matrix = getWorld2View2(view.R, view.T, view.trans, view.scale) 
        view.world_view_transform = torch.tensor(view_matrix).transpose(0, 1).cuda() 
        view.camera_center = view.world_view_transform.inverse()[3, :3]

        print(f"Rendering custom view...")
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_pkg = getattr(renderer_modules, 'render')(view, gaussians, pipe, background)
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        
        # ==================== 发送结果到本地 ====================
        
        # 直接调用发送函数，不再保存到本地磁盘
        print("Sending image to client...")
        send_image_to_client(conn, rendering)
        print("Done.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Custom Rendering Script")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    parser.add_argument("--port", default=12345, type=int, help="Port to listen on")
    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path

    # ==================== 网络服务器设置 ====================
    HOST = '0.0.0.0' # 监听所有网卡
    PORT = args.port
    
    print(f"==================================================")
    print(f"Waiting for connection on {HOST}:{PORT} ...")
    print(f"Please run your local client script NOW.")
    print(f"==================================================")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 允许端口复用，防止程序退出后端口被占用
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        
        # 阻塞等待客户端连接
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        
        # 禁用Nagle算法，降低延迟
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        safe_state(args.quiet)
        
        # 将连接对象传给渲染函数
        render_custom_view(conn, lp, pp, args.iteration, args.ape, args.explicit)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
        server_socket.close()
        print("Connection closed.")