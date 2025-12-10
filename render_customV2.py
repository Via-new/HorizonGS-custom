import os
import sys
import yaml
import torch
import numpy as np
import subprocess
import socket
import cv2
import struct
import threading
from queue import Queue, Full
from argparse import ArgumentParser
from utils.general_utils import safe_state, parse_cfg
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
    mat_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    mat_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    mat_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return np.dot(mat_z, np.dot(mat_y, mat_x))

def socket_sender_thread(conn, img_queue):
    """
    [独立线程] 消费者：从队列获取 Numpy 图像，编码并发送
    """
    print("[Sender Thread] Started.")
    try:
        while True:
            # 1. 从队列获取图像 (阻塞等待)
            img_np = img_queue.get()
            
            # None 是退出信号
            if img_np is None:
                break
            
            # 2. 图像预处理 (Alpha处理 + RGB转BGR)
            # 这里的计算是在发送线程做的，不会阻塞主线程的渲染
            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            
            # float(0-1) -> uint8(0-255)
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            # 3. JPEG 编码 (CPU耗时操作)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            success, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
            
            if not success:
                continue

            data = encimg.tobytes()
            size = len(data)

            # 4. 发送数据 (协议：4字节长度 + 数据体)
            conn.sendall(size.to_bytes(4, byteorder='big'))
            conn.sendall(data)
            
    except (ConnectionResetError, BrokenPipeError):
        print("[Sender Thread] Client disconnected.")
    except Exception as e:
        print(f"[Sender Thread] Error: {e}")
    finally:
        print("[Sender Thread] Exiting...")

def render_continuous_loop(img_queue, dataset, pipe, iteration, ape_code, explicit):
    """
    [主线程] 生产者：死循环渲染，将结果推入队列
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

        # 获取基础相机
        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        aerial_views = [c for c in all_cameras if c.image_type == "aerial"]
        if len(aerial_views) == 0:
            print("Error: No aerial views found!")
            return
        view = aerial_views[0] 
        print(f"\nUsing Base Camera: {view.image_name}")

        # ================= 相机初始化 (静态位置) =================
        # 你可以在这里加逻辑让它动起来，目前保持静态方便测试
        if isinstance(view.R, torch.Tensor): R_orig = view.R.cpu().numpy()
        else: R_orig = view.R.copy()
        if isinstance(view.T, torch.Tensor): T_orig = view.T.cpu().numpy()
        else: T_orig = view.T.copy()
        
        center_orig = -np.dot(R_orig.T, T_orig)
        move_x, move_y, move_z = 0.0, 0.0, 0.0
        new_center = center_orig + np.array([move_x, move_y, move_z], dtype=np.float32)
        
        delta_pitch, delta_yaw, delta_roll = 0.0, 0.0, 0.0
        R_delta = get_rotation_matrix(delta_pitch, delta_yaw, delta_roll)
        R_new = np.dot(R_delta, R_orig) 
        T_new = -np.dot(R_new, new_center)

        # 应用相机参数
        view.R = R_new
        view.T = T_new
        view_matrix = getWorld2View2(view.R, view.T, view.trans, view.scale) 
        view.world_view_transform = torch.tensor(view_matrix).transpose(0, 1).cuda() 
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        
        # 准备背景
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        print(">>> Starting Continuous Rendering Loop <<<")
        
        # Frame Counter for FPS calculation (optional)
        # start_time = time.time()
        # frames = 0

        # ================= 死循环渲染 =================
        while True:
            try:
                # 1. 渲染 (GPU)
                render_pkg = getattr(renderer_modules, 'render')(view, gaussians, pipe, background)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                
                # 2. 转 CPU Numpy (必要的传输开销)
                # permute: (C, H, W) -> (H, W, C)
                img_np = rendering.detach().permute(1, 2, 0).cpu().numpy()
                
                # 3. 放入队列 (非阻塞)
                # 如果队列满了(说明发送线程处理不过来/网络卡)，直接丢弃这一帧，不等待
                # 这样可以保证队列里永远是最新的帧，降低延迟
                try:
                    img_queue.put_nowait(img_np)
                except Full:
                    pass # Drop frame
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Render loop error: {e}")
                break

if __name__ == "__main__":
    parser = ArgumentParser(description="Real-time Custom Rendering Server")
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    parser.add_argument("--port", default=12345, type=int)
    args = parser.parse_args(sys.argv[1:])

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path

    HOST = '0.0.0.0'
    PORT = args.port
    
    print(f"==================================================")
    print(f"Server listening on {HOST}:{PORT}")
    print(f"==================================================")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        conn, addr = server_socket.accept()
        print(f"Client connected: {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # 禁用 Nagle

        # 创建一个队列，容量设为 2
        # 容量小意味着如果网络卡，渲染线程不会积压老图，而是直接丢弃，保证实时性
        img_queue = Queue(maxsize=2)

        # 启动发送线程
        sender = threading.Thread(target=socket_sender_thread, args=(conn, img_queue))
        sender.daemon = True # 设置为守护线程，主程序退出时它自动退出
        sender.start()

        # 初始化随机种子
        safe_state(args.quiet)
        
        # 主线程进入死循环渲染
        render_continuous_loop(img_queue, lp, pp, args.iteration, args.ape, args.explicit)
        
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        if 'conn' in locals(): conn.close()
        server_socket.close()