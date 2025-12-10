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
import termios
import fcntl
import time
from queue import Queue, Full
from argparse import ArgumentParser
from utils.general_utils import safe_state, parse_cfg
from utils.graphics_utils import getWorld2View2

# ================= GPU 设置 =================
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

# ================= 全局变量与锁 =================
CAM_STATE = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,      # 位置偏移
    'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0 # 角度偏移
}
STATE_LOCK = threading.Lock()
IS_RUNNING = True
old_terminal_attr = None 

# 步长设置
MOVE_STEP = 0.2
ANGLE_STEP = 0.2

# 全局配置 (由命令行参数设置)
GLOBAL_SCALE_FACTOR = 1.0 
TARGET_FPS = 60.0  # 目标FPS

# ================= 终端控制工具函数 =================
def set_terminal_raw_mode():
    global old_terminal_attr
    fd = sys.stdin.fileno()
    old_terminal_attr = termios.tcgetattr(fd)
    new_attr = termios.tcgetattr(fd)
    new_attr[3] &= ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSANOW, new_attr)
    fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)

def restore_terminal_mode():
    global old_terminal_attr
    if old_terminal_attr is not None:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSANOW, old_terminal_attr)

# ================= 数学工具函数 =================
def get_rotation_matrix(rx, ry, rz):
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)
    mat_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    mat_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    mat_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return np.dot(mat_z, np.dot(mat_y, mat_x))

# ================= 线程 1: 键盘监听 =================
def keyboard_listener_thread():
    print(">>> Keyboard Control Enabled <<<")
    print("Move: W/S (X), A/D (Y), Q/E (Z)")
    print("Rotate: I/K (Pitch), J/L (Yaw), U/O (Roll)")
    print("Press Ctrl+C to exit.")
    
    set_terminal_raw_mode()
    
    try:
        while IS_RUNNING:
            try:
                key = sys.stdin.read(1)
                if not key:
                    time.sleep(0.01)
                    continue
                
                key = key.lower()
                need_print = False 
                
                # with STATE_LOCK:
                #     if key == 'w': CAM_STATE['x'] += MOVE_STEP; need_print = True
                #     elif key == 's': CAM_STATE['x'] -= MOVE_STEP; need_print = True
                #     elif key == 'a': CAM_STATE['y'] += MOVE_STEP; need_print = True
                #     elif key == 'd': CAM_STATE['y'] -= MOVE_STEP; need_print = True
                #     elif key == 'q': CAM_STATE['z'] -= MOVE_STEP; need_print = True
                #     elif key == 'e': CAM_STATE['z'] += MOVE_STEP; need_print = True
                #     elif key == 'i': CAM_STATE['pitch'] += ANGLE_STEP; need_print = True
                #     elif key == 'k': CAM_STATE['pitch'] -= ANGLE_STEP; need_print = True
                #     elif key == 'j': CAM_STATE['yaw'] -= ANGLE_STEP; need_print = True
                #     elif key == 'l': CAM_STATE['yaw'] += ANGLE_STEP; need_print = True
                #     elif key == 'u': CAM_STATE['roll'] -= ANGLE_STEP; need_print = True
                #     elif key == 'o': CAM_STATE['roll'] += ANGLE_STEP; need_print = True

                with STATE_LOCK:
                    if key == 'w': CAM_STATE['y'] -= MOVE_STEP; need_print = True
                    elif key == 's': CAM_STATE['y'] += MOVE_STEP; need_print = True
                    elif key == 'a': CAM_STATE['z'] -= MOVE_STEP; need_print = True
                    elif key == 'd': CAM_STATE['z'] += MOVE_STEP; need_print = True
                    elif key == 'q': CAM_STATE['x'] -= MOVE_STEP; need_print = True
                    elif key == 'e': CAM_STATE['x'] += MOVE_STEP; need_print = True
                    elif key == 'i': CAM_STATE['yaw'] -= ANGLE_STEP;need_print = True
                    elif key == 'k': CAM_STATE['yaw'] += ANGLE_STEP;  need_print = True
                    elif key == 'j': CAM_STATE['roll'] += ANGLE_STEP; need_print = True
                    elif key == 'l': CAM_STATE['roll'] -= ANGLE_STEP; need_print = True
                    elif key == 'u': CAM_STATE['pitch'] -= ANGLE_STEP; need_print = True
                    elif key == 'o': CAM_STATE['pitch'] += ANGLE_STEP; need_print = True
                
                if need_print:
                    info_str = (
                        f"\r[Camera] Pos(X,Y,Z): {CAM_STATE['x']:.1f}, {CAM_STATE['y']:.1f}, {CAM_STATE['z']:.1f} | "
                        f"Rot(P,Y,R): {CAM_STATE['pitch']:.0f}, {CAM_STATE['yaw']:.0f}, {CAM_STATE['roll']:.0f}      "
                    )
                    print(info_str, end="", flush=True)

            except IOError:
                pass
            except Exception as e:
                pass
    finally:
        restore_terminal_mode()
        print("\nKeyboard listener stopped.")

# ================= 线程 2: 网络发送 =================
def socket_sender_thread(conn, img_queue):
    print(f"[Sender Thread] Started. Scale: {GLOBAL_SCALE_FACTOR}")
    try:
        while IS_RUNNING:
            try:
                img_np = img_queue.get(timeout=1.0) 
            except:
                continue
            
            if img_np is None: break

            if img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            if GLOBAL_SCALE_FACTOR != 1.0:
                h, w = img_bgr.shape[:2]
                new_w = int(w * GLOBAL_SCALE_FACTOR)
                new_h = int(h * GLOBAL_SCALE_FACTOR)
                img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            success, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
            
            if not success: continue

            data = encimg.tobytes()
            size = len(data)

            try:
                conn.sendall(size.to_bytes(4, byteorder='big'))
                conn.sendall(data)
            except (BrokenPipeError, ConnectionResetError):
                print("\n[Sender Thread] Client disconnected.")
                break
            
    except Exception as e:
        print(f"[Sender Thread] Error: {e}")

# ================= 线程 3 (主线程): 渲染循环 =================
def render_continuous_loop(img_queue, dataset, pipe, iteration, ape_code, explicit):
    with torch.no_grad():
        if pipe.no_prefilter_step > 0: pipe.add_prefilter = False
        else: pipe.add_prefilter = True
            
        print("Loading Scene...")
        scene_modules = __import__('scene')
        renderer_modules = __import__('gaussian_renderer')
        
        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(scene_modules, model_config['name'])(**model_config['kwargs'])
        scene = scene_modules.Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit)

        # === 添加调试代码 ===
        print(f"Total cameras loaded: {len(scene.getTrainCameras()) + len(scene.getTestCameras())}")
        all_cams = scene.getTrainCameras() + scene.getTestCameras()
        if len(all_cams) > 0:
            print(f"Sample camera name: {all_cams[0].image_name}")
            print(f"Sample camera type: {all_cams[0].image_type}")
            
        # 打印所有检测到的类型
        types = set([c.image_type for c in all_cams])
        print(f"Detected image types: {types}")
        # ===================

        gaussians.eval()

        all_cameras = scene.getTrainCameras() + scene.getTestCameras()
        aerial_views = [c for c in all_cameras if c.image_type == "aerial"]
        street_views = [c for c in all_cameras if c.image_type == "street"]
        
        if not aerial_views:
            print("Error: No aerial views found!")
            return
        view = aerial_views[12]
        # view = aerial_views[65]
        # view = street_views[87]
        if isinstance(view.R, torch.Tensor): R_orig = view.R.cpu().numpy()
        else: R_orig = view.R.copy()
        
        if isinstance(view.T, torch.Tensor): T_orig = view.T.cpu().numpy()
        else: T_orig = view.T.copy()
        
        center_orig = -np.dot(R_orig.T, T_orig)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        print(f">>> Rendering Loop Started (Limit: {TARGET_FPS} FPS) <<<")
        last_warning_time = 0.0
        
        # [关键] 计算目标帧间隔
        target_frame_time = 1.0 / TARGET_FPS

        while IS_RUNNING:
            loop_start_time = time.time() # 记录开始时间

            try:
                with STATE_LOCK:
                    curr_x = CAM_STATE['x']
                    curr_y = CAM_STATE['y']
                    curr_z = CAM_STATE['z']
                    curr_p = CAM_STATE['pitch']
                    curr_y_angle = CAM_STATE['yaw']
                    curr_r = CAM_STATE['roll']
                
                new_center = center_orig + np.array([curr_x, curr_y, curr_z], dtype=np.float32)
                R_delta = get_rotation_matrix(curr_p, curr_y_angle, curr_r)
                R_new = np.dot(R_delta, R_orig)
                T_new = -np.dot(R_new, new_center)
                
                view.R = R_new
                view.T = T_new
                view_matrix = getWorld2View2(view.R, view.T, view.trans, view.scale)
                view.world_view_transform = torch.tensor(view_matrix).transpose(0, 1).cuda()
                view.camera_center = view.world_view_transform.inverse()[3, :3]

                render_pkg = getattr(renderer_modules, 'render')(view, gaussians, pipe, background)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                
                img_np = rendering.detach().permute(1, 2, 0).cpu().numpy()
                
                try:
                    img_queue.put_nowait(img_np)
                except Full:
                    current_time = time.time()
                    if current_time - last_warning_time > 5.0: # 减少报警频率
                        print(f"\n[Warning] Send Queue Full! (Your GPU > Network).")
                        last_warning_time = current_time
                    pass

                # [关键] 动态休眠以稳定 FPS
                # 计算渲染和入队花费了多少时间
                elapsed = time.time() - loop_start_time
                # 如果花费时间少于目标间隔，就睡一会儿
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)

            except Exception as e:
                print(f"Render Error: {e}")
                break

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    parser.add_argument("--port", default=12345, type=int)
    parser.add_argument("--scale_factor", default=1.0, type=float)
    # [新增] FPS 限制参数
    parser.add_argument("--fps_limit", default=60, type=int, help="Target FPS limit to prevent queue overflow")
    args = parser.parse_args(sys.argv[1:])

    GLOBAL_SCALE_FACTOR = args.scale_factor
    TARGET_FPS = float(args.fps_limit)

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path

    HOST = '0.0.0.0'
    print(f"Waiting for connection on {HOST}:{args.port}...")
    print(f"Scale: {GLOBAL_SCALE_FACTOR}, Target FPS: {TARGET_FPS}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, args.port))
        server_socket.listen(1)
        conn, addr = server_socket.accept()
        print(f"Connected: {addr}")
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        img_queue = Queue(maxsize=2)

        kb_thread = threading.Thread(target=keyboard_listener_thread, daemon=True)
        kb_thread.start()

        sender = threading.Thread(target=socket_sender_thread, args=(conn, img_queue), daemon=True)
        sender.start()

        safe_state(args.quiet)
        render_continuous_loop(img_queue, lp, pp, args.iteration, args.ape, args.explicit)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        IS_RUNNING = False
        restore_terminal_mode()
        if 'conn' in locals(): conn.close()
        server_socket.close()
        print("Server closed.")