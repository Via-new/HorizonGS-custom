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

# ================= 全局变量 =================
CAM_STATE = {
    'x': 0.0, 'y': 0.0, 'z': 0.0,
    'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0
}
STATE_LOCK = threading.Lock()
IS_RUNNING = True
CLIENT_CONNECTED = False  # 连接状态标志
old_terminal_attr = None 

# === [显示模式标志] ===
# 默认全关
DEBUG_LEVEL_COLOR_FLAG = False
DEBUG_CONFLICT_RED_FLAG = False    # [模式1 X] 仅显示冲突点 (彩色/红色)
DEBUG_CONFLICT_NORMAL_FLAG = False # [模式2 V] 仅显示冲突点 (原色)
DEBUG_CONFLICT_EXCLUDE_FLAG = False # [模式3 B] 显示除冲突点外的所有点 (剔除模式)

CONFLICT_MASK = None               # 冲突点掩膜 (Tensor)
CONFLICT_COLORS = None             # 冲突点颜色 (Tensor [N, 3])
# =====================

MOVE_STEP = 0.002
ANGLE_STEP = 0.05
GLOBAL_SCALE_FACTOR = 1.0 
TARGET_FPS = 60.0

# ================= 终端控制 =================
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

# ================= 数学工具 =================
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
    # === [修改提示信息] ===
    print(">>> Keyboard: W/S/A/D/Q/E (Move), I/K/J/L/U/O (Rot)")
    print(">>> Modes: 'C'(Levels), 'X'(Conflict Color), 'V'(Conflict Norm), 'B'(Exclude Conflict) <<<")
    
    set_terminal_raw_mode()
    try:
        while IS_RUNNING:
            try:
                key = sys.stdin.read(1)
                if not key:
                    time.sleep(0.01)
                    continue
                
                key = key.lower()
                need_print_mode = False
                need_print_cam = False
                
                # === [模式切换逻辑] ===
                with STATE_LOCK:
                    global DEBUG_LEVEL_COLOR_FLAG, DEBUG_CONFLICT_RED_FLAG, DEBUG_CONFLICT_NORMAL_FLAG, DEBUG_CONFLICT_EXCLUDE_FLAG
                    
                    # 'C'键：切换 Level 颜色
                    if key == 'c':
                        DEBUG_LEVEL_COLOR_FLAG = not DEBUG_LEVEL_COLOR_FLAG
                        if DEBUG_LEVEL_COLOR_FLAG: 
                            DEBUG_CONFLICT_RED_FLAG = False
                            DEBUG_CONFLICT_NORMAL_FLAG = False
                            DEBUG_CONFLICT_EXCLUDE_FLAG = False
                        need_print_mode = True

                    # 'X'键：切换 仅显示冲突点 (彩色/红色)
                    elif key == 'x':
                        if CONFLICT_MASK is None:
                            print(f"\r[Err] No conflict mask! " + " "*30)
                        else:
                            DEBUG_CONFLICT_RED_FLAG = not DEBUG_CONFLICT_RED_FLAG
                            if DEBUG_CONFLICT_RED_FLAG: 
                                DEBUG_LEVEL_COLOR_FLAG = False
                                DEBUG_CONFLICT_NORMAL_FLAG = False
                                DEBUG_CONFLICT_EXCLUDE_FLAG = False
                            need_print_mode = True

                    # 'V'键：切换 仅显示冲突点 (原色)
                    elif key == 'v':
                        if CONFLICT_MASK is None:
                            print(f"\r[Err] No conflict mask! " + " "*30)
                        else:
                            DEBUG_CONFLICT_NORMAL_FLAG = not DEBUG_CONFLICT_NORMAL_FLAG
                            if DEBUG_CONFLICT_NORMAL_FLAG:
                                DEBUG_LEVEL_COLOR_FLAG = False
                                DEBUG_CONFLICT_RED_FLAG = False
                                DEBUG_CONFLICT_EXCLUDE_FLAG = False
                            need_print_mode = True

                    # 'B'键：剔除冲突点 (显示剩下的所有点)
                    elif key == 'b':
                        if CONFLICT_MASK is None:
                            print(f"\r[Err] No conflict mask! " + " "*30)
                        else:
                            DEBUG_CONFLICT_EXCLUDE_FLAG = not DEBUG_CONFLICT_EXCLUDE_FLAG
                            if DEBUG_CONFLICT_EXCLUDE_FLAG:
                                DEBUG_LEVEL_COLOR_FLAG = False
                                DEBUG_CONFLICT_RED_FLAG = False
                                DEBUG_CONFLICT_NORMAL_FLAG = False
                            need_print_mode = True
                # ====================

                # 打印模式状态
                if need_print_mode:
                    c = "ON" if DEBUG_LEVEL_COLOR_FLAG else "OFF"
                    x = "ON" if DEBUG_CONFLICT_RED_FLAG else "OFF"
                    v = "ON" if DEBUG_CONFLICT_NORMAL_FLAG else "OFF"
                    b = "ON" if DEBUG_CONFLICT_EXCLUDE_FLAG else "OFF"
                    print(f"\r[Mode] Levels(C):{c:<3} | Color(X):{x:<3} | Norm(V):{v:<3} | Excl(B):{b:<3}        ", end="", flush=True)

                # 相机移动处理
                with STATE_LOCK:
                    if key == 'w': CAM_STATE['y'] -= MOVE_STEP; need_print_cam = True
                    elif key == 's': CAM_STATE['y'] += MOVE_STEP; need_print_cam = True
                    elif key == 'a': CAM_STATE['z'] -= MOVE_STEP; need_print_cam = True
                    elif key == 'd': CAM_STATE['z'] += MOVE_STEP; need_print_cam = True
                    elif key == 'q': CAM_STATE['x'] -= MOVE_STEP; need_print_cam = True
                    elif key == 'e': CAM_STATE['x'] += MOVE_STEP; need_print_cam = True
                    elif key == 'i': CAM_STATE['yaw'] -= ANGLE_STEP;need_print_cam = True
                    elif key == 'k': CAM_STATE['yaw'] += ANGLE_STEP;  need_print_cam = True
                    elif key == 'j': CAM_STATE['roll'] += ANGLE_STEP; need_print_cam = True
                    elif key == 'l': CAM_STATE['roll'] -= ANGLE_STEP; need_print_cam = True
                    elif key == 'u': CAM_STATE['pitch'] -= ANGLE_STEP; need_print_cam = True
                    elif key == 'o': CAM_STATE['pitch'] += ANGLE_STEP; need_print_cam = True
                
                if need_print_cam:
                    print(f"\r[Cam] Pos: {CAM_STATE['x']:.2f},{CAM_STATE['y']:.2f},{CAM_STATE['z']:.2f} | Rot: {CAM_STATE['pitch']:.1f},{CAM_STATE['yaw']:.1f},{CAM_STATE['roll']:.1f}      ", end="", flush=True)

            except IOError: pass
            except Exception: pass
    finally:
        restore_terminal_mode()

# ================= 线程 2: 网络发送 =================
def socket_sender_thread(conn, img_queue):
    global CLIENT_CONNECTED
    print(f"[Sender] Thread Started. Scale: {GLOBAL_SCALE_FACTOR}")
    try:
        while IS_RUNNING and CLIENT_CONNECTED:
            try:
                img_np = img_queue.get(timeout=0.5) 
            except:
                continue 
            
            if img_np is None: break

            if img_np.shape[2] == 4: img_np = img_np[:, :, :3]
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            
            if GLOBAL_SCALE_FACTOR != 1.0:
                h, w = img_bgr.shape[:2]
                img_bgr = cv2.resize(img_bgr, (int(w*GLOBAL_SCALE_FACTOR), int(h*GLOBAL_SCALE_FACTOR)))

            success, encimg = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success: continue

            data = encimg.tobytes()
            try:
                conn.sendall(len(data).to_bytes(4, byteorder='big'))
                conn.sendall(data)
            except (BrokenPipeError, ConnectionResetError):
                print("\n[Sender] Client disconnected detected.")
                CLIENT_CONNECTED = False 
                break
    except Exception as e:
        print(f"[Sender] Error: {e}")
        CLIENT_CONNECTED = False

# ================= 阶段 1: 加载场景 =================
def load_scene_once(dataset, pipe, iteration, ape_code, explicit):
    print(">>> Loading Scene Models (This happens only once) <<<")
    with torch.no_grad():
        if pipe.no_prefilter_step > 0: pipe.add_prefilter = False
        else: pipe.add_prefilter = True
            
        scene_modules = __import__('scene')
        renderer_modules = __import__('gaussian_renderer') 
        
        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(scene_modules, model_config['name'])(**model_config['kwargs'])
        scene = scene_modules.Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit)
        gaussians.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        return scene, gaussians, background, renderer_modules

# ================= 阶段 2: 渲染循环 =================
def render_loop(scene, gaussians, background, renderer_modules, pipe, img_queue, start_image_name=None):
    global CAM_STATE
    with STATE_LOCK:
        CAM_STATE = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

    # --- 寻找初始视角 ---
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    selected_view = None
    
    if start_image_name:
        print(f"Searching for initial view: {start_image_name} ...")
        for cam in all_cameras:
            if start_image_name in cam.image_name:
                selected_view = cam
                break
        if not selected_view: print(f"Warning: {start_image_name} not found.")

    if selected_view is None:
        street_views = [c for c in all_cameras if c.image_type == "street"]
        aerial_views = [c for c in all_cameras if c.image_type == "aerial"]
        if street_views: selected_view = street_views[0]
        elif aerial_views: selected_view = aerial_views[0]
        elif all_cameras: selected_view = all_cameras[0]
        else: return # No cameras

    view = selected_view
    print(f"Start Rendering View: {view.image_name}")

    if isinstance(view.R, torch.Tensor): R_orig = view.R.cpu().numpy()
    else: R_orig = view.R.copy()
    if isinstance(view.T, torch.Tensor): T_orig = view.T.cpu().numpy()
    else: T_orig = view.T.copy()
    center_orig = -np.dot(R_orig.T, T_orig)

    target_frame_time = 1.0 / TARGET_FPS

    while IS_RUNNING and CLIENT_CONNECTED:
        loop_start_time = time.time()
        try:
            with STATE_LOCK:
                curr_x, curr_y, curr_z = CAM_STATE['x'], CAM_STATE['y'], CAM_STATE['z']
                curr_p, curr_yaw, curr_r = CAM_STATE['pitch'], CAM_STATE['yaw'], CAM_STATE['roll']
                
                cur_debug_color = DEBUG_LEVEL_COLOR_FLAG
                cur_conflict_red = DEBUG_CONFLICT_RED_FLAG
                cur_conflict_norm = DEBUG_CONFLICT_NORMAL_FLAG
                cur_conflict_exclude = DEBUG_CONFLICT_EXCLUDE_FLAG
                
            # === [应用 Level Color 模式] ===
            if hasattr(gaussians, 'debug_level_color'):
                gaussians.debug_level_color = cur_debug_color
            
            # === [准备 Render 参数] ===
            override_mask = None
            override_color = None
            
            if CONFLICT_MASK is not None:
                # 模式1 & 2: 仅显示冲突点
                if cur_conflict_red or cur_conflict_norm:
                    override_mask = CONFLICT_MASK
                    
                    # 模式1: 彩色显示 (如果数据存在) 或 红色 (回退)
                    if cur_conflict_red:
                        if CONFLICT_COLORS is not None:
                            override_color = CONFLICT_COLORS
                        else:
                            # 制造全红
                            num_anchors = gaussians.get_anchor.shape[0]
                            override_color = torch.zeros((num_anchors, 3), device="cuda")
                            override_color[:, 0] = 1.0 # Red
                
                # 模式3: 剔除冲突点 (显示剩下的)
                elif cur_conflict_exclude:
                    override_mask = ~CONFLICT_MASK
            # ============================
            
            # 计算新位姿
            new_center = center_orig + np.array([curr_x, curr_y, curr_z], dtype=np.float32)
            R_delta = get_rotation_matrix(curr_p, curr_yaw, curr_r)
            R_new = np.dot(R_delta, R_orig)
            T_new = -np.dot(R_new, new_center)
            
            view.R = R_new
            view.T = T_new
            view_matrix = getWorld2View2(view.R, view.T, view.trans, view.scale)
            view.world_view_transform = torch.tensor(view_matrix).transpose(0, 1).cuda()
            view.world_view_transform_inv = view.world_view_transform.inverse()
            view.camera_center = view.world_view_transform_inv[3, :3]

            # === [渲染调用] ===
            render_pkg = getattr(renderer_modules, 'render')(
                view, 
                gaussians, 
                pipe, 
                background,
                override_mask=override_mask,  # 过滤 Mask
                override_color=override_color # 颜色覆盖 (取代了 force_red)
            )
            # ====================================

            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            img_np = rendering.detach().permute(1, 2, 0).cpu().numpy()
            
            try:
                img_queue.put_nowait(img_np)
            except Full:
                pass 

            elapsed = time.time() - loop_start_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)

        except Exception as e:
            print(f"Render Loop Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n[Render Loop] Stopped (Client Disconnected).")

# ================= 主程序 =================
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--explicit", action="store_true")
    parser.add_argument("--port", default=12345, type=int)
    parser.add_argument("--scale_factor", default=1.0, type=float)
    parser.add_argument("--start_view", default=None, type=str)
    parser.add_argument("--fps_limit", default=60, type=int)
    args = parser.parse_args(sys.argv[1:])

    GLOBAL_SCALE_FACTOR = args.scale_factor
    TARGET_FPS = float(args.fps_limit)

    with open(os.path.join(args.model_path, "config.yaml")) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        lp.model_path = args.model_path

    # 1. 预加载场景
    safe_state(args.quiet)
    scene_data, gaussians_data, bg_data, r_mod = load_scene_once(lp, pp, args.iteration, args.ape, args.explicit)

    # ================= [加载冲突数据 (索引+颜色)] =================
    conflict_path = "debug_anchor_conflict/conflict_indices.pt"
    if not os.path.exists(conflict_path):
        conflict_path = "../debug_anchor_conflict/conflict_indices.pt"
    
    if os.path.exists(conflict_path):
        try:
            print(f"[Info] Loading Conflict Data from {conflict_path}...")
            loaded_data = torch.load(conflict_path)
            
            num_anchors = gaussians_data.get_anchor.shape[0]
            CONFLICT_MASK = torch.zeros(num_anchors, dtype=torch.bool, device="cuda")
            
            # 兼容新旧格式
            if isinstance(loaded_data, dict):
                indices = loaded_data['indices'].to("cuda")
                colors = loaded_data['colors'].to("cuda")
                
                # 填充全局颜色 Tensor (默认全黑，只有冲突点有色)
                CONFLICT_COLORS = torch.zeros((num_anchors, 3), device="cuda")
                CONFLICT_COLORS[indices] = colors
                
                CONFLICT_MASK[indices] = True
                print(f"[Info] Loaded V2.1 Format: {indices.shape[0]} points with COLORS.")
                
            else:
                # 旧格式: 只有 Tensor
                indices = loaded_data.to("cuda")
                CONFLICT_MASK[indices] = True
                CONFLICT_COLORS = None # 无颜色，将回退到全红
                print(f"[Info] Loaded V2.0 Format: {indices.shape[0]} points (Indices only).")
                
        except Exception as e:
            print(f"[Error] Failed to load conflict indices: {e}")
            CONFLICT_MASK = None
            CONFLICT_COLORS = None
    else:
        print(f"[Warn] Conflict indices not found. 'Conflict Only' mode disabled.")
    # ======================================================

    # 2. 启动键盘监听
    kb_thread = threading.Thread(target=keyboard_listener_thread, daemon=True)
    kb_thread.start()

    # 3. 网络服务主循环
    HOST = '0.0.0.0'
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, args.port))
    server_socket.listen(1)

    print(f"\n[Server] Ready on port {args.port}. Waiting for connections...")
    
    try:
        while IS_RUNNING:
            print("\nWaiting for client...")
            conn, addr = server_socket.accept()
            print(f"[Server] Connected: {addr}")
            
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            CLIENT_CONNECTED = True
            img_queue = Queue(maxsize=2)
            
            sender = threading.Thread(target=socket_sender_thread, args=(conn, img_queue), daemon=True)
            sender.start()
            
            render_loop(scene_data, gaussians_data, bg_data, r_mod, pp, img_queue, args.start_view)
            
            try:
                conn.close()
            except:
                pass
            print("[Server] Client session ended.")
            
    except KeyboardInterrupt:
        print("\nStopping Server...")
    finally:
        IS_RUNNING = False
        server_socket.close()
        restore_terminal_mode()