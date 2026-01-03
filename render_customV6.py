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
CLIENT_CONNECTED = False 
old_terminal_attr = None 

# === [显示模式标志] ===
DEBUG_LEVEL_COLOR_FLAG = False
DEBUG_CONFLICT_RED_FLAG = False    
DEBUG_CONFLICT_NORMAL_FLAG = False 
DEBUG_CONFLICT_CLASS_FLAG = False  # 'B' 模式主开关

# 冲突相关数据
CONFLICT_MASK_ALL = None           # 所有冲突点的 Mask
CONFLICT_COLORS = None             # 所有冲突点的颜色
CONFLICT_SUB_MASKS = {}            # { 'Green': mask, 'Red': mask ... }
CURRENT_SUB_MODE_IDX = 0           # 当前显示的子类别索引
SUB_MODES = ["ALL"]                # 子类别列表

FORCE_OPAQUE = False               # 默认为 False
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
    rx = np.radians(rx); ry = np.radians(ry); rz = np.radians(rz)
    mat_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    mat_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    mat_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return np.dot(mat_z, np.dot(mat_y, mat_x))

# ================= 线程 1: 键盘监听 =================
def keyboard_listener_thread():
    print(">>> Controls: W/S/A/D/Q/E (Move), I/K/J/L/U/O (Rot), Arrows (Adjust Steps)")
    print(">>> Modes: 'B'(Conflict Mode), 'N'(Next Category), 'M'(Toggle Opaque), 'C'(Levels)")
    
    set_terminal_raw_mode()
    try:
        while IS_RUNNING:
            try:
                key = sys.stdin.read(1)
                if not key:
                    time.sleep(0.01)
                    continue
                
                
                # === [新增: 方向键调整步长] ===
                if key == '\x1b':
                    try:
                        seq = sys.stdin.read(2)
                        if seq and len(seq) == 2:
                            global MOVE_STEP, ANGLE_STEP
                            updated = False
                            if seq == '[A': # Up: Increase MOVE_STEP
                                MOVE_STEP += 0.001
                                updated = True
                            elif seq == '[B': # Down: Decrease MOVE_STEP
                                MOVE_STEP = max(0.0001, MOVE_STEP - 0.001)
                                updated = True
                            elif seq == '[D': # Left: Increase ANGLE_STEP
                                ANGLE_STEP += 0.01
                                updated = True
                            elif seq == '[C': # Right: Decrease ANGLE_STEP
                                ANGLE_STEP = max(0.01, ANGLE_STEP - 0.01)
                                updated = True
                            
                            if updated:
                                print(f"\r[Step] Move: {MOVE_STEP:.4f} | Angle: {ANGLE_STEP:.3f}      ", end="", flush=True)
                                continue
                    except IOError: pass
                # ==============================
                key = key.lower()
                need_print_mode = False
                need_print_cam = False
                
                with STATE_LOCK:
                    global DEBUG_LEVEL_COLOR_FLAG, DEBUG_CONFLICT_RED_FLAG, DEBUG_CONFLICT_NORMAL_FLAG, DEBUG_CONFLICT_CLASS_FLAG
                    global CURRENT_SUB_MODE_IDX, FORCE_OPAQUE
                    
                    # 'C': Level Color
                    if key == 'c':
                        DEBUG_LEVEL_COLOR_FLAG = not DEBUG_LEVEL_COLOR_FLAG
                        if DEBUG_LEVEL_COLOR_FLAG: 
                            DEBUG_CONFLICT_RED_FLAG = False
                            DEBUG_CONFLICT_NORMAL_FLAG = False
                            DEBUG_CONFLICT_CLASS_FLAG = False
                        need_print_mode = True

                    # 'B': Conflict Class Mode (Main Switch)
                    elif key == 'b':
                        if CONFLICT_MASK_ALL is None:
                            print(f"\r[Err] No conflict mask! " + " "*30)
                        else:
                            DEBUG_CONFLICT_CLASS_FLAG = not DEBUG_CONFLICT_CLASS_FLAG
                            if DEBUG_CONFLICT_CLASS_FLAG:
                                DEBUG_LEVEL_COLOR_FLAG = False
                                DEBUG_CONFLICT_RED_FLAG = False
                                DEBUG_CONFLICT_NORMAL_FLAG = False
                            need_print_mode = True

                    # 'N': Next Category (Cycle Sub-modes)
                    elif key == 'n':
                        if DEBUG_CONFLICT_CLASS_FLAG:
                            CURRENT_SUB_MODE_IDX = (CURRENT_SUB_MODE_IDX + 1) % len(SUB_MODES)
                            need_print_mode = True

                    # 'M': Toggle Force Opaque (Solid vs Transparent)
                    elif key == 'm':
                        FORCE_OPAQUE = not FORCE_OPAQUE
                        need_print_mode = True

                    # 'V': Conflict Norm (Original Colors)
                    elif key == 'v':
                        if CONFLICT_MASK_ALL is None:
                            print(f"\r[Err] No conflict mask! " + " "*30)
                        else:
                            DEBUG_CONFLICT_NORMAL_FLAG = not DEBUG_CONFLICT_NORMAL_FLAG
                            if DEBUG_CONFLICT_NORMAL_FLAG:
                                DEBUG_LEVEL_COLOR_FLAG = False
                                DEBUG_CONFLICT_RED_FLAG = False
                                DEBUG_CONFLICT_CLASS_FLAG = False
                            need_print_mode = True

                    # 'X': Conflict Red (Fallback)
                    elif key == 'x':
                        if CONFLICT_MASK_ALL is None:
                            print(f"\r[Err] No conflict mask! " + " "*30)
                        else:
                            DEBUG_CONFLICT_RED_FLAG = not DEBUG_CONFLICT_RED_FLAG
                            if DEBUG_CONFLICT_RED_FLAG: 
                                DEBUG_LEVEL_COLOR_FLAG = False
                                DEBUG_CONFLICT_NORMAL_FLAG = False
                                DEBUG_CONFLICT_CLASS_FLAG = False
                            need_print_mode = True

                # Print Status
                if need_print_mode:
                    c_status = "ON" if DEBUG_LEVEL_COLOR_FLAG else "OFF"
                    b_status = f"ON [{SUB_MODES[CURRENT_SUB_MODE_IDX]}]" if DEBUG_CONFLICT_CLASS_FLAG else "OFF"
                    o_status = "Solid" if FORCE_OPAQUE else "Alpha"
                    print(f"\r[Mode] Levels(C):{c_status:<3} | Conflict(B):{b_status:<15} | Style(M):{o_status}      ", end="", flush=True)

                # Camera Move
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
            except:
                print("\n[Sender] Client disconnected.")
                CLIENT_CONNECTED = False 
                break
    except Exception as e:
        print(f"[Sender] Error: {e}")
        CLIENT_CONNECTED = False

# ================= 阶段 1: 加载场景 =================
def load_scene_once(dataset, pipe, iteration, ape_code, explicit):
    print(">>> Loading Scene Models... <<<")
    with torch.no_grad():
        if pipe.no_prefilter_step > 0: pipe.add_prefilter = False
        else: pipe.add_prefilter = True
            
        scene_modules = __import__('scene')
        # [修改] 使用 Scheme 1 渲染器
        from gaussian_renderer import render_scheme1 as renderer_modules
        print("[INFO] Loaded Scheme 1 Renderer (Mask-aware)")

        model_config = dataset.model_config
        model_config['kwargs']['ape_code'] = ape_code
        gaussians = getattr(scene_modules, model_config['name'])(**model_config['kwargs'])
        scene = scene_modules.Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, explicit=explicit)
        gaussians.eval()

        # 加载 Scheme 1 的 Mask (如果有)
        mask_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "anchor_source_mask.pt")
        # 尝试不同路径
        if not os.path.exists(mask_path):
             mask_path = os.path.join(dataset.model_path, "anchor_source_mask.pt")

        if os.path.exists(mask_path):
            print(f"[INFO] Found Anchor Source Mask: {mask_path}")
            mask_tensor = torch.load(mask_path, map_location="cuda")
            if mask_tensor.shape[0] == gaussians.get_anchor.shape[0]:
                gaussians.anchor_source_mask = mask_tensor
                print("[INFO] Mask loaded successfully.")
            else:
                print(f"[WARN] Mask shape {mask_tensor.shape} mismatch with anchors {gaussians.get_anchor.shape}!")
        else:
             print("[WARN] No Anchor Source Mask found. Scheme 1 will run in default mode (All shared).")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return scene, gaussians, background, renderer_modules

# ================= 阶段 2: 渲染循环 =================
def render_loop(scene, gaussians, background, renderer_modules, pipe, img_queue, start_image_name=None):
    global CAM_STATE
    with STATE_LOCK:
        CAM_STATE = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}

    # Find initial view
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
                
                show_level_color = DEBUG_LEVEL_COLOR_FLAG
                show_conflict = DEBUG_CONFLICT_CLASS_FLAG
                show_red = DEBUG_CONFLICT_RED_FLAG
                show_norm = DEBUG_CONFLICT_NORMAL_FLAG
                
                sub_mode_name = SUB_MODES[CURRENT_SUB_MODE_IDX]
                force_opaque = FORCE_OPAQUE
                
            # Apply Level Color Mode
            if hasattr(gaussians, 'debug_level_color'):
                gaussians.debug_level_color = show_level_color
            
            # Prepare Overrides
            override_mask = None
            override_color = None
            
            if CONFLICT_MASK_ALL is not None:
                if show_conflict:
                    if sub_mode_name == "ALL":
                        override_mask = CONFLICT_MASK_ALL
                    elif sub_mode_name in CONFLICT_SUB_MASKS:
                        override_mask = CONFLICT_SUB_MASKS[sub_mode_name]
                elif show_red or show_norm:
                    override_mask = CONFLICT_MASK_ALL

                if show_conflict:
                    if CONFLICT_COLORS is not None:
                        override_color = CONFLICT_COLORS
                    else:
                        num_anchors = gaussians.get_anchor.shape[0]
                        override_color = torch.zeros((num_anchors, 3), device="cuda")
                        override_color[:, 0] = 1.0 
                elif show_red:
                    num_anchors = gaussians.get_anchor.shape[0]
                    override_color = torch.zeros((num_anchors, 3), device="cuda")
                    override_color[:, 0] = 1.0 # Red

            # Calc Pose
            new_center = center_orig + np.array([curr_x, curr_y, curr_z], dtype=np.float32)
            R_delta = get_rotation_matrix(curr_p, curr_yaw, curr_r)
            R_new = np.dot(R_delta, R_orig)
            T_new = -np.dot(R_new, new_center)
            
            view.R = R_new; view.T = T_new
            view_matrix = getWorld2View2(view.R, view.T, view.trans, view.scale)
            view.world_view_transform = torch.tensor(view_matrix).transpose(0, 1).cuda()
            view.world_view_transform_inv = view.world_view_transform.inverse()
            view.camera_center = view.world_view_transform_inv[3, :3]

            # Call Render (Now using Scheme 1)
            render_pkg = getattr(renderer_modules, 'render')(
                view, gaussians, pipe, background,
                override_mask=override_mask,
                override_color=override_color,
                force_opaque=force_opaque
            )

            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            img_np = rendering.detach().permute(1, 2, 0).cpu().numpy()
            
            try: img_queue.put_nowait(img_np)
            except Full: pass 

            elapsed = time.time() - loop_start_time
            if elapsed < target_frame_time: time.sleep(target_frame_time - elapsed)

        except Exception as e:
            print(f"Render Loop Error: {e}")
            break
    print("\n[Render Loop] Stopped.")

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

    # Load Scene
    safe_state(args.quiet)
    scene_data, gaussians_data, bg_data, r_mod = load_scene_once(lp, pp, args.iteration, args.ape, args.explicit)

    # ================= [解析冲突数据] =================
    conflict_path = "debug_anchor_conflict/conflict_indices.pt"
    if not os.path.exists(conflict_path): conflict_path = "../debug_anchor_conflict/conflict_indices.pt"
    
    if os.path.exists(conflict_path):
        try:
            print(f"[Info] Loading Conflict Data from {conflict_path}...")
            loaded_data = torch.load(conflict_path)
            num_anchors = gaussians_data.get_anchor.shape[0]
            
            CONFLICT_MASK_ALL = torch.zeros(num_anchors, dtype=torch.bool, device="cuda")
            CONFLICT_COLORS = torch.zeros((num_anchors, 3), device="cuda")
            
            if isinstance(loaded_data, dict):
                indices = loaded_data['indices'].to("cuda")
                colors = loaded_data['colors'].to("cuda") # [N_conflict, 3] float
                
                CONFLICT_MASK_ALL[indices] = True
                CONFLICT_COLORS[indices] = colors
                
                # --- 生成子 Mask ---
                print("[Info] Parsing conflict sub-categories...")
                c = colors
                # Green: G>0.5, R<0.5, B<0.5
                mask_green = (c[:, 1] > 0.5) & (c[:, 0] < 0.5) & (c[:, 2] < 0.5)
                # Cyan: G>0.5, B>0.5, R<0.5
                mask_cyan = (c[:, 1] > 0.5) & (c[:, 2] > 0.5) & (c[:, 0] < 0.5)
                # Red: R>0.5, G<0.5, B<0.5
                mask_red = (c[:, 0] > 0.5) & (c[:, 1] < 0.5) & (c[:, 2] < 0.5)
                # Yellow: R>0.5, G>0.5, B<0.5
                mask_yellow = (c[:, 0] > 0.5) & (c[:, 1] > 0.5) & (c[:, 2] < 0.5)
                # Purple: R>0.5, B>0.5, G<0.5
                mask_purple = (c[:, 0] > 0.5) & (c[:, 2] > 0.5) & (c[:, 1] < 0.5)
                
                def make_global_mask(local_mask, all_indices):
                    global_m = torch.zeros(num_anchors, dtype=torch.bool, device="cuda")
                    global_m[all_indices[local_mask]] = True
                    return global_m

                CONFLICT_SUB_MASKS = {
                    "GREEN (Artifacts)": make_global_mask(mask_green, indices),
                    "CYAN (Floating)": make_global_mask(mask_cyan, indices),
                    "RED (Opacity)": make_global_mask(mask_red, indices),
                    "YELLOW (Geom)": make_global_mask(mask_yellow, indices),
                    "PURPLE (Dual)": make_global_mask(mask_purple, indices)
                }
                
                SUB_MODES = ["ALL"] + list(CONFLICT_SUB_MASKS.keys())
                for name, m in CONFLICT_SUB_MASKS.items():
                    print(f"  > {name}: {m.sum().item()} points")
                
            else:
                print("[Warn] Old format detected. Only 'ALL' mode available.")
                indices = loaded_data.to("cuda")
                CONFLICT_MASK_ALL[indices] = True
                CONFLICT_COLORS = None 
                
        except Exception as e:
            print(f"[Error] Failed to load conflict indices: {e}")
            CONFLICT_MASK_ALL = None
    else:
        print(f"[Warn] Conflict indices not found.")
    # ======================================================

    # Threads
    kb_thread = threading.Thread(target=keyboard_listener_thread, daemon=True)
    kb_thread.start()

    HOST = '0.0.0.0'
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, args.port))
    server_socket.listen(1)

    print(f"\n[Server] Ready on port {args.port}. Waiting...")
    
    try:
        while IS_RUNNING:
            conn, addr = server_socket.accept()
            print(f"[Server] Connected: {addr}")
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            CLIENT_CONNECTED = True
            img_queue = Queue(maxsize=2)
            sender = threading.Thread(target=socket_sender_thread, args=(conn, img_queue), daemon=True)
            sender.start()
            
            render_loop(scene_data, gaussians_data, bg_data, r_mod, pp, img_queue, args.start_view)
            
            try: conn.close()
            except: pass
            print("[Server] Client session ended.")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        IS_RUNNING = False
        server_socket.close()
        restore_terminal_mode()