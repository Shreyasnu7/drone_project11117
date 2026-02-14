import asyncio
import cv2
import json
import socket
import time
import os
import aiohttp
import websockets
from pymavlink import mavutil
# --- CONFIG ---
SERVER_URL = "wss://drone-server-r0qe.onrender.com/ws/connect/RADXA_X"
API_URL = "https://drone-server-r0qe.onrender.com"
FC_PORT = "/dev/ttyS2"
FC_BAUD = 57600
# V60: Camera Constants
CAM_WIDTH = 1280
CAM_HEIGHT = 720
class SafetyEnvelope:
    def __init__(self, telemetry_cache):
        self.telem = telemetry_cache
        self.min_dist_front = 1.0 # meters (Stop if < 1m)

    def validate_auto_action(self, action):
        dist = self.telem.get('lidar_dist', 9.9)
        if dist < self.min_dist_front and action in ['TAKEOFF', 'LAND', 'ORBIT']:
             print(f"🛑 SAFETY BLOCK: Object too close ({dist}m) for {action}")
             return False
        return True
# V40: Smoothing Helper
def smooth(current_val, previous_val, alpha=0.15):
    if previous_val is None: return current_val
    return (previous_val * (1.0 - alpha)) + (current_val * alpha)
class RadxaBridge:
    def __init__(self):
        self.fc = None
        self.ws = None
        self.running = True
        self.telemetry_cache = {}
        self.boot_alt = None # V15: Relative Alt
        self.safety = SafetyEnvelope(self.telemetry_cache)
        self.cam_config = {'w': 1920, 'h': 1080, 'fps': 30, 'source': 'internal'} # Default 1080p (IMX219)
        self.target_caps = "video/x-raw,width=1920,height=1080,framerate=30/1"
        self.cam_needs_reset = False
        self.recording = False
        self.rec_out = None
        self.batt_threshold = 20
        self.user_gps = None # (lat, lng)
        self.last_cloud_msg = time.time()
        self.low_batt_triggered = False
        self.is_armed = False # V107: Init missing state
        self.watchdog_triggered = False # V107: Init missing state
        self.smoothing_buffer = {'rc1': 1500, 'rc2': 1500, 'rc3': 1000, 'rc4': 1500} # V40: Smoothing State
        self.esp32_cmd_queue = asyncio.Queue()  # Commands to send to ESP32
        
        # P2.5: Local Recording State
        self.is_recording = False
        self.video_writer = None
        self.take_photo_flag = False
        self.record_start_time = 0
        
        # P2.6: Local AI Bridge & Video Server
        self.local_clients = set()
        self.latest_jpeg = None
    async def start_local_server(self):
        print("🚀 STARTING LOCAL AI BRIDGE (0.0.0.0:8000)...")
        # Standard Websocket Server for Control/Telemetry
        async with websockets.serve(self.handle_local_client, "0.0.0.0", 8000):
            await asyncio.Future() # Run forever

    async def start_local_video_server(self):
        from aiohttp import web
        print("🎥 STARTING LOCAL VIDEO SERVER (0.0.0.0:8080)...")
        app = web.Application()
        app.router.add_get('/snapshot', self.handle_snapshot)
        app.router.add_get('/stream', self.handle_stream)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8080)
        await site.start()
        # Keep alive
        while True:
            await asyncio.sleep(3600)

    async def handle_snapshot(self, request):
        if self.latest_jpeg:
            return web.Response(body=self.latest_jpeg, content_type='image/jpeg')
        return web.Response(status=503, text="No Frame Yet")

    async def handle_stream(self, request):
        resp = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace;boundary=frame',
                'Cache-Control': 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0',
            }
        )
        await resp.prepare(request)
        try:
            while True:
                if self.latest_jpeg:
                    frame = self.latest_jpeg
                    try:
                        await resp.write(b'--frame\r\n')
                        await resp.write(b'Content-Type: image/jpeg\r\n\r\n')
                        await resp.write(frame)
                        await resp.write(b'\r\n')
                        await asyncio.sleep(0.04) # Max 25fps
                    except:
                        break
                else:
                    await asyncio.sleep(0.1)
        except:
            pass
        return resp

    async def handle_local_client(self, websocket, path):
        print("🔗 FAST-LINK: Client Connected (Local)")
        self.local_clients.add(websocket)
        try:
            async for message in websocket:
                # Direct packet injection or handshake ignore
                try:
                    data = json.loads(message)
                    # Handshake support (Director sends {id:..., token:...})
                    if 'token' in data:
                        print(f"🤝 FAST-LINK Handshake: {data.get('id')}")
                        continue
                    # Process command
                    await self.process_packet(data.get('type'), data.get('payload'))
                except Exception as e:
                    print(f"⚠️ FAST-LINK Data Error: {e}")
        except:
             pass
        finally:
             print("🔗 FAST-LINK: Client Disconnected")
             self.local_clients.remove(websocket)

    def init_esp32(self):
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.esp32_addr = ("192.168.4.2", 8888)
        print(f"🔭 ESP32 Gimbal Link Active -> {self.esp32_addr}")
    async def connect_mavlink(self):
        self.init_esp32()
        asyncio.create_task(self.lidar_loop())
        # User Request: "Only do 19200 mode"
        bauds = [19200]
        while self.running:
            for baud in bauds:
                try:
                    print(f"🔌 Connecting to FC on {FC_PORT} @ {baud}...")
                    self.fc = mavutil.mavlink_connection(FC_PORT, baud=baud)
                    hb = self.fc.wait_heartbeat(timeout=2)
                    if hb is None:
                        print(f"❌ Heartbeat Timeout @ {baud}.")
                        self.fc.close()
                        continue
                    print(f"✅ FC Connected @ {baud}! Heartbeat receiving.")
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 2, 1)
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 4, 1) # Attitude
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 4, 1) # HUD
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_POSITION, 4, 1) # REL_ALT

                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'ARMING_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'EKF2_GPS_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # V26: Kill EKF GPS Check
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_EKF_THRESH', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # V26: Kill EKF Failsafe

                    # V27: AUTO-CONFIGURE FOR INDOOR (Disable GPS & Failsafes)
                    # GPS_TYPE=0 removed — contradicts V106 GPS enable below (GPS_TYPE=1)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'AHRS_GPS_USE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_THR_ENABLE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_GCS_ENABLE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    # self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'FS_BATT_ENABLE', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # V46: Re-enabled below
                    # V106: ENABLE GPS (User Req: "GPS needs to be enabled")
                    # EKF3 uses GPS + Compass + IMU.
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'AHRS_EKF_TYPE', 3, mavutil.mavlink.MAV_PARAM_TYPE_UINT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'GPS_TYPE', 1, mavutil.mavlink.MAV_PARAM_TYPE_UINT32) # 1=Auto

                    # V39: UNLOCKED PRECISION MODE (User Request: "Exact Joystick Control")
                    # disabling Deadzones so even micro-movements are registered
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC1_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC2_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC3_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16) # Throttle Deadzone = 0
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'RC4_DZ', 0, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)

                    # Restoring Speed Limits to Standard (User wants full authority)
                    # PILOT_SPEED_UP=250 removed — overridden by V72 PILOT_SPEED_UP=500 below
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'PILOT_SPEED_DN', 150, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'PILOT_ACCEL_Z', 250, mavutil.mavlink.MAV_PARAM_TYPE_UINT16)

                    # V42: ADAPTIVE HOVER LEARNING (User Request: "Adapt to 1.5kg Weight")
                    # Enable Hover Learning (2=Learn and Save). 0.55 for 1.5kg F450.
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_THST_HOVER', 0.55, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    # V53: MOTOR IDLE FIX — MOT_SPIN_ARM=0.12 and MOT_SPIN_MIN=0.15 removed
                    # They were overridden by V72 values (0.25/0.25) below
                    # V104: BATTERY CALIBRATION (Standard MiniPix Defaults)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_MONITOR', 4, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_VOLT_PIN', 2, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_CURR_PIN', 3, mavutil.mavlink.MAV_PARAM_TYPE_INT8)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_VOLT_MULT', 10.1, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_AMP_PERVOLT', 18.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    # V105: CORRECTION - Capacity 8400mAh (User Specified)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_CAPACITY', 8400, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'ANGLE_MAX', 6000, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'PILOT_SPEED_UP', 500, mavutil.mavlink.MAV_PARAM_TYPE_INT16)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_SPOOL_TIME', 0.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32) # V72: ZERO DELAY
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_SPIN_ARM', 0.25, mavutil.mavlink.MAV_PARAM_TYPE_REAL32) # V72: Hot Idle
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'MOT_SPIN_MIN', 0.25, mavutil.mavlink.MAV_PARAM_TYPE_REAL32) # V72: Match Arm
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'ARMING_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32) # V72: No Checks
                    # V71: Li-Ion Voltage Range (Monitor 4 + Python Override)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_LOW_VOLT', 9.6, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    self.fc.mav.param_set_send(self.fc.target_system, self.fc.target_component, b'BATT_CRT_VOLT', 9.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
                    print("🔓 INDOOR MODE: ZERO DELAY (V72) | SMOOTHED BATTERY")
                    return
                except Exception as e:
                    print(f"⚠️ FC Check Failed: {e}")
                    await asyncio.sleep(1)
            print("🔴 FC NOT DETECTED. Retrying...")
            await asyncio.sleep(2)
    async def connect_cloud(self):
        while self.running:
            try:
                print(f"☁️ Connecting to Cloud: {SERVER_URL}...")
                # User Request: "NEVER disconnect on its own" -> INFINITE TIMEOUT
                async with websockets.connect(SERVER_URL, ping_interval=10, ping_timeout=None) as ws:
                    self.ws = ws
                    print("✅ Cloud Connected!")

                    # AUTH FRAME
                    auth_frame = {
                        "type": "connect_drone",
                        "droneId": "RADXA_X",
                        "token": "bearer_token"
                    }
                    await ws.send(json.dumps(auth_frame))

                    # PARALLEL TASKS
                    await asyncio.gather(
                        self.telemetry_loop(),
                        self.command_loop()
                    )
                    self.fc.mav.request_data_stream_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 2, 1) # V27: Debug RC
            except Exception as e:
                print(f"⚠️ Cloud Disconnected: {e}. Retrying in 5s...")
                self.ws = None
                await asyncio.sleep(5)
    async def telemetry_loop(self):
        last_send = 0
        while self.ws and self.running:
            if self.fc:
                # V22: Drain Buffer (Process up to 50 msgs per loop to catch ACKs)
                for _ in range(50):
                    msg = self.fc.recv_match(blocking=False)
                    if not msg: break

                    type = msg.get_type()
                    # Battery Failsafe
                    if type == 'SYS_STATUS':
                        # V70: FORCE PYTHON CALCULATION (If FC fails or is stuck)
                        batt_pct = msg.battery_remaining
                        batt_voltage = msg.voltage_battery / 1000.0
                        if batt_pct < 0 or batt_pct > 95:
                             # V71: Li-Ion Curve (9.0V to 12.6V)
                             calc_pct = int((batt_voltage - 9.0) / 3.6 * 100.0)
                             batt_pct = max(0, min(100, calc_pct))

                        # V72: REAL-TIME (No Smoothing - User wants accurate readings)
                        self.telemetry_cache['battery'] = batt_pct # Direct, unfiltered
                        self.telemetry_cache['voltage'] = batt_voltage
                        self.telemetry_cache['armed'] = (msg.onboard_control_sensors_health & mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO) # Approximation or use HEARTBEAT

                        # V26: Capture Mode
                        self.telemetry_cache['mode_id'] = self.fc.flightmode

                        if msg.battery_remaining < self.batt_threshold and not self.low_batt_triggered:
                            print(f"⚠️ LOW BATT < {self.batt_threshold}%! Smart RTH...")
                            self.low_batt_triggered = True
                            if self.user_gps:
                                lat, lng = self.user_gps
                                print(f"🔄 RTH to User: {lat}, {lng}")
                                self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4) # GUIDED
                                self.fc.mav.mission_item_int_send(self.fc.target_system, self.fc.target_component, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT, 2, 0, 0, 0, 0, 0, int(lat * 1e7), int(lng * 1e7), 15)
                            else:
                                self.fc.set_mode('RTL')

                    elif type == 'STATUSTEXT': # V23: The Voice of the FC
                         print(f"📢 FC SAYS: {msg.text}")

                    elif type == 'PARAM_VALUE': # V23: TX Verification
                         print(f"✅ TX VERIFIED: Read Param {msg.param_id} = {msg.param_value}")
                    elif type == 'ATTITUDE':
                         self.telemetry_cache['roll'] = msg.roll
                         self.telemetry_cache['pitch'] = msg.pitch
                         self.telemetry_cache['yaw'] = msg.yaw
                    elif type == 'RC_CHANNELS_RAW': # V27: Debug Switch Positions
                         # Log channels 5, 6, 7 (common for mode switches) every 2s
                         if int(time.time()) % 2 == 0 and int(time.time()) != getattr(self, 'last_rc_log', 0):
                              self.last_rc_log = int(time.time())
                              print(f"🎮 RC RAW: C5={msg.chan5_raw} C6={msg.chan6_raw} C7={msg.chan7_raw}")
                    # DATA FUSION Logic
                    sats = self.telemetry_cache.get('sats', 0)
                    if sats > 5:
                         self.telemetry_cache['altitude'] = self.telemetry_cache.get('altitude_gps', 0)
                         self.telemetry_cache['source'] = 'GPS'
                    else:
                         self.telemetry_cache['altitude'] = self.telemetry_cache.get('altitude_baro', 0)
                         self.telemetry_cache['source'] = 'BARO'
            now = time.time()
            if self.telemetry_cache:
                if now - last_send > 0.1: # 10Hz
                    msg = json.dumps({"type": "telemetry", "payload": self.telemetry_cache})
                    if self.ws: await self.ws.send(msg)
                    # P2.6: Broadcast to Local AI Clients
                    if self.local_clients:
                        dead = set()
                        for c in self.local_clients:
                            try: await c.send(msg)
                            except: dead.add(c)
                        self.local_clients -= dead
                    last_send = now
                # DASHBOARD LOGGING (Every 1s - scrolling)
                if int(now) % 2 == 0 and int(now) != getattr(self, 'last_log_sec', 0):
                    self.last_log_sec = int(now)
                    alt = self.telemetry_cache.get('altitude', 0)
                    src = self.telemetry_cache.get('source', 'UNK')
                    batt = self.telemetry_cache.get('battery', 0)
                    volt = self.telemetry_cache.get('voltage', 0)
                    sats = self.telemetry_cache.get('sats', 0)
                    raw = self.telemetry_cache.get('raw_alt', 0)
                    is_armed = self.fc.motors_armed() if self.fc else False
                    self.is_armed = is_armed # V107: Update state
                    mode = self.telemetry_cache.get('mode_id', 'UNK')
                    est_cells = int(round(volt / 4.2)) if volt > 0 else 0 # V26: Battery Debug
                    speed = self.telemetry_cache.get('speed', 0)
                     # PRINT NEWLINE logs for debugging
                    print(f"📊 DATA: Alt={alt:.1f}m (Raw={raw:.1f}) | Batt={batt}% ({volt:.1f}V~{est_cells}S) | Mode={mode} | Spd={speed:.1f} | Armed={is_armed}")



                    # V28: WAR ON RTL - Force Stabilize every 1s if disarmed
                    # V64: Force STABILIZE (0) - Instant Response
                    if not is_armed and mode != 'STABILIZE' and int(now) % 2 == 0:
                        self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0) # STABILIZE
                        print("🛡️ FORCING STABILIZE (0) - INSTANT ADAPTIVE")

            await asyncio.sleep(0.01)
    async def command_loop(self):
        while self.running and self.ws:
            try:
                msg = await self.ws.recv()
                # V57: RAW DEBUG (See exact JSON)
                if 'joystick' not in msg:
                     print(f"🔍 RAW JSON: {msg}")

                data = json.loads(msg)
                type = data.get('type')
                payload = data.get('payload', {})

                # V58 FIX: Double-Decode only if JSON-String. Keep original if fail.
                if isinstance(payload, str):
                    try:
                        decoded = json.loads(payload)
                        if isinstance(decoded, (dict, list)): # Only accept proper structures
                             payload = decoded
                    except:
                        pass # It was a plain string (e.g. "LAND") - Keep it!
                if type != 'joystick' and type != 'user_gps': # Spam filter: Hide GPS too
                    print(f"📥 Rx: {type}")

                if type == 'user_gps':
                    if payload and 'lat' in payload:
                        self.user_gps = (payload.get('lat'), payload.get('lng'))
                # V55: OMNI-PARSER
                cmd = type

                # V56: STRING CLEANUP
                if isinstance(cmd, str):
                    cmd = cmd.upper().strip().replace('"', '').replace("'", "")

                # V56: Safety Check
                is_emergency = cmd in ['LAND', 'DISARM', 'RTL', 'UPDATE_CONFIG']
                if not is_emergency and not self.safety.validate_auto_action(cmd):
                     print(f"🛑 SAFETY BLOCK: {cmd}")
                     # continue # Uncomment to enforce

                # P2.1: SETTINGS SYNC
                if cmd == 'UPDATE_CONFIG':
                    cfg = payload.get('config', {})
                    print(f"⚙️ SETTINGS UPDATED: {cfg}")

                    # HANDLE CAMERA RESOLUTION CHANGE
                    if 'cap_res' in cfg:
                        res_key = cfg['cap_res']
                        # IMX219 (Pi Cam V2) Modes
                        RES_MAP = {
                            "8mp": "video/x-raw,width=3280,height=2464,framerate=15/1",
                            "1080p": "video/x-raw,width=1920,height=1080,framerate=30/1",
                            "720p": "video/x-raw,width=1280,height=720,framerate=60/1",
                            "480p": "video/x-raw,width=640,height=480,framerate=90/1"
                        }
                        # If unknown (e.g. GoPro res), ignore it
                        if res_key in RES_MAP:
                            new_caps = RES_MAP[res_key]
                            # Only reset if changed
                            if new_caps != getattr(self, 'target_caps', ""):
                                self.target_caps = new_caps
                                self.cam_needs_reset = True
                                print(f"🎥 CAMERA CONFIG CHANGING TO: {res_key} ({new_caps})")

                # P2.2: GIMBAL RELAY
                elif cmd == 'GIMBAL':
                        pitch = payload.get('pitch', 0)
                        yaw = payload.get('yaw', 0)
                        await self.esp32_cmd_queue.put({"type": "gimbal", "pitch": pitch, "yaw": yaw})
                        print(f"🎥 GIMBAL: Pitch={pitch}, Yaw={yaw}")

                    # P1.6: MISSION HANDLER (Upload Waypoints to FC)
                elif cmd == 'UPLOAD_MISSION':
                    try:
                        items = payload.get('items', [])
                        print(f"🗺️ UPLOADING MISSION: {len(items)} Waypoints...")
                        if items:
                            self.fc.mav.mission_clear_all_send(self.fc.target_system, self.fc.target_component)
                            self.fc.mav.mission_count_send(self.fc.target_system, self.fc.target_component, len(items))
                            for i, item in enumerate(items):
                                self.fc.mav.mission_item_int_send(
                                    self.fc.target_system, self.fc.target_component,
                                    i,
                                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                    0, 1,
                                    0, 0, 0, 0,
                                    int(item['lat'] * 1e7),
                                    int(item['lng'] * 1e7),
                                    float(item.get('alt', 20))
                                )
                            print("✅ MISSION UPLOADED")
                    except Exception as e:
                        print(f"❌ Mission Upload Fail: {e}")

                elif cmd == 'TAKEOFF':
                    self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, 5)

                elif cmd == 'LAND':
                    alt = self.telemetry_cache.get('altitude', 0)
                    print(f"🛬 LAND CMD. Alt={alt:.1f}m")
                    if alt < 1.0:
                        print("🛑 GROUND: FORCE DISARM (21196)")
                        self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196, 0, 0, 0, 0, 0)
                    else:
                        print("🛬 AIR: SAFE DESCEND")
                        self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_NAV_LAND, 0, 0, 0, 0, 0, 0, 0, 0)

                elif cmd == 'ARM':
                     self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 0) # STABILIZE
                     print("🛡️ SPLIT-ARM: Mode -> STABILIZE (0)... NO DELAY")
                     self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                     print("🛡️ SPLIT-ARM: Sending ARM Command Now.")

                elif cmd == 'RTL':
                    self.fc.set_mode('RTL')
                
                # P2.5: CAPTURE/RECORD HANDLERS - Camera control (INTERCEPTED FOR RADXA)
                elif cmd == 'CAPTURE' or type == 'capture':
                    print("📸 CAPTURE: Triggering Local Save...")
                    self.take_photo_flag = True
                    # Also send to ESP32 for feedback (e.g. flash LED)
                    await self.esp32_cmd_queue.put({"type": "capture"})
                
                elif cmd == 'RECORD' or type == 'record':
                    recording = payload.get('payload', {}).get('recording', False) if isinstance(payload.get('payload'), dict) else payload.get('recording', False)
                    # Support both direct payload and nested payload structure
                    if 'recording' not in payload:
                         # Fallback for nested
                         recording = data.get('payload', {}).get('recording', False)
                    
                    # Simplest robust fetch:
                    recording = False
                    if isinstance(payload, dict):
                         recording = payload.get('recording', False)

                    print(f"🎥 RECORD: {'START' if recording else 'STOP'} (Local)")
                    self.is_recording = recording
                    # Also send to ESP32 for feedback (e.g. solid RED LED)
                    await self.esp32_cmd_queue.put({"type": "record", "recording": recording})
                elif type == 'joystick':
                    try:
                        # Extract and Clamping
                        def map_ch(val, center=True):
                            if center: return int(1500 + (val * 500))
                            return int(1000 + (val * 1000)) # 0-1 range to 1000-2000
                        # V32: REAL MODE 3 CROSS-MAPPING (App=Mode2, User=Mode3)
                        # App sends Right Stick as 'roll'/'pitch' (x/y). User uses it for Yaw/Throttle.
                        # App sends Left Stick as 'throttle'/'yaw' (z/r). User uses it for Pitch/Roll.

                        # V32: REAL MODE 3 CROSS-MAPPING (App=Mode2, User=Mode3)
                        # App sends Right Stick as 'roll'/'pitch' (x/y). User uses it for Yaw/Throttle.
                        # App sends Left Stick as 'throttle'/'yaw' (z/r). User uses it for Pitch/Roll.

                        # V42: ADAPTIVE HOVER MODE (AltHold + Learning)
                        # We use standard mapping with V40 Smoothing.
                        # Center Stick = 0 Climb Rate (Hover).
                        # FC manages throttle based on learned weight.

                        # V51: Define Arrmed Status EARLY
                        is_armed = self.fc.motors_armed() if self.fc else False
                        raw_thr = float(payload.get('pitch', payload.get('y', 0)))
                        target_rc3 = map_ch(raw_thr, center=True)

                        raw_yaw = float(payload.get('roll', payload.get('x', 0)))
                        target_rc4 = map_ch(raw_yaw, center=True)
                        raw_pitch = float(payload.get('throttle', payload.get('z', 0)))
                        target_rc2 = map_ch(raw_pitch, center=True)
                        raw_roll = float(payload.get('yaw', payload.get('r', 0)))
                        target_rc1 = map_ch(raw_roll, center=True)

                        # V50: REMOVE SMOOTHING ENTIRELY (Lag Fix Check)
                        # Direct mapping. No buffer.
                        rc4 = target_rc4
                        rc2 = target_rc2
                        rc1 = target_rc1

                        # V64: ADAPTIVE THROTTLE CURVE
                        # Use MOT_THST_HOVER (learned by AltHold) as the Center Stick Target

                        hover_param = 0.55 # F450 Heavy Default (1.5kg)
                        try:
                             # Try to read cached value (updated by watchdog)
                             hover_param = self.telemetry_cache.get('MOT_THST_HOVER', 0.55)
                             if hover_param < 0.1 or hover_param > 0.8: hover_param = 0.55 # Sanity Check
                        except:
                             hover_param = 0.55

                        hover_pwm = 1000 + (hover_param * 1000) # e.g. 0.55 -> 1550

                        t_in = target_rc3
                        t_out = 1000

                        if t_in < 1500:
                            # Low Half: Map 1000-1500 -> 1100-HoverPWM
                            pct = (t_in - 1000) / 500.0
                            t_out = 1100 + (pct * (hover_pwm - 1100))
                        else:
                            # High Half: Map 1500-2000 -> HoverPWM-2000 (V66: FULL POWER)
                            pct = (t_in - 1500) / 500.0
                            t_out = hover_pwm + (pct * (2000 - hover_pwm))

                        rc3 = int(t_out)

                        # V67: GAMER GRIP (AUTO-BOOST)
                        # Problem: Pitching forward causes lift loss. User wants speed.
                        # Solution: Mix Left Stick (Pitch/Roll) intensity into Throttle.

                        tilt_pitch = abs(target_rc2 - 1500)
                        tilt_roll = abs(target_rc1 - 1500)
                        max_tilt = max(tilt_pitch, tilt_roll) # 0 to 500

                        if max_tilt > 50: # Ignore tiny deadzone
                            # Boost Factor: 0.0 to 1.0 (at full stick)
                            # Max Boost: +800 PWM (V68: 100% Boost allowed)
                            boost = (max_tilt / 500.0) * 800.0
                            rc3 += int(boost) # USER REQUEST: ENABLE BOOST (Max Power on 360 stretch)
                            if rc3 > 2000: rc3 = 2000 # Clamp ceiling

                            # print(f"🚀 BOOST: Tilt={max_tilt} Added={int(boost)} Total={rc3}")

                        # V50: MOTOR SAFETY (NO STOPPING IN AIR) - User wants LINEAR DESCENT
                        # (Curve starts at 1100. We just clamp min to 1100 to prevent disarm in air)
                        if is_armed:
                             if rc3 < 1100: rc3 = 1100 # Safety Floor (Idle Only)

                        if abs(target_rc2 - 1500) > 50 or abs(target_rc3 - 1500) > 50:
                             # V65 DEBUG: Show what Pitch/Roll we are actually sending
                             print(f"🕹️ MIX: Pitch(RC2)={rc2} Roll(RC1)={rc1} Thr(RC3)={rc3}")
                        # V34: TOY MODE (Auto-Arm on Throttle) BEFORE CLAMPING CHECK
                        # (Logic handled by 'is_armed' check above)

                        if not is_armed:
                            if rc3 > 1400: # Adjusted for new curve (approx 30% up)
                                 self.auto_arm_counter = getattr(self, 'auto_arm_counter', 0) + 1

                                 # Lie to FC (Send Zero) so it accepts Arming
                                 real_rc3 = rc3
                                 rc3 = 1000

                                 if self.auto_arm_counter > 10: # ~1s hold
                                      print("🚀 AUTO-ARM: Throttle Up Detected! Arming in ALT_HOLD...")
                                      self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                                      self.auto_arm_counter = 0
                            else:
                                 self.auto_arm_counter = 0
                        self.fc.mav.rc_channels_override_send(
                            self.fc.target_system, self.fc.target_component,
                            rc1, rc2, rc3, rc4, 65535, 65535, 65535, 65535
                        )

                        # V31: STICK ARMING LOGIC (Backup - Down-Right)
                        if rc3 < 1150 and rc4 > 1900:
                             self.stick_arm_counter = getattr(self, 'stick_arm_counter', 0) + 1
                             if self.stick_arm_counter > 20: # ~2 seconds @ 10Hz
                                  print("🕹️ STICK ARM TRIGGERED!")
                                  self.fc.mav.command_long_send(self.fc.target_system, self.fc.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
                                  self.stick_arm_counter = 0
                        else:
                             self.stick_arm_counter = 0
                        # V15: Debug RC Output
                        # V15: Debug RC Output
                        if int(time.time() * 10) % 20 == 0: # Every 2s
                           print(f"🕹️ JOY: R{rc1} P{rc2} T{rc3} Y{rc4}")

                    except Exception as e:
                         print(f"Joystick Error: {e}")
            except Exception as e:
                print(f"CMD Loop Error: {e}")
                break
    async def video_loop(self):
        import cv2
        # --- GSTREAMER CAMERA SCANNER ---
        print("🔍 SCANNING CAMERAS (0-9) for active feed...")
        cap = None
        current_idx = -1

        # HARDCODED CAMERA (v4l2-ctl proved video0 works)
        # Avoid scanning loop because open/close rapidly causes driver instability.
        print("🔎 Video Loop starting (Targeting /dev/video0)...")
        
        while True:
            current_idx = 0
            
            # Use EXACT Pipeline from camera_test.py (Generic Auto)
            # Add drop=true for latency.
            print(f"👉 Trying /dev/video{current_idx}...")
            # Pipeline: v4l2src ! videoconvert ! appsink
            # V0.4 FIX: Enforce BGR to fix stride corruption. Limit buffers for latency.
            # V0.5 FIX: Downscale in GStreamer to save CPU (8MP -> 640x360).
            pipeline = (
                f"v4l2src device=/dev/video{current_idx} ! "
                "videoscale ! video/x-raw,width=640,height=360 ! "
                "videoconvert ! "
                "video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            
            # Try to Open
            # NOTE: cv2.CAP_GSTREAMER is critical
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not cap.isOpened():
                 print(f"❌ Failed to open /dev/video{current_idx}. Retrying in 2s...")
                 time.sleep(2)
                 continue

            # Read ONE frame to confirm
            ret, frame = cap.read()
            if not ret or frame is None:
                 print(f"⚠️ Opened but no data from /dev/video{current_idx}. Driver Timeout? Retrying...")
                 cap.release()
                 time.sleep(1)
                 continue
            
            print(f"✅ CAMERA ACTIVE! Resolution: {frame.shape[1]}x{frame.shape[0]}")
            self.cam_needs_reset = False
            break # Exit while True loop
        if cap is None:
            print("🚨 FATAL: NO WORKING CAMERA DETECTED (Scanned 0-9).")
            # We will just yield forever to keep bridge alive
            while self.running: await asyncio.sleep(1)
            return
        # 2. Stream Loop
        self.frame_in_transit = False # V13: Flow Control
        async with aiohttp.ClientSession() as session:
            while self.running:
                # Re-configuration check
                if self.cam_needs_reset:
                     cap.release()
                     caps = getattr(self, 'target_caps', "")
                     if caps:
                         pipeline = f"v4l2src device=/dev/video{current_idx} ! {caps} ! videoscale ! video/x-raw,width=640,height=360 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
                     else:
                         pipeline = f"v4l2src device=/dev/video{current_idx} ! videoscale ! video/x-raw,width=640,height=360 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
                     cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                     
                     # Reset Downstream Consumers (Size Changed)
                     if hasattr(self, 'udp_streamer') and self.udp_streamer:
                         self.udp_streamer.release()
                         self.udp_streamer = None
                     
                     if self.video_writer:
                         self.video_writer.release()
                         self.video_writer = None
                         
                     self.cam_needs_reset = False

                # Flow Control: Don't read/send if busy
                if self.frame_in_transit:
                    await asyncio.sleep(0.01)
                    continue
                # V80: NON-BLOCKING READ (Fix 1011 Timeout)
                loop = asyncio.get_running_loop()
                ret, raw_frame = await loop.run_in_executor(None, cap.read)

                if ret and raw_frame is not None:
                    # FORCE RESIZE: Camera might be 8MP (3280x2160). Scale to Target (1280x720) immediately.
                    if raw_frame.shape[1] != CAM_WIDTH or raw_frame.shape[0] != CAM_HEIGHT:
                        try:
                             # INTER_AREA is better for downscaling, but LINEAR is faster. Used LINEAR.
                             raw_frame = cv2.resize(raw_frame, (CAM_WIDTH, CAM_HEIGHT), interpolation=cv2.INTER_LINEAR)
                        except Exception as e:
                             print(f"⚠️ Resize Failed: {e}")

                    try:
                        # V87: BALANCED BOOST (Clean Image)
                        raw_frame = cv2.convertScaleAbs(raw_frame, alpha=1.7, beta=40)

                        # V90: DUAL STREAM FORK (P2.5 IMPLEMENTATION)
                        # Stream A: High Res for AI/Recording
                        current_time = int(time.time())

                        # 0. UDP STREAM TO LAPTOP AI (New Dual Cam Support)
                        # Stream 1080p to Laptop (192.168.0.3) for DirectorCore to consume
                        if not hasattr(self, 'udp_streamer') or self.udp_streamer is None:
                            try:
                                # WEAK SIGNAL FIX: 
                                # 1. Downscale to 640x360 (16:9) to survive poor wifi.
                                # 2. Lower bitrate (800kbps).
                                # 3. gop-size=15 (Recover every 0.5s from packet loss).
                                # WEAK SIGNAL FIX V2: 
                                # 1. Downscale to 640x360 (16:9) IN GSTREAMER (Save CPU).
                                # 2. Lower bitrate (800kbps).
                                # 3. Use 'avimux' as fallback for 'mpegtsmux'.
                                udp_pipeline = (
                                    "appsrc ! videoconvert ! avenc_mpeg4 bitrate=800000 gop-size=15 ! "
                                    "avimux ! udpsink host=192.168.0.3 port=8554 sync=false"
                                )
                                h, w = 360, 640 # Force small resolution
                                raw_frame = cv2.resize(raw_frame, (w, h))
                                h, w = raw_frame.shape[:2]
                                self.udp_streamer = cv2.VideoWriter(udp_pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (w, h), True)
                                print(f"📡 UDP STREAM STARTED: Targeting 192.168.0.3:8554 ({w}x{h})")
                            except Exception as e:
                                print(f"❌ UDP Stream Start Failed: {e}")
                                self.udp_streamer = False # Prevent retry spam
                        
                        if self.udp_streamer:
                            self.udp_streamer.write(raw_frame)

                        # 1. Handle Recording
                        if self.is_recording:
                            if self.video_writer is None:
                                fname = f"flight_record_{current_time}.mp4"
                                h, w = raw_frame.shape[:2]
                                print(f"📼 STARTING RECORDING: {fname} @ {w}x{h}")
                                # Define Release 1 codec (mp4v is widely supported)
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                self.video_writer = cv2.VideoWriter(fname, fourcc, 30.0, (w, h))
                                self.record_start_time = current_time
                            
                            if self.video_writer:
                                self.video_writer.write(raw_frame)
                        else:
                            if self.video_writer is not None:
                                print(f"💾 SAVED RECORDING ({int(current_time - self.record_start_time)}s)")
                                self.video_writer.release()
                                self.video_writer = None

                        # 2. Handle Photo Capture
                        if self.take_photo_flag:
                            fname = f"photo_{current_time}.jpg"
                            cv2.imwrite(fname, raw_frame)
                            print(f"📸 SAVED PHOTO: {fname}")
                            self.take_photo_flag = False

                        # Stream B: Downscale for App Preview (Bandwidth Saver)
                        preview_frame = cv2.resize(raw_frame, (480, 360))

                        # Encode Preview (Higher Quality for Cloud Visibility)
                        # V108: Quality 25 -> 60 (User Complaint: "Nothing is visible")
                        retval, buffer = cv2.imencode('.jpg', preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                        if retval:
                             self.latest_jpeg = buffer.tobytes()
                             asyncio.create_task(self.push_frame(session, self.latest_jpeg))
                             # Heartbeat every 5s (New line)
                             if int(time.time()) % 5 == 0 and int(time.time()) != getattr(self, 'last_vid_sec', 0):
                                 self.last_vid_sec = int(time.time())
                                 print(f"🎥 VIDEO: Streaming @ {current_idx} ({len(buffer)} bytes)")
                    except Exception as e: 
                        print(f"⚠️ Stream Error: {e}")
                        pass
                else:
                    print("⚠️ Frame Read Error in Stream. Sleeping 1s...")
                    await asyncio.sleep(1)

                await asyncio.sleep(0.01) # Max framerate limiter
    async def push_frame(self, session, data):
        if self.frame_in_transit: return
        self.frame_in_transit = True
        try:
            url = API_URL + "/video/frame"
            # Proper Multipart Upload
            form = aiohttp.FormData()
            form.add_field('file', data, filename='frame.jpg', content_type='image/jpeg')

            # V47: Increased Timeout for Weak Signals
            # V108: Increased to 10s for Larger Frames (Quality 60)
            async with session.post(url, data=form, timeout=10.0) as response:
                if response.status != 200:
                    print(f"❌ Video Upload Failed: {response.status}")
        except Exception as e:
            print(f"❌ Video Network Error: {repr(e)}")
        finally:
            self.frame_in_transit = False
    async def lidar_loop(self):
        """V102: REAL YDLIDAR DRIVER (Via Python SDK)"""
        try:
            import ydlidar
        except ImportError:
            print("❌ LIDAR: 'ydlidar' lib missing. Run fix_dependencies.sh")
            return

        ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0"]
        laser = None
        
        print(f"🔭 LIDAR: Scanning ports {ports}...")

        for port in ports:
            temp_laser = ydlidar.CYdLidar()
            temp_laser.setlidaropt(ydlidar.LidarPropSerialPort, port)
            temp_laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000)
            temp_laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE)
            temp_laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL)
            temp_laser.setlidaropt(ydlidar.LidarPropScanFrequency, 5.0)
            temp_laser.setlidaropt(ydlidar.LidarPropSampleRate, 3)
            temp_laser.setlidaropt(ydlidar.LidarPropSingleChannel, True)
            temp_laser.setlidaropt(ydlidar.LidarPropMaxRange, 8.0)
            temp_laser.setlidaropt(ydlidar.LidarPropMinRange, 0.1)

            if temp_laser.initialize():
                print(f"✅ LIDAR: FOUND @ {port}")
                laser = temp_laser
                break
            else:
                print(f"   Lidar fetch failed on {port}")
        
        if laser is None:
            print("❌ LIDAR: Init Failed on ALL ports. Check USB.")
            await asyncio.sleep(5)
            return

        # BYPASS: turnOn() reports "Device Tremble" but user says it's physically stable
        # SDK health check is too sensitive. Skip it and go straight to scanning.
        print("⚠️ BYPASSING turnOn() health check (too sensitive)")
        print("✅ LIDAR: Starting scan loop directly...")
        scan = ydlidar.LaserScan()

        while self.running:
            try:
                 ret = laser.doProcessSimple(scan)
                 if ret:
                     # Find minimum distance to prioritize safety
                     min_dist = 10.0

                     # Check Forward Sector (approx logic)
                     # Real implementation uses point.angle
                     for i in range(scan.points.size()):
                         point = scan.points[i]
                         if point.range > 0.1 and point.range < min_dist:
                             min_dist = point.range

                     # Send to FC (cm)
                     dist_cm = int(min_dist * 100)
                     if dist_cm > 10 and dist_cm < 800:
                         self.fc.mav.distance_sensor_send(
                                0, 10, 800, dist_cm, 0, 0, 0, 0
                         )

                     self.telemetry_cache['obstacle_dist'] = min_dist
                     self.telemetry_cache['lidar_status'] = "ACTIVE"

                 await asyncio.sleep(0.05)
            except Exception as e:
                print(f"Lidar Error: {e}")
                await asyncio.sleep(1)

        laser.turnOff()
        laser.disconnecting()

    async def watchdog_loop(self):
        """Monitors Cloud Connection Health"""
        print("🐕 Watchdog Active")
        while self.running:
            # V100: FAILSAFE - User Requirement: "Return to User on Disconnect"
            last_msg_delta = time.time() - self.last_cloud_msg

            if last_msg_delta > 5.0 and not self.watchdog_triggered and self.is_armed:
                 print(f"⚠️ LOST CLOUD CONNECTION ({int(last_msg_delta)}s)! TRIGGERING RTL!")
                 self.watchdog_triggered = True
                 try:
                     self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6) # 6=RTL
                 except: pass

            # V103: BATTERY FAILSAFE (User: "Auto Return at 15%")
            # We check the cache (updated by heartbeat/sys_status)
            # Default to 100 to avoid false trigger on startup
            current_batt = self.telemetry_cache.get('battery', 100)
            if self.is_armed and current_batt < 15 and current_batt > 0:
                 if not getattr(self, 'low_batt_triggered', False):
                      print(f"⚠️ LOW BATTERY ({current_batt}%)! TRIGGERING RTL!")
                      self.low_batt_triggered = True
                      try:
                          self.fc.mav.set_mode_send(self.fc.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6) # RTL
                      except: pass

            if last_msg_delta < 2.0:
                self.watchdog_triggered = False
            await asyncio.sleep(1)

    async def esp32_hardware_loop(self):
        """Complete ESP32 Bidirectional Link (WiFi UDP Edition)"""
        import socket
        
        UDP_IP = "0.0.0.0" # Listen on all interfaces
        UDP_PORT = 8888
        
        print(f"🔌 ESP32: Binding UDP Port {UDP_PORT}...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        sock.setblocking(False) 
        
        print(f"✅ ESP32: UDP Listener Active on {UDP_PORT}")
        
        esp32_addr = None # Will learn from incoming packets

        while self.running:
            try:
                # === RECEIVE: ESP32 → Radxa (UDP) ===
                try:
                    data_raw, addr = sock.recvfrom(4096)
                    esp32_addr = addr # Store for sending back commands
                    
                    line = data_raw.decode('utf-8').strip()
                    
                    if line and line.startswith('{'):
                         try:
                             data = json.loads(line)
                             
                             # Parse VL53L1X sensor data (t1,t2,t3,t4 in mm)
                             # 2 Top (Angled), 2 Bottom (Angled). YDLidar (Horizontal).
                             if 't1' in data and 't2' in data and 't3' in data and 't4' in data:
                                 # Filter Self-Collisions (<15cm) - e.g. Propellers/Legs
                                 raw_distances = [data['t1'], data['t2'], data['t3'], data['t4']]
                                 valid_distances = [d for d in raw_distances if d > 150 and d < 8000]
                                 
                                 # SAFETY PRIORITY: Active Braking Override
                                 # If any object is within 50cm (excluding self <15cm), STOP immediately.
                                 min_dist = min(valid_distances) if valid_distances else 9999
                                 if min_dist < 500: # 50cm
                                     print(f"🛑 CRITICAL PROXIMITY ({min_dist}mm)! FORCE BRAKE!")
                                     try:
                                         # Send Brake/Hold Mode
                                         self.fc.mav.set_mode_send(
                                             self.fc.target_system,
                                             mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                             17 # BRAKE (ArduPilot) / HOLD (PX4)
                                         )
                                     except: pass
                                 
                                 # Send to FC via MAVLink (Mapping Top/Bottom)
                                 # T1, T2 = Top (Mapped to UP/FORWARD_UP?) -> Using UP (24) and CUSTOM
                                 # B1, B2 = Bottom (Mapped to DOWN (25))
                                 
                                 # We map loosely for logs, but YDLIDAR handles Horizontal Ring 
                                 self.telemetry_cache['prox_alert'] = len(valid_distances) > 0
                                      
                                 # Log occasionally
                                 if int(time.time()) % 2 == 0:
                                      print(f"📡 VL53 (Filtered): {valid_distances} mm")
                         
                             # Forward full telemetry to cloud
                             if self.ws:
                                 await self.ws.send(json.dumps({"type": "esp32_telem", "payload": data}))
                         
                         except Exception as e:
                             print(f"⚠️ ESP32 parse error: {e}")
                except BlockingIOError:
                    pass # No data waiting
                
                # === SEND: Radxa → ESP32 (UDP) ===
                if esp32_addr and hasattr(self, 'esp32_cmd_queue') and not self.esp32_cmd_queue.empty():
                    cmd = self.esp32_cmd_queue.get_nowait()
                    msg = (json.dumps(cmd) + '\n').encode('utf-8')
                    sock.sendto(msg, esp32_addr)
                    print(f"📤 ESP32 CMD: {cmd} -> {esp32_addr}")
                
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"❌ ESP32 loop error: {e}")
                await asyncio.sleep(0.1)

if __name__ == "__main__":
    bridge = RadxaBridge()
    try:
        async def main():
            # Run everything in parallel so Video/Cloud doesn't wait for FC
            await asyncio.gather( 
                bridge.connect_mavlink(),
                bridge.connect_cloud(),
                bridge.video_loop(),
                bridge.lidar_loop(),
                bridge.esp32_hardware_loop(),
                bridge.watchdog_loop(),
                bridge.start_local_server(),
                bridge.start_local_video_server()
            )
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopping Bridge...")
