import cv2
import numpy as np
from ultralytics import YOLO
import winsound
import threading
import time
import os
from twilio.rest import Client
from flask import Flask, Response, render_template_string, request, jsonify, session, redirect, url_for

# --- كود تصحيح التوافق مع بايثون 3.14 (YOLOv8 FUSE BUG) ---
import ultralytics.nn.tasks
try:
    _orig_fuse = ultralytics.nn.tasks.BaseModel.fuse
    def safe_fuse(self, *args, **kwargs):
        try:
            return _orig_fuse(self, *args, **kwargs)
        except AttributeError:
            print("⚠️ Sentina Core: PyTorch layer fusion bypassed for compatibility.")
            return self
    ultralytics.nn.tasks.BaseModel.fuse = safe_fuse
except AttributeError:
    pass
# -----------------------------------------------------------------

app = Flask(__name__)
app.secret_key = 'SAIDIA_SECURE_KEY_2026'

# --- قاعدة بيانات تجريبية للمستخدمين ---
users_db = {
    "saidia@gmail.com": "123456",
    "prof": "saidia2026",
    "zakaria": "admin"
}

# تحميل الموديل الذكي
model = YOLO('yolov8n.pt') 

# قفل لمنع تداخل الخيوط (Threads) أثناء تشغيل الـ YOLO على نفس الموديل
model_lock = threading.Lock()

# --- إعدادات Twilio ---
account_sid = 'ACd48c9531132e9272bebe915021f57c4f'  
auth_token = '96c2d4e121f0172efd85f17d08c84c73'       
client = Client(account_sid, auth_token)
twilio_number = '+17433477427' 
my_phone = '+212706341918'     

# المتغيرات العالمية للنظام
call_sent = False
is_alarm_playing = False
alert_level = 0
people_detected = 0 
panic_scores = {}    
manual_emergency_mode = False

stationary_timers = {} 
prev_gray_crops = {}

# مخازن البث المباشر للكاميرات الأربعة
frame_cam1, frame_cam2, frame_cam3, frame_cam4 = None, None, None, None

def make_emergency_call():
    global call_sent
    if not call_sent:
        try:
            call_sent = True
            print("📞 [TWILIO] جاري الاتصال بالطوارئ لايف...")
            client.calls.create(
                twiml='<Response><Say language="fr-FR">Attention! Le systeme Sentina Core a detecte une noyade critique.</Say></Response>',
                to=my_phone, from_=twilio_number
            )
            print("✅ [TWILIO] تم الاتصال بنجاح!")
        except Exception as e:
            print(f"❌ [TWILIO] خطأ في الاتصال: {e}")
            call_sent = False

def trigger_siren():
    global is_alarm_playing
    try:
        winsound.PlaySound("alert.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
        time.sleep(4)
    except:
        winsound.Beep(2000, 1000)
    is_alarm_playing = False

# --- دالة معالجة الكاميرات الفردية (مصلحة بالكامل لحل الـ Lag والـ Detection) ---
def process_camera(cam_id, video_path):
    global frame_cam1, frame_cam2, frame_cam3, frame_cam4, people_detected, alert_level, manual_emergency_mode, is_alarm_playing, call_sent
    
    print(f"🎬 [CAMERA_{cam_id}] جاري إعداد مصدر الفيديو...")
    
    if not os.path.exists(video_path):
        print(f"ℹ️ [CAMERA_{cam_id}] الملف {video_path} غير موجود، سيتم تشغيل التغذية عبر test.mp4 تلقائياً.")
        video_path = "test.mp4"
        
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened(): 
        print(f"⚠️ [CAMERA_{cam_id}] تعذر قراءة الملف، يتم الانتقال إلى الـ Webcam...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"❌ [CAMERA_{cam_id}] خطأ حرج: لا يوجد أي مصدر فيديو متاح!")
        return

    print(f"✅ [CAMERA_{cam_id}] تم تشغيل بث الفيديو بنجاح.")

    frame_count = 0  # عداد ذكي للتحكم في الـ Lag

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.02)
                continue
            
            frame_count += 1
            # ⚡ حيلة تسريع الأداء: معالجة إطار وتخطي إطار لتقليل جهد الـ CPU بنسبة 50% وإنهاء الـ Lag تماماً
            if frame_count % 2 != 0:
                continue

            frame = cv2.resize(frame, (480, 360))
            h, w, _ = frame.shape
            
            with model_lock:
                # 🎯 رفع الدقة لـ 320 وخفض الـ conf لـ 0.20 لضمان رصد الغريق بدقة وظهور الـ Boxes
                results = model.track(frame, persist=True, imgsz=320, conf=0.20, classes=[0], verbose=False)
                
            current_frame_ids = []
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()
                
                for box, raw_id in zip(boxes, ids):
                    obj_id = int(raw_id) + (cam_id * 1000) 
                    x1, y1, x2, y2 = map(int, box)
                    if (x2 - x1) < 10 or (y2 - y1) < 10: continue 
                    
                    current_frame_ids.append(obj_id)
                    person_crop = frame[y1:y2, x1:x2]
                    if person_crop.size == 0: continue
                    
                    gray_crop = cv2.resize(cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY), (30, 30))
                    motion_intensity = 0
                    if obj_id in prev_gray_crops:
                        flow = cv2.calcOpticalFlowFarneback(prev_gray_crops[obj_id], gray_crop, None, 0.5, 3, 10, 3, 5, 1.2, 0)
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion_intensity = np.mean(mag)
                        
                    prev_gray_crops[obj_id] = gray_crop
                    if obj_id not in panic_scores: panic_scores[obj_id] = 0
                    if obj_id not in stationary_timers: stationary_timers[obj_id] = None
                    
                    # 💡 المنطق الجديد والمطور: كشف الخطر إذا كان الشخص ساكناً جداً (إغماء) أو يتحرك بعنف عشوائي في مكانه (تخبط وصعود وهبوط)
                    if motion_intensity < 0.6 or motion_intensity > 2.5:
                        if stationary_timers[obj_id] is None: stationary_timers[obj_id] = time.time()
                        if time.time() - stationary_timers[obj_id] > 2.0:  # تسريع الاستجابة لـ 2 ثوانٍ بدل 4
                            panic_scores[obj_id] = min(100, panic_scores[obj_id] + 15)
                    else:
                        stationary_timers[obj_id] = None
                        panic_scores[obj_id] = max(0, panic_scores[obj_id] - 5)
                        
                    if manual_emergency_mode: panic_scores[obj_id] = 100
                    p_score = panic_scores.get(obj_id, 0)
                    color = (0, 0, 255) if p_score >= 75 else (0, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID:{obj_id % 1000} RISK:{int(p_score)}%", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if manual_emergency_mode:
                alert_level = 95
            else:
                all_scores = list(panic_scores.values())
                alert_level = max(all_scores) if all_scores else 0

            zone_names = {1: "ZONE A // NORTH", 2: "ZONE B // WEST", 3: "ZONE C // SOUTH", 4: "ZONE D // EAST"}
            cv2.putText(frame, zone_names.get(cam_id, f"ZONE {cam_id}"), (15, int(h-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if alert_level >= 75 or manual_emergency_mode:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8) 
                if cam_id == 1: 
                    threading.Thread(target=make_emergency_call, daemon=True).start()
                    if not is_alarm_playing:
                        is_alarm_playing = True
                        threading.Thread(target=trigger_siren, daemon=True).start()
            else:
                if not manual_emergency_mode: call_sent = False

            _, buf = cv2.imencode('.jpg', frame)
            globals()[f"frame_cam{cam_id}"] = buf.tobytes()
            time.sleep(0.01)  # تسريع التدفق البرمجي للـ Stream
            
        except Exception as err:
            print(f"❌ [ERROR CAMERA {cam_id}]: {err}")
            time.sleep(1)

# تشغيل القنوات الأربعة بالتوازي
threading.Thread(target=process_camera, args=(1, "video1.mp4"), daemon=True).start()
threading.Thread(target=process_camera, args=(2, "video2.mp4"), daemon=True).start()
threading.Thread(target=process_camera, args=(3, "video3.mp4"), daemon=True).start()
threading.Thread(target=process_camera, args=(4, "video4.mp4"), daemon=True).start()

def generate_stream(cam_id):
    while True:
        f = globals().get(f"frame_cam{cam_id}")
        if f: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + f + b'\r\n')
        time.sleep(0.04)

# =================================================================
# 1. UI TEMPLATES
# =================================================================

NAVBAR_HTML = '''
<nav class="bg-slate-900 border-b border-slate-800 w-full z-50 relative">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-20">
            <div class="flex items-center">
                <i class="fa-solid fa-microchip text-blue-500 text-3xl mr-3"></i>
                <a href="/" class="font-black text-2xl tracking-tight text-white">SENTINA <span class="text-blue-500">CORE</span></a>
            </div>
            <div class="flex items-center space-x-6">
                <a href="/" class="text-slate-300 hover:text-white font-medium transition">Accueil</a>
                <a href="/tests" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-xl font-bold transition shadow-lg shadow-blue-500/25">
                    <i class="fa-solid fa-vial mr-1.5"></i> Tests & Screenshots
                </a>
                <a href="/login" class="text-slate-300 hover:text-white font-medium transition">Connexion</a>
                <a href="/register" class="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-xl font-semibold border border-slate-700 transition">S'inscrire</a>
            </div>
        </div>
    </div>
</nav>
'''

LANDING_HTML = f'''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>SENTINA CORE - Accueil</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-950 text-slate-100 font-sans min-h-screen flex flex-col">
    {NAVBAR_HTML}
    <div class="flex-grow flex items-center justify-center px-4 sm:px-6 lg:px-8 py-20">
        <div class="max-w-4xl text-center">
            <div class="inline-flex items-center gap-2 bg-blue-950/50 border border-blue-500/30 text-blue-400 px-4 py-1.5 rounded-full text-xs font-bold tracking-wide mb-6 uppercase font-mono">
                <i class="fa-solid fa-shield-halved"></i> Projet de Fin d'Études // Zakaria Mhiouah
            </div>
            <h1 class="text-5xl tracking-tight font-black text-white sm:text-6xl md:text-7xl">
                La Sécurité Intelligente<br>
                <span class="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-500">Pour Vos Plages</span>
            </h1>
            <p class="mt-6 text-lg text-slate-400 max-w-2xl mx-auto">
                SENTINA CORE utilise l'Intelligence Artificielle de pointe pour surveiller, détecter et alerter en cas de risque de noyade à Saïdia.
            </p>
            <div class="mt-10 flex justify-center gap-4">
                <a href="/register" class="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl transition shadow-xl shadow-blue-500/25">
                    Commencer l'essai
                </a>
                <a href="/tests" class="px-8 py-4 bg-slate-900 hover:bg-slate-800 text-slate-200 font-bold rounded-xl border border-slate-800 transition">
                    <i class="fa-solid fa-images mr-2 text-blue-500"></i> Voir les Tests
                </a>
            </div>
        </div>
    </div>
</body>
</html>
'''

REGISTER_HTML = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Inscription - SENTINA CORE</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-950 flex items-center justify-center min-h-screen p-4 text-white">
    <div class="w-full max-w-md bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-2xl">
        <h1 class="text-2xl font-black text-center mb-6">Créer un compte</h1>
        <form method="POST" class="space-y-4">
            <div><label class="block text-sm font-medium text-slate-400 mb-1">Nom</label><input type="text" name="name" required class="w-full p-3 bg-slate-950 border border-slate-800 rounded-xl focus:border-blue-500 outline-none"></div>
            <div><label class="block text-sm font-medium text-slate-400 mb-1">Email</label><input type="email" name="email" required class="w-full p-3 bg-slate-950 border border-slate-800 rounded-xl focus:border-blue-500 outline-none"></div>
            <div><label class="block text-sm font-medium text-slate-400 mb-1">Mot de passe</label><input type="password" name="password" required class="w-full p-3 bg-slate-950 border border-slate-800 rounded-xl focus:border-blue-500 outline-none"></div>
            <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3.5 rounded-xl transition mt-2">S'inscrire</button>
        </form>
    </div>
</body>
</html>
'''

LOGIN_HTML = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Connexion - SENTINA CORE</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
</head>
<body class="bg-slate-950 flex items-center justify-center min-h-screen p-4 text-white">
    <div class="w-full max-w-md bg-slate-900 p-8 rounded-2xl border border-slate-800 shadow-2xl">
        <h1 class="text-2xl font-black text-center mb-6">Connexion</h1>
        <form method="POST" class="space-y-4">
            <div><label class="block text-sm font-medium text-slate-400 mb-1">Email</label><input type="text" name="email" required class="w-full p-3 bg-slate-950 border border-slate-800 rounded-xl focus:border-blue-500 outline-none"></div>
            <div><label class="block text-sm font-medium text-slate-400 mb-1">Mot de passe</label><input type="password" name="password" required class="w-full p-3 bg-slate-950 border border-slate-800 rounded-xl focus:border-blue-500 outline-none"></div>
            <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3.5 rounded-xl transition mt-2">Se connecter</button>
        </form>
    </div>
</body>
</html>
'''

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Dashboard - SENTINA CORE</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { transition: background 0.3s ease; }
        .danger-bg { background-color: #240202 !important; animation: pulseRed 1s infinite alternate; }
        @keyframes pulseRed { 
            0% { box-shadow: inset 0 0 50px rgba(255,0,0,0.3); } 
            100% { box-shadow: inset 0 0 120px rgba(255,0,0,0.7); } 
        }
    </style>
</head>
<body id="mainBody" class="bg-slate-950 text-white p-6 min-h-screen flex flex-col justify-between">
    
    <div class="flex justify-between items-center mb-6 bg-slate-900/80 p-4 rounded-xl border border-slate-800 shadow-2xl">
        <div class="flex items-center gap-3">
            <h1 class="text-2xl font-black tracking-wider">SENTINA CORE <span class="text-blue-500">CONTROL PANEL</span></h1>
            <span class="text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded font-mono">v2.0 Multi-Cam Grid</span>
        </div>
        
        <div class="flex items-center gap-3">
            <button onclick="triggerPanic()" class="bg-red-600 hover:bg-red-700 text-white font-bold px-4 py-2 rounded-xl text-xs uppercase tracking-wider flex items-center gap-2 cursor-pointer border border-red-400 animate-bounce shadow-lg shadow-red-600/30">
                <i class="fa-solid fa-triangle-exclamation animate-pulse"></i> 🚨 TRIGGER EMERGENCY
            </button>
            <button onclick="resetSystem()" class="bg-emerald-600 hover:bg-emerald-700 text-white font-bold px-4 py-2 rounded-xl text-xs uppercase tracking-wider flex items-center gap-2 cursor-pointer border border-emerald-400 shadow-lg shadow-emerald-600/30">
                <i class="fa-solid fa-rotate-left"></i> RESET SYSTEM
            </button>
            <a href="/logout" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-4 py-2 rounded-xl text-xs font-semibold border border-slate-700 transition">Déconnexion</a>
        </div>
    </div>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4 flex-grow mb-4">
        
        <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 flex flex-col relative overflow-hidden group">
            <div class="text-[10px] text-blue-400 font-bold font-mono mb-2 flex justify-between">
                <span>🛰️ CAMERA_01 // ZONE_NORTH</span> <span class="text-green-400">● LIVE</span>
            </div>
            <div class="flex-grow bg-black rounded-lg overflow-hidden aspect-video relative">
                <img src="/stream/1" class="w-full h-full object-cover">
            </div>
        </div>

        <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 flex flex-col relative overflow-hidden group">
            <div class="text-[10px] text-purple-400 font-bold font-mono mb-2 flex justify-between">
                <span>🛰️ CAMERA_02 // ZONE_WEST</span> <span class="text-green-400">● LIVE</span>
            </div>
            <div class="flex-grow bg-black rounded-lg overflow-hidden aspect-video relative">
                <img src="/stream/2" class="w-full h-full object-cover">
            </div>
        </div>

        <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 flex flex-col relative overflow-hidden group">
            <div class="text-[10px] text-yellow-400 font-bold font-mono mb-2 flex justify-between">
                <span>🛰️ CAMERA_03 // ZONE_SOUTH</span> <span class="text-green-400">● LIVE</span>
            </div>
            <div class="flex-grow bg-black rounded-lg overflow-hidden aspect-video relative">
                <img src="/stream/3" class="w-full h-full object-cover">
            </div>
        </div>

        <div class="bg-slate-900 p-3 rounded-xl border border-slate-800 flex flex-col relative overflow-hidden group">
            <div class="text-[10px] text-teal-400 font-bold font-mono mb-2 flex justify-between">
                <span>🛰️ CAMERA_04 // ZONE_EAST</span> <span class="text-green-400">● LIVE</span>
            </div>
            <div class="flex-grow bg-black rounded-lg overflow-hidden aspect-video relative">
                <img src="/stream/4" class="w-full h-full object-cover">
            </div>
        </div>

    </div>

    <footer class="text-center text-[10px] text-slate-600 uppercase font-mono tracking-widest mt-2">
        // SENTINA SECURITY CONTROL AREA // OPERATIONAL MATRIX // 2026
    </footer>

    <script>
        function triggerPanic() { fetch('/api/trigger', { method: 'POST' }); }
        function resetSystem() { fetch('/api/reset', { method: 'POST' }); }
        function checkStatus() {
            fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                const body = document.getElementById('mainBody');
                if(data.danger) { body.classList.add('danger-bg'); } else { body.classList.remove('danger-bg'); }
            });
        }
        setInterval(checkStatus, 800);
    </script>
</body>
</html>
'''

# =================================================================
# 2. ROUTES DU SITE WEB
# =================================================================

@app.route('/')
def home():
    return render_template_string(LANDING_HTML)

@app.route('/tests')
def tests():
    static_path = os.path.join(app.root_path, 'static')
    images = []
    if os.path.exists(static_path):
        files = os.listdir(static_path)
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    dynamic_cards = ""
    for i, img in enumerate(images, 1):
        dynamic_cards += f'''
        <div class="bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-xl group hover:border-blue-500/40 transition">
            <div class="h-64 bg-slate-950 overflow-hidden relative">
                <img src="/static/{img}" class="w-full h-full object-cover group-hover:scale-105 transition duration-300">
            </div>
            <div class="p-6">
                <div class="text-xs font-bold text-blue-400 font-mono">SCREENSHOT_{i:02d}</div>
                <h3 class="text-lg font-bold text-white mt-1">Capture de Test {i}</h3>
                <p class="text-slate-400 text-sm mt-2">Fichier: <span class="text-amber-400 font-mono text-xs">{img}</span>. Validation des performances.</p>
            </div>
        </div>
        '''

    CUSTOM_TESTS_HTML = f'''
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Tests - SENTINA CORE</title>
        <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    </head>
    <body class="bg-slate-950 text-slate-100 min-h-screen">
        {NAVBAR_HTML}
        <div class="max-w-7xl mx-auto px-4 py-12">
            <h1 class="text-4xl font-black text-center mb-8">Galerie des Tests ({len(images)} images)</h1>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {dynamic_cards}
            </div>
        </div>
    </body>
    </html>
    '''
    return render_template_string(CUSTOM_TESTS_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users_db[request.form.get('email')] = request.form.get('password')
        return redirect(url_for('login'))
    return render_template_string(REGISTER_HTML)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users_db and users_db[email] == password:
            session['authenticated'] = True
            session['user'] = email
            return redirect(url_for('dashboard'))
    return render_template_string(LOGIN_HTML)

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if 'authenticated' not in session: return redirect(url_for('login'))
    return render_template_string(DASHBOARD_HTML, user=session.get('user'))

@app.route('/stream/<int:cam_id>')
def stream_camera(cam_id):
    return Response(generate_stream(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/trigger', methods=['POST'])
def api_trigger():
    global manual_emergency_mode
    manual_emergency_mode = True
    print("🚨 [ALERT] تم تفعيل إنذار الطوارئ اليدوي من الـ Dashboard!")
    return jsonify({"status": "danger"})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global manual_emergency_mode, call_sent, panic_scores
    manual_emergency_mode = False
    call_sent = False
    panic_scores.clear()
    print("🟢 [ALERT] تم إلغاء الطوارئ وتصفير العدادات بنجاح.")
    return jsonify({"status": "safe"})

@app.route('/api/status')
def api_status():
    global manual_emergency_mode, alert_level
    is_danger = manual_emergency_mode or alert_level >= 75
    return jsonify({"danger": is_danger, "alert_level": int(alert_level)})

@app.route('/get_telemetry')
def get_telemetry():
    return jsonify({"alert_level": int(alert_level), "people_detected": people_detected, "call_sent": call_sent})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)