import io
import os
import signal
import threading
import time
import cv2
import numpy as np
import psutil
from flask import Flask, Response, jsonify, render_template_string, request
from picamera2 import Picamera2
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

picam2 = None
latest_frame = None
frame_lock = threading.Lock()
is_detecting = False

confidence_threshold = 0.20

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "best_int8.tflite")
model_ncnn = None

CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y'
]

latest_detections = []
detections_lock = threading.Lock()


def initialize_camera():
    global picam2
    print("Inicializando Picamera2...")
    picam2 = Picamera2()
    # "RGB888" no picamera2 entrega 3 canais já na ordem BGR
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}, controls={"FrameDurationLimits": (100000, 100000)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("Câmera pronta com cores normais!")


def capture_loop():
    global latest_frame
    while True:
        try:
            arr = picam2.capture_array("main")
            with frame_lock:
                latest_frame = arr
        except Exception as e:
            print(f"[Erro de captura]: {e}")
        time.sleep(0.1)


def detection_worker():
    global latest_frame, is_detecting, latest_detections, confidence_threshold

    print(f"Carregando modelo TFLite INT8 de: {model_path}...")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']  # Ex: [1, 3, 320, 320] ou [1, 320, 320, 3]

    # Detecta automaticamente se o formato é NCHW (canais no índice 1) ou NHWC (canais no índice 3)
    if input_shape[1] == 3 or input_shape[1] == 1:
        is_nchw = True
        h_in, w_in = input_shape[2], input_shape[3]
    else:
        is_nchw = False
        h_in, w_in = input_shape[1], input_shape[2]

    input_dtype = input_details[0]['dtype']

    print(f"Modelo TFLite pronto! Formato: {'NCHW' if is_nchw else 'NHWC'}, Resolução: {w_in}x{h_in}")

    # Log de latência por quadro (usado pelo analisar_log.py)
    log_f = open(os.path.join(BASE_DIR, "infer_log.csv"), "w", buffering=1)
    log_f.write("t,infer_ms,total_ms,n_deteccoes\n")

    while True:
        if is_detecting:
            current_arr = None
            with frame_lock:
                if latest_frame is not None:
                    current_arr = latest_frame.copy()

            if current_arr is not None:
                try:
                    t_start = time.perf_counter()
                    img_rgb = cv2.cvtColor(current_arr, cv2.COLOR_BGR2RGB)
                    img_resized = cv2.resize(img_rgb, (w_in, h_in))

                    if input_dtype == np.float32:
                        norm_img = img_resized.astype(np.float32) / 255.0
                    else:
                        scale, zero_point = input_details[0]['quantization']
                        if scale > 0:
                            norm_img = (img_resized.astype(np.float32) / 255.0 / scale + zero_point).astype(input_dtype)
                        else:
                            norm_img = img_resized.astype(input_dtype)

                    if is_nchw:
                        norm_img = np.transpose(norm_img, (2, 0, 1))

                    input_data = np.expand_dims(norm_img, axis=0)

                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    t_inf = time.perf_counter()
                    interpreter.invoke()
                    infer_ms = (time.perf_counter() - t_inf) * 1000

                    output_data = interpreter.get_tensor(output_details[0]['index'])

                    if output_details[0]['dtype'] in (np.int8, np.uint8):
                        scale, zero_point = output_details[0]['quantization']
                        if scale > 0:
                            output_data = (output_data.astype(np.float32) - zero_point) * scale

                    output = output_data[0]
                    if output.shape[0] < output.shape[1]:
                        output = output.T  # Transpõe para [num_anchors, 4 + num_classes]

                    boxes_cxcywh = output[:, :4]
                    scores_matrix = output[:, 4:]

                    class_ids = np.argmax(scores_matrix, axis=1)
                    confidences = np.max(scores_matrix, axis=1)

                    mask = confidences >= confidence_threshold
                    filtered_boxes = boxes_cxcywh[mask]
                    filtered_scores = confidences[mask]
                    filtered_class_ids = class_ids[mask]

                    boxes_for_nms = []
                    if len(filtered_boxes) > 0:
                        is_normalized = np.max(filtered_boxes) <= 1.5
                        scale_x = 640.0 if is_normalized else (640.0 / w_in)
                        scale_y = 480.0 if is_normalized else (480.0 / h_in)

                        for box in filtered_boxes:
                            cx, cy, w_box, h_box = box
                            x1 = int((cx - w_box / 2.0) * scale_x)
                            y1 = int((cy - h_box / 2.0) * scale_y)
                            wb = int(w_box * scale_x)
                            hb = int(h_box * scale_y)
                            boxes_for_nms.append([x1, y1, wb, hb])

                    indices = cv2.dnn.NMSBoxes(
                        bboxes=boxes_for_nms,
                        scores=filtered_scores.tolist(),
                        score_threshold=confidence_threshold,
                        nms_threshold=0.45
                    )

                    new_detections = []
                    if len(indices) > 0:
                        indices = indices.flatten() if hasattr(indices, 'flatten') else indices
                        for idx in indices:
                            x1, y1, wb, hb = boxes_for_nms[idx]
                            cls_id = int(filtered_class_ids[idx])
                            score = float(filtered_scores[idx])

                            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Letra_{cls_id}"

                            left = max(0, x1)
                            top = max(0, y1)
                            right = min(640, x1 + wb)
                            bottom = min(480, y1 + hb)

                            new_detections.append({
                                "class": class_name,
                                "score": score,
                                "box": [left, top, right, bottom]
                            })

                    with detections_lock:
                        latest_detections = new_detections

                    total_ms = (time.perf_counter() - t_start) * 1000
                    log_f.write(f"{time.time():.3f},{infer_ms:.1f},{total_ms:.1f},{len(new_detections)}\n")

                except Exception as e:
                    print(f"[Erro TFLite]: {e}")

        time.sleep(0.2)


def draw_and_encode():
    with frame_lock:
        if latest_frame is None:
            return None
        img_bgr = latest_frame.copy()

    # Espelha a imagem apenas para exibição (o modelo continua usando o frame original)
    img_bgr = cv2.flip(img_bgr, 1)
    W = img_bgr.shape[1]

    with detections_lock:
        for det in latest_detections:
            left, top, right, bottom = det["box"]

            # Espelha também as coordenadas da caixa
            left, right = W - right, W - left

            cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 0, 255), 3)

            label = f"{det['class']}: {det['score']:.1%}"
            cv2.rectangle(img_bgr, (left, max(0, top - 25)), (left + 120, top), (0, 0, 0), -1)
            cv2.putText(
                img_bgr, label, (left + 5, max(0, top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )

    _, jpeg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return jpeg.tobytes()


def generate_frames():
    while True:
        frame_bytes = draw_and_encode()
        if frame_bytes is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.1)


@app.route("/")
def index():
    return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LIBRAS - Realtime YOLO</title>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script>
                function startDetection() {
                    $.post('/start');
                    $('#startBtn').prop('disabled', true);
                    $('#stopBtn').prop('disabled', false);
                }
                function stopDetection() {
                    $.post('/stop');
                    $('#startBtn').prop('disabled', false);
                    $('#stopBtn').prop('disabled', true);
                }
                function updateConfidence() {
                    var confidence = $('#confidence').val();
                    $.post('/update_confidence', {confidence: confidence});
                }
                function updateDetections() {
                    $.get('/get_detections', function(data) {
                        $('#detections').empty();
                        if (data.length === 0) {
                            $('#detections').html('<em>Nenhuma letra detectada</em>');
                        } else {
                            data.sort((a, b) => b.score - a.score).forEach(d => {
                                $('#detections').append(`<p style="font-size: 20px;"><strong>Letra ${d.class}</strong>: ${(d.score * 100).toFixed(1)}%</p>`);
                            });
                        }
                    });
                }
                function updateStats() {
                    $.get('/get_stats', function(s) {
                        $('#stats').html(
                            `RAM da Pi: <strong>${s.ram_percent}%</strong> ` +
                            `(${s.ram_used_mb} / ${s.ram_total_mb} MB) — ` +
                            `App: <strong>${s.app_percent}%</strong> (${s.app_mb} MB)`
                        );
                    });
                }
                function closeApp() {
                    if (confirm('Encerrar aplicação?')) {
                        $.ajax({
                            url: '/close',
                            type: 'POST',
                            headers: { 'X-Requested-With': 'XMLHttpRequest' },
                            success: function(data) { alert(data); window.close(); }
                        });
                    }
                }
                $(document).ready(function() {
                    setInterval(updateDetections, 300);
                    setInterval(updateStats, 2000);
                    updateStats();
                });
            </script>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center; background-color: #f4f4f9;">
            <h1>Reconhecimento LIBRAS em Tempo Real</h1>
            <p id="stats" style="font-size: 16px; color: #555;">Memória: carregando...</p>
            <img src="{{ url_for('video_feed') }}" width="640" height="480" style="border: 2px solid #333; border-radius: 8px;" />
            <br><br>
            <button id="startBtn" onclick="startDetection()" style="padding: 10px 20px; font-size: 16px;">Iniciar Detecção</button>
            <button id="stopBtn" onclick="stopDetection()" disabled style="padding: 10px 20px; font-size: 16px;">Parar Detecção</button>
            <button onclick="closeApp()" style="padding: 10px 20px; font-size: 16px; background-color: #ff4d4d; color: white;">Fechar</button>
            <br><br>
            <label for="confidence" style="font-size: 16px;">Confiança Mínima:</label>
            <input type="number" id="confidence" name="confidence" min="0.05" max="0.5" step="0.05" value="0.20" onchange="updateConfidence()" style="font-size: 16px; width: 60px;">
            <br><br>
            <h3>Detecções Atuais:</h3>
            <div id="detections">Clique em 'Iniciar Detecção'</div>
        </body>
        </html>
    """)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start_detection():
    global is_detecting
    is_detecting = True
    return "", 204


@app.route("/stop", methods=["POST"])
def stop_detection():
    global is_detecting
    is_detecting = False
    return "", 204


@app.route("/update_confidence", methods=["POST"])
def update_confidence():
    global confidence_threshold
    confidence_threshold = float(request.form["confidence"])
    return "", 204


@app.route("/get_detections")
def get_detections():
    global latest_detections
    if not is_detecting:
        return jsonify([])
    with detections_lock:
        return jsonify(latest_detections)


@app.route("/get_stats")
def get_stats():
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return jsonify({
        "ram_percent": mem.percent,
        "app_percent": round(proc.memory_percent(), 1),
        "ram_used_mb": round(mem.used / 1024 / 1024),
        "ram_total_mb": round(mem.total / 1024 / 1024),
        "app_mb": round(proc.memory_info().rss / 1024 / 1024),
    })


@app.route("/close", methods=["POST"])
def close_app():
    global is_detecting
    is_detecting = False
    cleanup()
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        def shutdown():
            os.kill(os.getpid(), signal.SIGINT)
        threading.Thread(target=shutdown).start()
        return "Servidor finalizado.", 200
    return "OK", 200


def cleanup():
    global picam2, is_detecting
    is_detecting = False
    if picam2:
        try:
            picam2.stop()
        except:
            pass


if __name__ == "__main__":
    try:
        initialize_camera()

        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()

        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()

        app.run(host="0.0.0.0", port=5000, threaded=True)
    except KeyboardInterrupt:
        print("Encerrando...")
    finally:
        cleanup()