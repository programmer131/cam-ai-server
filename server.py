from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch
import os
from datetime import datetime

app = Flask(__name__)
model = YOLO('yolo26n.pt')

# Create directories for saving images
SAVE_DIR = 'processed_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# Detect execution provider
device = 'cuda' if torch.cuda.is_available() else 'cpu'
execution_provider = 'CUDA' if device == 'cuda' else 'CPU'
can_use_gpu = torch.cuda.is_available()

# Request counter
request_counter = 0

@app.route('/v1/vision/detection', methods=['POST'])
def detect():
    global request_counter
    request_counter += 1
    req_id = request_counter
    
    start_time = time.time()
    
    try:
        # Get image from request
        if 'image' not in request.files:
            print(f"[Request {req_id}] ERROR: No image provided")
            return jsonify({
                'success': False,
                'error': 'No image provided',
                'message': 'Image file is required'
            }), 400
        
        # Get min_confidence parameter (default 0.4)
        min_confidence = float(request.form.get('min_confidence', 0.4))
        
        file = request.files['image']
        original_filename = file.filename
        
        print(f"\n{'='*60}")
        print(f"[Request {req_id}] Processing: {original_filename}")
        print(f"[Request {req_id}] Min confidence: {min_confidence}")
        
        # Read and decode image
        process_start = time.time()
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        img_h, img_w = img.shape[:2]
        print(f"[Request {req_id}] Image size: {img_w}x{img_h}")
        
        # Run inference
        inference_start = time.time()
        results = model(img, verbose=False, device=device)
        inference_ms = int((time.time() - inference_start) * 1000)
        
        print(f"[Request {req_id}] Inference time: {inference_ms}ms")
        
        # Parse results and filter by confidence
        predictions = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            
            # Filter by min_confidence
            if conf < min_confidence:
                continue
                
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            
            predictions.append({
                'confidence': round(conf, 2),
                'label': model.names[cls],
                'x_min': int(x1),
                'y_min': int(y1),
                'x_max': int(x2),
                'y_max': int(y2)
            })
        
        # Print detections
        print(f"[Request {req_id}] Detections: {len(predictions)}")
        for i, pred in enumerate(predictions, 1):
            print(f"  {i}. {pred['label']}: {pred['confidence']:.2f} at ({pred['x_min']},{pred['y_min']})-({pred['x_max']},{pred['y_max']})")
        
        # Save processed image with bounding boxes
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_filename = f"{timestamp}_req{req_id}_{os.path.splitext(original_filename)[0]}.jpg"
        save_path = os.path.join(SAVE_DIR, save_filename)
        
        # Draw boxes on image
        plotted_img = results[0].plot()
        cv2.imwrite(save_path, plotted_img)
        
        print(f"[Request {req_id}] Saved to: {save_path}")
        
        process_ms = int((time.time() - process_start) * 1000)
        analysis_ms = int((time.time() - start_time) * 1000)
        
        print(f"[Request {req_id}] Total process time: {process_ms}ms")
        print(f"[Request {req_id}] Total round-trip: {analysis_ms}ms")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'message': 'Object detection successful',
            'predictions': predictions,
            'count': len(predictions),
            'inferenceMs': inference_ms,
            'processMs': process_ms,
            'analysisRoundTripMs': analysis_ms,
            'moduleId': 'yolo26-detection',
            'moduleName': 'YOLO26 Object Detection',
            'command': 'detect',
            'executionProvider': execution_provider,
            'canUseGPU': can_use_gpu,
            'savedImage': save_filename
        })
    
    except Exception as e:
        analysis_ms = int((time.time() - start_time) * 1000)
        print(f"[Request {req_id}] ERROR: {str(e)}")
        print(f"[Request {req_id}] Failed after {analysis_ms}ms")
        print(f"{'='*60}\n")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Detection failed',
            'analysisRoundTripMs': analysis_ms
        }), 500

@app.route('/v1/vision/detection', methods=['GET'])
def status():
    return jsonify({
        'success': True,
        'message': 'YOLO26 Detection Server is running',
        'moduleId': 'yolo26-detection',
        'moduleName': 'YOLO26 Object Detection',
        'command': 'status',
        'executionProvider': execution_provider,
        'canUseGPU': can_use_gpu,
        'totalRequests': request_counter
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"Starting YOLO Detection Server on port 5000...")
    print(f"Execution Provider: {execution_provider}")
    print(f"GPU Available: {can_use_gpu}")
    print(f"Model: YOLO26n")
    print(f"Images will be saved to: {os.path.abspath(SAVE_DIR)}/")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
