from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch

app = Flask(__name__)
model = YOLO('yolo26n.pt')

# Detect execution provider
device = 'cuda' if torch.cuda.is_available() else 'cpu'
execution_provider = 'CUDA' if device == 'cuda' else 'CPU'
can_use_gpu = torch.cuda.is_available()

@app.route('/v1/vision/detection', methods=['POST'])
def detect():
    start_time = time.time()
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided',
                'message': 'Image file is required'
            }), 400
        
        # Get min_confidence parameter (default 0.4)
        min_confidence = float(request.form.get('min_confidence', 0.4))
        
        file = request.files['image']
        
        # Read and decode image
        process_start = time.time()
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run inference
        inference_start = time.time()
        results = model(img, verbose=False, device=device)
        inference_ms = int((time.time() - inference_start) * 1000)
        
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
        
        process_ms = int((time.time() - process_start) * 1000)
        analysis_ms = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'message': 'Object detection successful',
            'predictions': predictions,
            'count': len(predictions),
            'inferenceMs': inference_ms,
            'processMs': process_ms,
            'analysisRoundTripMs': analysis_ms,
            'moduleId': 'yolov8-detection',
            'moduleName': 'YOLOv8 Object Detection',
            'command': 'detect',
            'executionProvider': execution_provider,
            'canUseGPU': can_use_gpu
        })
    
    except Exception as e:
        analysis_ms = int((time.time() - start_time) * 1000)
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
        'message': 'YOLOv8 Detection Server is running',
        'moduleId': 'yolov8-detection',
        'moduleName': 'YOLOv8 Object Detection',
        'command': 'status',
        'executionProvider': execution_provider,
        'canUseGPU': can_use_gpu
    })

if __name__ == '__main__':
    print(f"Starting YOLO Detection Server on port 5000...")
    print(f"Execution Provider: {execution_provider}")
    print(f"GPU Available: {can_use_gpu}")
    app.run(host='0.0.0.0', port=5000, threaded=True)