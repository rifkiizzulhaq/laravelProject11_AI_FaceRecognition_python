from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

DATASET_PATH = 'dataset'
TRAINER_PATH = 'trainer'
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(TRAINER_PATH, exist_ok=True)

# Tambahkan route untuk root path
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'running',
        'message': 'Face Recognition Server is active'
    })

# Di bagian initialize_face_recognizer()
def initialize_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=35.0  # Diperketat dari 40.0
    )
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return recognizer, detector
def check_face_similarity(new_face_img, threshold=85.0):
    """
    Check if a face is already registered by comparing with existing dataset
    Returns (is_duplicate, existing_face_id)
    """
    try:
        # Skip if no training file exists yet
        trainer_path = os.path.join(TRAINER_PATH, 'trainer.yml')
        if not os.path.exists(trainer_path):
            return False, None

        # Initialize recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=12,
            grid_x=8,
            grid_y=8
        )
        recognizer.read(trainer_path)

        # Try to recognize the face
        face_id, confidence = recognizer.predict(new_face_img)
        
        print(f"Similarity check - Face ID: {face_id}, Confidence: {confidence}")
        
        # If confidence is low enough, it means the face is similar to an existing one
        if confidence < threshold:
            return True, face_id
            
        return False, None

    except Exception as e:
        print(f"Error checking face similarity: {str(e)}")
        return False, None

# Tambahkan endpoint untuk clear face data
@app.route('/clear_face_data', methods=['POST'])
def clear_face_data():
    """
    Endpoint untuk menghapus semua data wajah
    """
    try:
        # Hapus semua file di folder dataset
        for file in os.listdir(DATASET_PATH):
            file_path = os.path.join(DATASET_PATH, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Hapus trainer file
        trainer_path = os.path.join(TRAINER_PATH, 'trainer.yml')
        if os.path.exists(trainer_path):
            os.remove(trainer_path)

        print("All face data cleared successfully")
        return jsonify({'message': 'All face data cleared successfully'})

    except Exception as e:
        print(f"Error clearing face data: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Update fungsi register_face dengan lebih banyak variasi
@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400

        face_id = request.form.get('face_id')
        if not face_id:
            return jsonify({'error': 'No face_id provided'}), 400

        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400
        
        if len(faces) > 1:
            return jsonify({'error': 'Multiple faces detected'}), 400

        # Get the detected face
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        # Generate more variations
        variations = []
        
        # Original
        variations.append(face_img)
        
        # Brightness variations (lebih banyak variasi)
        for alpha in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            variant = cv2.convertScaleAbs(face_img, alpha=alpha, beta=0)
            variations.append(variant)
        
        # Rotation variations (lebih banyak sudut)
        for angle in [-10, -7, -5, -3, 3, 5, 7, 10]:
            matrix = cv2.getRotationMatrix2D((100, 100), angle, 1.0)
            variant = cv2.warpAffine(face_img, matrix, (200, 200))
            variations.append(variant)
        
        # Scale variations
        for scale in [0.95, 0.97, 1.0, 1.03, 1.05]:
            width = int(200 * scale)
            height = int(200 * scale)
            scaled = cv2.resize(face_img, (width, height))
            if scale != 1.0:
                scaled = cv2.resize(scaled, (200, 200))
            variations.append(scaled)

        # Gaussian blur variations untuk simulasi foto tidak fokus
        for sigma in [0.5, 1.0, 1.5]:
            blurred = cv2.GaussianBlur(face_img, (5, 5), sigma)
            variations.append(blurred)

        # Noise variations
        for i in range(3):
            noise = np.random.normal(0, 2, face_img.shape).astype(np.uint8)
            variant = cv2.add(face_img, noise)
            variations.append(variant)

        # Save all variations
        for i, variant in enumerate(variations):
            filename = f"face.{face_id}.{i}.jpg"
            filepath = os.path.join(DATASET_PATH, filename)
            cv2.imwrite(filepath, variant)

        # Train the model after saving variations
        try:
            train_face()
            training_success = True
        except Exception as train_error:
            print(f"Training error: {str(train_error)}")
            training_success = False

        response_data = {
            'status': 'success' if training_success else 'partial_success',
            'message': 'Face registered and trained successfully' if training_success else 'Face registered but training failed',
            'face_id': face_id,
            'variations_count': len(variations)
        }

        if not training_success:
            response_data['training_error'] = str(train_error)

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in register_face: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/train_face', methods=['POST'])
def train_face():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=12,
            grid_x=8,
            grid_y=8
        )
        
        face_samples = []
        face_ids = []
        
        # Verify dataset directory exists and has files
        if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
            print("No training data available")
            return jsonify({'error': 'No training data available'}), 400

        for filename in os.listdir(DATASET_PATH):
            if filename.endswith(".jpg"):
                path = os.path.join(DATASET_PATH, filename)
                face_id = int(filename.split('.')[1])
                
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {filename}")
                    continue
                
                # Normalize image
                img = cv2.equalizeHist(img)
                
                # Verify image dimensions
                if img.shape[0] != 200 or img.shape[1] != 200:
                    img = cv2.resize(img, (200, 200))
                
                face_samples.append(img)
                face_ids.append(face_id)

        if len(face_samples) < 1:  # Changed from 10 to 1 for testing
            print("Insufficient training data")
            return jsonify({'error': 'Insufficient training data'}), 400

        print(f"Training with {len(face_samples)} samples")
        recognizer.train(face_samples, np.array(face_ids))
        
        # Save trainer file
        trainer_path = os.path.join(TRAINER_PATH, 'trainer.yml')
        recognizer.save(trainer_path)
        print(f"Training completed and model saved to {trainer_path}")

        return jsonify({'message': 'Training completed successfully'})

    except Exception as e:
        print(f"Error in train_face: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    try:
        print("Received face recognition request")
        
        if 'image' not in request.files:
            print("No image file uploaded")
            return jsonify({'error': 'No image file uploaded'}), 400

        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            print("No face detected in image")
            return jsonify({'error': 'No face detected in image'}), 400

        # Get the detected face
        (x, y, w, h) = faces[0]
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        face_img = cv2.equalizeHist(face_img)

        # Initialize recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=12,
            grid_x=8,
            grid_y=8
        )

        # Load trained model
        trainer_path = os.path.join(TRAINER_PATH, 'trainer.yml')
        if not os.path.exists(trainer_path):
            print("No trained model found")
            return jsonify({'error': 'Face recognition model not trained'}), 400

        recognizer.read(trainer_path)
        
        # Recognize face
        face_id, confidence = recognizer.predict(face_img)
        print(f"Recognition result - Face ID: {face_id}, Confidence: {confidence}")

        # Meningkatkan threshold menjadi 100.0 (lebih toleran)
        CONFIDENCE_THRESHOLD = 100.0
        
        if confidence < CONFIDENCE_THRESHOLD:  # Lower value means better match
            print(f"Face recognized - ID: {face_id}, Confidence: {confidence}")
            return jsonify({
                'face_id': str(face_id),
                'confidence': float(confidence),
                'message': 'Face recognized successfully'
            })
        else:
            print(f"Face not recognized - Confidence too low: {confidence}")
            return jsonify({
                'error': 'Face not recognized',
                'confidence': float(confidence),
                'threshold': float(CONFIDENCE_THRESHOLD)
            }), 401

    except Exception as e:
        print(f"Error in recognize_face: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)