from flask import Flask, render_template, Response, url_for
import datetime
import cv2
import json
import numpy as np
from FPTvision.app import FaceAnalysis
from FPTvision.model_zoo import arcface_onnx
from FPTvision.utils import face_align

# Khởi tạo mô hình
Detection = FaceAnalysis()
Detection.prepare(ctx_id=0, det_size=(640, 480))

Recognition = arcface_onnx.ArcFaceONNX()
Recognition.prepare(ctx_id=0)


app = Flask(__name__)



# Đọc embeddings từ file JSON
with open('./data/AI17BH1.json', 'r') as json_file:
    embeddings_dict = json.load(json_file)
    

# Trọng số cho mỗi vector embedding
WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]

students_dict = {}


def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    def save_face_image(frame, kps, student_id, img_type="in"):
        """Lưu hình ảnh khuôn mặt được crop vào đường dẫn chỉ định."""
        face_image = face_align.norm_crop(frame, kps)
        img_path = f'app/static/img/avatars/{student_id}/{img_type}.png'
        cv2.imwrite(img_path, face_image)

    def update_student_record(student_id, frame, kps, checkin=False):
        """Cập nhật thông tin check-in hoặc check-out cho sinh viên."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_type = "in" if checkin else "out"
        save_face_image(frame, kps, student_id, img_type)
        if student_id in students_dict:
            if checkin:
                students_dict[student_id]["CheckInTime"] = now
            else:
                students_dict[student_id]["CheckOutTime"] = now

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        faces = Detection.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)

            detected_embedding = Recognition.get(frame, face)
            
            identify_name = ""
            max_similarity = 0.0
            for person_name, embeddings in embeddings_dict.items():
                weighted_similarity = 0.0
                for i, (vector_name, vector) in enumerate(embeddings.items()):
                    similarity = Recognition.compute_sim(np.array(vector), detected_embedding)
                    weighted_similarity += WEIGHTS[i] * similarity

                if weighted_similarity > max_similarity:
                    identify_name = person_name
                    max_similarity = weighted_similarity

            if identify_name:
                student_id, _ = identify_name.split('-', 1)
                
                if student_id in students_dict:
                    if students_dict[student_id]["Status"] != "Present":
                        students_dict[student_id]["Status"] = "Present"
                        update_student_record(student_id, frame, face.kps, checkin=True)
                    else:
                        update_student_record(student_id, frame, face.kps, checkin=False)
                        
                cv2.putText(frame, f'{student_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (242, 111, 33), 2)
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/StudentChecking')
def update_student_data():
    return students_dict

@app.route('/initdata')
def init_data():
    # students_dict = {}
    for key, value in embeddings_dict.items():
        student_id, full_name = key.split('-', 1)
        checkin_time = value.get("checkin_time", "N/A")
        checkout_time = value.get("checkout_time", "N/A")
        student_data = {
            "StudentID": student_id,
            "FullName": full_name,
            "Avatar": url_for('static', filename=f'img/avatars/{student_id}/avatar.png'),
            "CheckInImage": url_for('static', filename=f'img/avatars/{student_id}/in.png'),
            "CheckInTime": checkin_time,    
            "CheckOutImage": url_for('static', filename=f'img/avatars/{student_id}/out.png'),
            "CheckOutTime": checkout_time,
            "HandRissing": 0,
            "Status": 'Processing',
            "Frames": 0,
            "DistractiveFrames": 0, 
        }
        # Sử dụng StudentID làm khóa
        students_dict[student_id] = student_data
    return students_dict

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/courses')
def courses():
    return render_template('courses.html')

@app.route('/classroom')
def classroom():
    return render_template('classroom.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
