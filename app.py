from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = r'C:\Users\kulde\Downloads\projects\excelr_kd\computer vision project\photos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load YOLO model
model = YOLO(r"C:\Users\kulde\Downloads\projects\excelr_kd\computer vision project\yolov10x.pt")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform object detection
            results = model(filepath)
            
            # Get detected objects
            detected_objects = results[0].boxes.cls.tolist()
            detected_names = [model.names[int(obj)] for obj in detected_objects]
            
            # Save the image with bounding boxes
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                detected_filename = 'detected_' + filename
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], detected_filename), im)
            
            return render_template('result.html', filename=detected_filename, detected_objects=detected_names)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)