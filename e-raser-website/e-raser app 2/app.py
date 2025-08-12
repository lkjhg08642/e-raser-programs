import cv2
import base64
import numpy as np
from flask import Flask, render_template, Response
from flask import request, jsonify
import base64
import re
from fractions import Fraction

app = Flask(__name__)

wb_width = 0
wb_height = 0

# Replace with your Feather's BLE name or address
TARGET_NAME = "Nano33IoT_UART"
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

@app.route('/save_dimensions', methods=['POST'])
def save_dimensions():
    data = request.get_json()
    wb_width = data.get('width')
    wb_height = data.get('height')
    x = Fraction(int(wb_width)/int(wb_height)).limit_denominator()
    wb_width = x.numerator * 800
    wb_height = x.denominator * 800
    return jsonify({"message": f"Width received: {wb_width}"}), 200


@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    image_data = data['image']

    # Remove the base64 header if present
    match = re.match(r'data:image/(png|jpeg);base64,(.*)', image_data)
    if not match:
        return jsonify({"error": "Invalid image data format"}), 400

    ext, img_str = match.groups()
    filename = f"static/saved_image.jpeg"

    # Decode and save
    with open(filename, "wb") as f:
        f.write(base64.b64decode(img_str))


    return jsonify({"message": "Image saved successfully"}), 200

@app.route('/detect_board', methods=['POST'])
def detect_board():
    img = cv2.imread("static/saved_image.jpeg")
    cv2.imwrite("static/edited_image.jpeg", img)

    '''scale_percent = 20  # Resize to 40% of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img = cv2.resize(img, (width, height))
    orig = img.copy()'''

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced_gray, (3,3), 0)

    pixels = np.array(enhanced_gray)
    average_gray = pixels.mean()
    thresh_value = average_gray
    print(average_gray)
    thresh = cv2.threshold(blur, thresh_value-20, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((3, 3), np.uint8)
    thick_lines = cv2.erode(thresh, kernel, iterations=5)

    kernel = np.ones((7,7), np.uint8)
    morph = cv2.morphologyEx(thick_lines, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    page = np.zeros_like(img)
    cv2.drawContours(page, [big_contour], 0, (255,255,255), -1) 
    peri = cv2.arcLength(big_contour, True)

    corners = cv2.approxPolyDP(big_contour, 0.05 * peri, True)
    polygon = img.copy()
    #cv2.polylines(polygon, [corners], True, (0,0,255), 1, cv2.LINE_AA)

    if len(corners) == 4:
        cv2.polylines(polygon, [corners], True, (0,0,255), 1, cv2.LINE_AA)
        cv2.imwrite("static/edited_image.jpeg", polygon)


    return jsonify({"message": "Image saved successfully"}), 200

# Initialize OpenCV webcam capture
cap = cv2.VideoCapture(0)

'''
@app.route('/submit_info', methods=['POST'])
def submit_info():
    data = request.get_json()
    width = data.get('width')
    height = data.get('height')
    widthDimension = data.get('width_dimension')
    heightDimension = data.get('height_dimension')
    print(f"Received width: {width}\nReceived height: {height}\nReceived width dimension: {widthDimension}\nReceived height dimension: {heightDimension}")  # You can do whatever you want with it

    # Respond back
    return jsonify({"message": f"Width received: {width}"})
'''
# Function to generate frames for MJPEG streaming
def generate_frames():
    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break
        else:
            # Convert the frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                # Yield the frame in the MJPEG format
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                
# Route to serve MJPEG stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve the main page
@app.route('/')
def index():
    return render_template('index2.html')

#asyncio.run(mainloop())
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
