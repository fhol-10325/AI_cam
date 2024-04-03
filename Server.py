from flask import Flask, Response
import cv2

app = Flask(__name__)

def gen_frames():  
    # Camera setup
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    if not cap.isOpened():
        raise RuntimeError("Could not start camera.")

    while True:
        success, frame = cap.read()  # Read frame from camera
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate frame data

@app.route('/video_feed')
def video_feed():
    # Return the response generated along with the specific media type (mime type).
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)
