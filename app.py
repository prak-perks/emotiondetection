import io
import socket
import requests
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array, get_file
from keras.models import load_model
from flask import Flask, request, render_template,Response
import cv2

app = Flask(__name__)

# Load the emotion recognition model
model = load_model('model.h5')

#Loading the face detection algorithm
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Assigning the emotional labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function for searching the free port for Flask
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
    
    
# Function for capturing the video prediction 
def generate_frames():
    camera=cv2.VideoCapture(0)
    while True: 
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=6)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)
                    prediction = model.predict(roi)[0]
                    label=emotions[prediction.argmax()]
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    
                
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    
    
    
    
 # Intial page and Image prediction
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
       return(render_template('index.html'))
    if request.method == 'POST':
        # Get the image URL from the request
        image_url = request.form['image_url']
        # Download the image and convert it to grayscale
        image_data = requests.get(image_url).content
        
        np_array = np.frombuffer(image_data, np.uint8)
        
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_classifier.detectMultiScale(gray_image,scaleFactor=1.1,minNeighbors=20,flags=cv2.CASCADE_SCALE_IMAGE, minSize= [30, 30])
        
        if len(faces)!=0 :
            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
                roi_gray = gray_image[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        
        
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

                prediction = model.predict(roi)[0]
                label=emotions[prediction.argmax()]
                return render_template('result.html',image_url=image_url,predicted_emotion=label)
        else :
            image = Image.open(io.BytesIO(image_data)).convert('L')
            # Resize the image to 48x48 pixels
            image = image.resize((48, 48))
            # Convert the image to a numpy array and normalize it
            image = img_to_array(image) / 255.0
            # Add an extra dimension to the array to match the input shape of the model
            image = np.expand_dims(image, axis=0)
            # Make a prediction
            prediction = model.predict(image)
            # Convert the prediction to a human-readable label
            #emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            predicted_emotion = emotions[np.argmax(prediction)]
            # Render the result template with the predicted emotion
            return render_template('result.html',image_url=image_url,predicted_emotion=predicted_emotion)

# Opening the video prediction
@app.route('/videopage')
def videopage():
    return render_template('videopage.html')

# Feeding the video in the interface along with prediction
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

# Executing this python file
if __name__ == '__main__':
    free_port = find_free_port()
    app.run(debug=False, port=free_port)
