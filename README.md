# Face Expression Recognition Machine Learning Model
![image](https://github.com/prak-perks/emotiondetection/assets/117466688/b997e5f8-6ede-4b10-b3dd-d73fba3ae5f4)

Facial expression emotion recognition is a valuable indicator of a person's mental state, providing a plethora of emotional information and serving as an important element in interpersonal communicati
on. Its applications span diverse fields, including psychology and marketing. Our (ML) application aims to tackle the challenge of accurately identifying emotions from facial expressions. 

Tracking the emotions of users has the potential to completely revolutionize marketing strategies by enabling the analysis of user's emotional responses to different marketing campaigns. By tracking facial expressions, firms can successfully identify the campaigns and ads that are resonating with their target audience and refine their overall marketing strategies accordingly. Personalizing content based on the emotional response of users can enhance their experience and increase the likelihood of desired actions being taken, such as making a purchase. Additionally, ad targeting can be improved by showing users ads related to content that they have expressed positive emotions towards, thereby increasing the effectiveness of campaigns and reducing ad fatigue.

## System Design
![systemdesign](https://github.com/prak-perks/emotiondetection/assets/117466688/dfc68d33-17e4-4574-9469-83b1f706c124)

Step 1: User input - Users can input an image URL by copying and pasting it on the website. 

Step 2: Backend - The backend receives the input image and performs the following tasks: 

Decode the image to grayscale to simplify the input for the CNN model 

Apply the pre-trained CNN model to the image to recognize the emotion. The results are then passed to the Flask API 

Step 3: Frontend - The Flask API sends the predicted emotion back to the website, where it is displayed to the user. This system design uses a pre-trained CNN model to recognize emotions based on grayscale images. Users can input their images via a website, which then sends the input to the backend. The backend processes the image and sends the predicted emotion back to the frontend, which displays it on the website.
