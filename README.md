# face_recognition_flask_app_one_shot_learning
Using a siamese network, face recognition with one shot learning

Deployment using Flask:

Now as the model for face recognition is finalized further we can proceed to building a simple web app using Flask.
Main components in Face Recognition Web App:

    Capturing Image of a Person
    Adding Person to database
    Predicting the person
    Deleting the identity from Database


Capturing Image: With the help of webcam image of the person is captured and saved in the specify directory of that identity.


Adding Person to Database: By entering the name of the person and uploading the image model predicts the 128d encodings of that person and saves in the database. If the person is already exists in database then it wont add in database.
Adding person to database with encodings.

Deleting a Person From Database: By entering the name of the person and clicking delete the person’s information is deleted from database.
Deleting person’s info

Predicting person : By uploading image of the person we can predict the person in the image if he exists in database else model will be predicted as unknown. After adding new person to database we need to keep an eye on threshold value for predicting unknown.
![image](https://user-images.githubusercontent.com/52566550/176743550-451d026b-5f18-471c-920e-bb0beccb42f2.png)
