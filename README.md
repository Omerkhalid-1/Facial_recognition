
## Installation
To install the required dependencies, run:
```sh
pip install -r requirements.txt
```

## Usage

### 1. Signup (Register a New User)
To sign up a new user, use the following command:
```sh
python detector.py --signup
```
- The system will start capturing video from your webcam.
- Press 'q' to stop recording and save the frames.
- The face encodings will be stored for future login.

### 2. Train the Model
To train the face recognition model using the stored images:
```sh
python detector.py --train
```
- This command processes images in the `training` directory and encodes faces.

### 3. Validate the Model
To validate the model against images in the `validation` directory:
```sh
python detector.py --validate
```
- This will check if the trained model can correctly recognize faces in the validation images.

### 4. Test (Login with an Unknown Image)
To test the model with an unknown image:
```sh
python detector.py --test -f path/to/image.jpg
```
- Replace `path/to/image.jpg` with the actual path of the image.
- The system will attempt to recognize the face in the image.
