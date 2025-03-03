# detector.py
from pathlib import Path
import face_recognition
import pickle 
import argparse
from collections import Counter
from PIL import Image, ImageDraw
import cv2  
import dlib
import time
import numpy as np   


BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument(
    "--validate", action="store_true", help="Validate trained model"
)
parser.add_argument(
    "--test", action="store_true", help="Test the model with an unknown image"
)
parser.add_argument(
    "--signup", action="store", help="Sign up a new user by processing a video and storing frames"
)
parser.add_argument(
    "--realtime", action="store_true", help="Enable real-time facial recognition"
)  # New argument for real-time recognition
parser.add_argument(
    "--login", action="store_true", help="Login using facial recognition. Display the name of the user if recognized."
)

parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        if filepath.is_file():
            # Check if the file is an image by verifying its extension
            if filepath.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]:
                print(f"Skipping non-image file: {filepath}")
                continue  # Skip non-image files
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def login(model: str = "hog", duration: int = 10, threshold: float = 0.6) -> None:
    """
    Capture video from the webcam, extract frames, and compare them with the stored encodings to authenticate users.
    If the faces are recognized, the name of the user will be displayed in the output; if not, the face will be displayed as unknown.
    The timer will be 20 seconds, or it will terminate when a recognized face is detected.
    """

    # Load stored encodings
    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    
    # Capture video from the webcam
    video_capture = cv2.VideoCapture(0)  # 0 is the default camera
    video_capture.set(cv2.CAP_PROP_FPS, 120)  # Increase fps for better frame capture

    detected_faces = []
    start_time = time.time()  # Start time for the 20-second duration

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define fixed bounding box dimensions (x, y, width, height)
    fixed_bbox_width, fixed_bbox_height = 800, 800  # Define bounding box size
    fixed_bbox_x = (frame_width - fixed_bbox_width) // 2
    fixed_bbox_y = (frame_height - fixed_bbox_height) // 2

    print("Starting authentication. Look at the camera and place your face inside the bounding box.")
    print("Press 'q' to stop early.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Draw the fixed bounding box on the frame
        cv2.rectangle(frame, (fixed_bbox_x, fixed_bbox_y), 
                      (fixed_bbox_x + fixed_bbox_width, fixed_bbox_y + fixed_bbox_height), 
                      (0, 255, 0), 2)

        # Convert the frame to RGB format for face recognition
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb, model=model)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        for bounding_box, unknown_encoding in zip(face_locations, face_encodings):
            top, right, bottom, left = bounding_box

            # Check if the face is within the fixed bounding box
            if (
                left > fixed_bbox_x and right < (fixed_bbox_x + fixed_bbox_width) and
                top > fixed_bbox_y and bottom < (fixed_bbox_y + fixed_bbox_height)
            ):
                name = _recognize_face(unknown_encoding, loaded_encodings, threshold=threshold)
                detected_faces.append(name)
                print(f"Detected: {name}")
            else:
                print("Face not within the bounding box, please adjust your position.")

        # Display the frame with the fixed bounding box
        cv2.imshow('Login - Place Your Face in the Box', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check for duration limit
        if time.time() - start_time > duration:
            break

    # Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

    # Determine the most frequent face detected
    if detected_faces:
        most_common_face = Counter(detected_faces).most_common(1)[0][0]
        if most_common_face != "Unknown":
            print(f"Authenticated as {most_common_face}.")
        else:
            print("No recognized face detected. Authentication failed.")
    else:
        print("No valid faces were detected during the login process.")

def SignUp(model: str = "hog") -> None:
    """
    Capture video from the webcam, extract faces from frames, save them as images in a user-named folder, and save face encodings in a separate folder.
    The process stops after capturing 30 face images.

    Args:
        model (str): Model to use for face encoding, 'hog' or 'cnn'.
    """
    # Prompt the user to enter their name
    user_name = input("Enter your name for signup: ").strip()
    
    # Create directories for the user's images and encodings
    user_dir = Path("training") / user_name
    encodings_dir = Path("encodings") / user_name
    user_dir.mkdir(parents=True, exist_ok=True)
    encodings_dir.mkdir(parents=True, exist_ok=True)

    # Capture video from the webcam
    video_capture = cv2.VideoCapture(0)  # 0 is the default camera
    video_capture.set(cv2.CAP_PROP_FPS, 100)
    frame_count = 0
    encodings = []

    print("Press 'q' to stop recording or wait until 30 faces are captured.")

    while frame_count < 30:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb, model=model)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            # Crop the face from the frame
            face_frame = frame[top:bottom, left:right]

            # Save the face frame as an image in the user's folder
            frame_filename = user_dir / f"face_{frame_count}.jpg"
            cv2.imwrite(str(frame_filename), face_frame)
            
            encodings.append(encoding)

            frame_count += 1

            if frame_count >= 30:  # Check if 30 frames have been captured
                break

        # Display the frame
        cv2.imshow('Webcam - Face Registration', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

    # Save encodings to a file in the encodings directory
    encodings_path = encodings_dir / f"{user_name}_encodings.pkl"
    with encodings_path.open("wb") as f:
        pickle.dump(encodings, f)

    print(f"Processed {frame_count} face images for {user_name} and stored encodings in {encodings_path}.")


def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()

def _recognize_face(unknown_encoding, loaded_encodings, threshold=0.6):
    """
    Recognize a face by comparing it with stored encodings using a threshold.

    Args:
        unknown_encoding: The face encoding of the unknown face.
        loaded_encodings: The loaded encodings from the pickle file.
        threshold: The confidence threshold for a match.

    Returns:
        The name of the recognized face or "Unknown".
    """
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding, tolerance=threshold
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    return "Unknown"

def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            if filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                print(f"Skipping non-image file: {filepath}")
                continue  # Skip non-image files
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )



def real_time_recognition(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH):
    # Load known face encodings
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
        rgb_small_frame = np.ascontiguousarray(frame[:, :, ::-1])
        
        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame, model=model)

        # Detect face encodings for each detected face
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Recognize the face
            name = _recognize_face(face_encoding, loaded_encodings)
            if not name:
                name = "Unknown"

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow("Video", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.realtime:  
        real_time_recognition(model=args.m)
    if args.signup:
        SignUp(model=args.m)
    if args.login:
        login(model=args.m, duration=5)