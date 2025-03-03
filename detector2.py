from pathlib import Path
import face_recognition
import pickle 
import argparse
from collections import Counter
from PIL import Image, ImageDraw
import cv2  # Import OpenCV

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

def SignUp(model: str = "hog") -> None:
    """
    Capture video from the webcam, extract frames, save them as images in a user-named folder, and encode faces.

    Args:
        model (str): Model to use for face encoding, 'hog' or 'cnn'.
    """
    # Prompt the user to enter their name
    user_name = input("Enter your name for signup: ").strip()
    
    # Create a directory for the user
    user_dir = Path("training") / user_name
    user_dir.mkdir(parents=True, exist_ok=True)

    # Capture video from the webcam
    video_capture = cv2.VideoCapture(0)  # 0 is the default camera
    frame_count = 0
    encodings = []
    names = []

    print("Press 'q' to stop recording.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb, model=model)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        # Save the frame as an image in the user's folder
        frame_filename = user_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(str(frame_filename), frame)
        
        for encoding in face_encodings:
            names.append(user_name)  # Use the provided user name
            encodings.append(encoding)

        frame_count += 1

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

    # Save encodings to file
    name_encodings = {"names": names, "encodings": encodings}
    with DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

    print(f"Processed {frame_count} frames for {user_name} and stored encodings.")



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

#encode_known_faces()

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

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

#recognize_faces("unknown.jpeg")

def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            if filepath.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                print(f"Skipping non-image file: {filepath}")
                continue  # Skip non-image files
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

# Removed recognize_faces("unknown.jpg")
#validate()

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)
    if args.signup:
        SignUp(model=args.m)