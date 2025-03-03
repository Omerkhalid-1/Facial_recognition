from pathlib import Path
import face_recognition
import pickle 
import cv2
import numpy as np
from collections import Counter

BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def SignUp(model: str = "hog") -> None:
    user_name = input("Enter your name for signup: ").strip()
    user_dir = Path("training") / user_name
    extracted_faces_dir = user_dir / "faces"  # Folder for extracted faces
    extracted_faces_dir.mkdir(parents=True, exist_ok=True)

    video_capture = cv2.VideoCapture(0)
    frame_count = 0
    encodings = []

    print("Press 'q' to stop recording. Capturing up to 20 frames...")

    while frame_count < 20:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb, model=model)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        if face_encodings:  # Only save frames with detected faces
            frame_filename = user_dir / f"frame_{frame_count}.jpg"
            cv2.imwrite(str(frame_filename), frame)

            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_image = frame[top:bottom, left:right]  # Crop the face
                face_filename = extracted_faces_dir / f"face_{frame_count}_{i}.jpg"
                cv2.imwrite(str(face_filename), face_image)

            encodings.extend(face_encodings)
            frame_count += 1

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if not encodings:
        print("No faces were captured. Please try again.")
        return

    # Load existing encodings if available
    if DEFAULT_ENCODINGS_PATH.exists():
        with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
            existing_data = pickle.load(f)
    else:
        existing_data = {"names": [], "encodings": []}

    # Append new encodings
    existing_data["names"].extend([user_name] * len(encodings))
    existing_data["encodings"].extend(encodings)

    # Save updated encodings
    with DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
        pickle.dump(existing_data, f)

    print(f"Signup completed. {frame_count} frames processed for {user_name}. Faces saved in {extracted_faces_dir}.")


def Login(model: str = "hog") -> None:
    video_capture = cv2.VideoCapture(0)
    frame_count = 0
    detected_names = []

    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    
    print("Capturing frames for login...")
    
    while frame_count < 20:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb, model=model)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        for unknown_encoding in face_encodings:
            name = _recognize_face(unknown_encoding, loaded_encodings)
            if name:
                detected_names.append(name)
        
        frame_count += 1
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if detected_names:
        most_common_name = Counter(detected_names).most_common(1)[0][0]
        print(f"Access granted: {most_common_name}")
    else:
        print("Access denied. No recognized face.")

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    return votes.most_common(1)[0][0] if votes else None

def main():
    while True:
        print("\nFace Recognition System")
        print("1. Sign Up")
        print("2. Login")
        print("3. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            SignUp()
        elif choice == "2":
            Login()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
