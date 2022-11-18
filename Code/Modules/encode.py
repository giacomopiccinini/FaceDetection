import face_recognition
import cv2
import pickle
from pathlib import Path
from tqdm import tqdm
from yaml import safe_load

def encode_references(path: str):
    
    """
    Crate encodings for reference faces
    """
    
    # Load admissible media formats (either videos or images)
    with open("Settings/format.yaml", "r") as file:

        # Load yaml file
        d = safe_load(file)

        # Split formats
        image_formats = d["image_formats"]
    
    # Create Path object
    p = Path(path)
    
    # Find all files
    files = list(p.rglob("*"))
    
    # Divide between videos and images
    images = [path for path in files if path.suffix in image_formats]
    
    # Get people names
    people = [path.parts[-2] for path in images]
    
    # Initialise encodings
    encodings = []
    
    for image in tqdm(images):
        
        # Read image from disk
        image = cv2.imread(image.__str__(), -1)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get bounding box
        box = face_recognition.face_locations(image, model="cnn")
        
        # Encode face
        encoding = face_recognition.face_encodings(image, box)
                
        # Append encoding
        #encodings.append(encoding)
        encodings = encodings + encoding
        
    # Create dictionary for the encodings
    data = {"encodings": encodings, "names":  people}
    
    with open(f"{path}/encodings.pkl", "wb") as file:
        
        # Write pickle
        file.write(pickle.dumps(data))