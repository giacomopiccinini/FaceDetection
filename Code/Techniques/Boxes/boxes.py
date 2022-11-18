import cv2
import numpy as np
import face_recognition
from copy import deepcopy


def convert_and_trim_bb(image, rectangle):

    """
    Convert the bounding box expression from
        (left, right, top, bottom) to
        (startX, startY, width, height)

    Returns:
        Tuple: (startX, startY, width, height)
    """
    # Extract the dlib bounding box features
    startX = rectangle.left()
    startY = rectangle.top()
    endX = rectangle.right()
    endY = rectangle.bottom()

    # Ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    # Compute the width and height of the bounding box
    width = endX - startX
    height = endY - startY

    return (startX, startY, width, height)


def write_boxes(image, boxes, encodings):

    """Write boxes from detection to image"""
    
   
    # If an encoding dictionary exists
    if encodings:
        
        # Encode all faces
        encoded_images = face_recognition.face_encodings(image, boxes)
                        
        # Initialise the list of names for each face detected
        names = []
        
        for encoded_image in encoded_images:
                                                                   
            # Attempt to match  face with known encodings
            matches = np.array(face_recognition.compare_faces(encodings["encodings"], encoded_image))
                    
            # Store unknown as name for the time being 
            name = "Unknown"
            
            # If at least one match is found
            if matches.any():
                                
                # Indices of matched faces 
                matched_indices = [i for (i, b) in enumerate(matches) if b]
                
                # Initialise dictionary for counting
                counts = {}
                
                for i in matched_indices:
                    
                    # Name of i-th match
                    name = encodings["names"][i]
                    
                    # Increase counter
                    counts[name] = counts.get(name, 0) + 1
                    
                # Majority vote
                name = max(counts, key=counts.get)
	
            # Update the list of names
            names.append(name)
            
    # Fetch boxes
    #boxes = [convert_and_trim_bb(image, r.rect) for r in results]

    # Create copy of image where to write the boxes
    boxed_image = deepcopy(image)
    
    for ((top, right, bottom, left), name) in zip(boxes, names):
        
	    # Draw the box
        cv2.rectangle(boxed_image, (left, top), (right, bottom), (0, 255, 0), 2)
     
        # Set text position
        text_position = top - 15 if top - 15 > 15 else top + 15
     
        # Write text
        cv2.putText(boxed_image, name, (left, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    #cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	#y = top - 15 if top - 15 > 15 else top + 15
	#cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
	#	0.75, (0, 255, 0), 2)

    # Write bounding box on image
    #for (x, y, w, h) in boxes:
    #     cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return boxed_image
