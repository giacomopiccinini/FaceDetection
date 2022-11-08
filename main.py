import dlib
import logging
import cv2

from Code.Parser.parser import parse
from Code.Loader.MediaLoader import MediaLoader

if __name__ == "__main__":

    # Parse arguments
    args = parse()
    
    # Load face detector
    logging.info("Loading face detection")
    #detector = dlib.cnn_face_detection_model_v1(args.model)
    
    Loader = MediaLoader(args.source)
    
    while True:

        cv2.imshow("Foto", Loader[0])

        # Assign to key "q" the quitting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
    cv2.waitKey(0) 
    
    #closing all open windows 
    cv2.destroyAllWindows() 


    
