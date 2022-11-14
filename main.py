import dlib
import logging
import cv2

from Code.Parser.parser import parse
from Code.Loader.MediaLoader import MediaLoader
from Code.Techniques.Boxes.boxes import write_boxes
from Code.Techniques.Blur.blur import blur

if __name__ == "__main__":

    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Parse arguments
    args = parse()

    # Check CUDA availability
    cuda_is_available = dlib.cuda.get_num_devices() > 0
    cuda_is_used = dlib.DLIB_USE_CUDA
    log.info(f"Cuda is available: {cuda_is_available}")
    log.info(f"Cuda is used: {cuda_is_used}")

    # Load face detector
    log.info("Loading face detection model")
    detector = dlib.cnn_face_detection_model_v1(args.model)

    # Load media
    log.info("Loading media")
    Data = MediaLoader(args.source)
    
    # Loop over images (or frames) in dataset
    for name, image, capture in Data:
                
        # Detect faces
        boxes = detector(image, args.upsample)
        
        # Write bounding boxes onto the image
        boxed_image = write_boxes(image, boxes)
        
        # Blur image (on request)
        if args.blur:
            blurred = blur(image, boxes)
                
        # Show image (on request)
        if args.show:
            cv2.imshow(name, boxed_image)
            
            # Different rules depending on the type of data
            if Data.mode == "Stream" or Data.mode == "Video":
                if cv2.waitKey(1) & 0xFF == ord("q"):
                        Data.stream.release()
                        break
            # Case of images
            else:
                cv2.waitKey(0)
                
        # Save image (on request)
        if args.save:
            
            # Select image to write
            image_to_write = blurred if args.blur else boxed_image
            
            # In the case of image
            if Data.mode == "Image":
                cv2.imwrite(f"Output/{name}", image_to_write)
            # In case of stream or video
            else:
                pass 