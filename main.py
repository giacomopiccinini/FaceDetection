import dlib
import logging
import cv2

from chronopy.Clock import Clock

from Code.Parser.parser import parse
from Code.Loader.MediaLoader import MediaLoader
from Code.Techniques.Boxes.boxes import write_boxes

if __name__ == "__main__":

    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Initiate clock
    clock = Clock()

    # Parse arguments
    args = parse()

    # Check CUDA availability
    cuda_is_available = dlib.cuda.get_num_devices() > 0
    cuda_is_used = dlib.DLIB_USE_CUDA
    log.info(f"Cuda is available: {cuda_is_available}")
    log.info(f"Cuda is used: {cuda_is_used}")

    # Load face detector
    log.info("Loading face detection model")
    clock.lap("Model loading")
    detector = dlib.cnn_face_detection_model_v1(args.model)

    # Load media
    log.info("Loading media")
    clock.lap("Media loading")
    Loader = MediaLoader(args.source)

    # Create bounding boxes
    i = 0
    while True:
        try:

            # Load i-th image (or frame)
            image, name = Loader[i]

            # Detect faces
            clock.lap(f"Detection {name}")
            results = detector(image, args.upsample)

            # Log results
            log.info(f"Detected {len(results)} in {name}")

            # Write bounding boxes onto the image
            boxed_image = write_boxes(image, results)

            # Move to next image
            i += 1

            # Show image (on request)
            if args.show:
                cv2.imshow(name, boxed_image)
                cv2.waitKey(0)

            # Save image (on request)
            if args.save:
                cv2.imwrite(name, boxed_image)

        except:
            break

    # Closing all open windows
    cv2.destroyAllWindows()

    clock.lap()
    clock.summary()
