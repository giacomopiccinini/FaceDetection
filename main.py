import dlib
import logging
import cv2

from chronopy.Clock import Clock

from Code.Parser.parser import parse
from Code.Loader.MediaLoader import MediaLoader
from Code.Techniques.Boxes.boxes import write_boxes
from Code.Techniques.Blur.blur import blur

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
            boxes = detector(image, args.upsample)

            # Log results
            log.info(f"Detected {len(boxes)} in {name}")

            # Write bounding boxes onto the image
            boxed_image = write_boxes(image, boxes)

            # Move to next image
            i += 1

            # Blur image (on request)
            if args.blur:
                blurred = blur(image, boxes)

            # Show image (on request)
            if args.show:
                cv2.imshow(name, boxed_image)
                if Loader.mode == "Stream":
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        Loader.stream.release()
                        break
                else:
                    cv2.waitKey(0)

            # Save image (on request)
            if args.save:
                if args.blur:
                    cv2.imwrite(f"Output/{name}", blurred)
                else:
                    cv2.imwrite(f"Output/{name}", boxed_image)

        except Exception as e:
            break

    # Closing all open windows
    cv2.destroyAllWindows()

    clock.lap()
    clock.summary()
