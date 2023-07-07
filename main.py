import dlib
import logging
import cv2
import os
import pickle
import face_recognition
from tqdm import tqdm

from Code.Parser.parser import parse
from Code.Loader.StreamLoader import StreamLoader
from Code.Loader.MediaLoader import MediaLoader
from Code.Modules.encode import encode_references
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

    # Initialise encodings
    encodings = None

    if args.source != "webcam":
        # Load media
        log.info("Loading media")
        Data = MediaLoader(args.source)
    else:
        # Load Stream
        log.info("Loading stream")
        Data = StreamLoader()

    if args.recognize:

        # Check if reference is present
        if "encodings.pkl" in os.listdir(args.recognize):
            # Load reference images
            log.info("Encodings file is present, loading it")
            # Load encodings
            try:
                encodings = pickle.loads(
                    open(f"{args.recognize}/encodings.pkl", "rb").read()
                )
                log.info("Encodings correctly loaded")
            except:
                log.error("Failed loading encodings.pkl")
        else:
            # Create encodings
            log.info("Encodings not found")
            log.info("Creating encodings")

            # Create encodings for references
            encodings = encode_references(args.recognize)

    # Set variables for video writing (when necessary)
    video_name, video_writer = None, None

    # Loop over images (or frames) in dataset
    for name, image, capture in tqdm(Data):

        # Detect faces
        boxes = face_recognition.face_locations(image, model="cnn")

        # Write boxes if not blurring
        if not args.blur:
            # Write bounding boxes onto the image
            boxed_image = write_boxes(image, boxes, encodings, tolerance=0.5)

        # Blur image (on request)
        if args.blur:
            blurred = blur(image, boxes, args.blur)

        # Select image to write (or show)
        image_to_write = blurred if args.blur else boxed_image

        # When streaming, always show the result
        if Data.mode == "Stream":
            cv2.imshow(name, image_to_write)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                Data.capture.release()
                break

        # Show image (on request) if not streaming
        if args.show:
            cv2.imshow(name, image_to_write)

            # Different rules depending on the type of data
            if Data.mode == "Video":
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    Data.capture.release()
                    break
            # Case of images
            else:
                cv2.waitKey(0)

        # Save image (on request)
        if args.save:

            # In the case of image
            if Data.mode == "Image":
                cv2.imwrite(f"Output/{name}", image_to_write)

            # In case of stream or video
            else:

                # If we can read from the video
                if capture and Data.mode == "Video":

                    # Get video FPS
                    fps = capture.get(cv2.CAP_PROP_FPS)

                    # Get video width
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

                    # Get video height
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Case of stream
                else:

                    # Fetch values
                    fps = 8  # Data.fps
                    width = Data.width
                    height = Data.height

                # Initiate video name the first time
                if not video_name:
                    video_name = name

                # Case of new video
                if video_name != name:

                    # Release old video writer
                    video_writer.release()

                    # Re-initialise video writer
                    video_writer = None

                    # Reset video name
                    video_name = name

                # If no video writer is instantiated
                if not video_writer:

                    # Create a new video writer
                    video_writer = cv2.VideoWriter(
                        f"Output/{name}",
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                video_writer.write(image_to_write)
