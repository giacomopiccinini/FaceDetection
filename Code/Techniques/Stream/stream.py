import cv2


def stream() -> None:

    """
    Stream the recording of the webcam on the computer
    """

    # Capture video
    video = cv2.VideoCapture(0)

    while True:

        # Capture the video frame by frame
        successful, frame = video.read()

        # Display the resulting frame
        cv2.imshow("Stream", frame)

        # Assign to key "q" the quitting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release capture
    video.release()

    # Destroy all windows or else you will have a frozen window hanging
    cv2.destroyAllWindows()
