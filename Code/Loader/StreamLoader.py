import cv2


class StreamLoader:

    """Class for loading stream from webcam"""

    # Constructor
    def __init__(self) -> None:

        """
        Initialise StreamLoader class.

        """

        # Set basic info
        self.path = None
        self.mode = "Stream"
        self.files = None
        self.shapes = None
        self.n_files = 1
        self.count = 0

        # Get stream
        self.capture = cv2.VideoCapture(0)

        # Get video FPS
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

        # Get video width
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Get video height
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self) -> int:

        """Return length of the Loader i.e. number of files"""

        return self.n_files

    def __iter__(self):

        """Set iteration number when looping"""

        self.count = 0

        return self

    def __next__(self):

        """Iterate through media"""

        # Increment counting
        self.count += 1

        # Capture the stream frame
        successful, image = self.capture.read()

        # Exit from stream
        if cv2.waitKey(1) == ord("q"):

            cv2.destroyAllWindows()
            raise StopIteration

        return "webcam.mp4", image, self.capture
