import cv2
from pathlib import Path
from yaml import safe_load
from pymediainfo import MediaInfo


class MediaLoader:

    """Class for loading media (images and videos)"""

    # Constructor
    def __init__(self, path: str) -> None:

        """
            Initialise ImageVideoLoader class starting from a path (string).
            The path could be either:
            - a directory
            - a file

            If a directory, all files therein are load recursively.
            If a single file, only that file is processed.

            Only files with extensions stored in "Settings/format.yaml" are retained.

        Input:
            path: The path to relevant file(s) or "webcam"

        Raises:
            Exception: The path does not exist
            Exception: All indicated files are not admissible
        """

        # Load admissible media formats (either videos or images)
        with open("Settings/format.yaml", "r") as file:

            # Load yaml file
            d = safe_load(file)

            # Split formats
            image_formats = d["image_formats"]
            video_formats = d["video_formats"]

        # Store path and loader mode
        self.path = Path(path).absolute()
        self.mode = "Media"

        # Load files

        # If path is a directory, load all files recursively
        if self.path.is_dir():
            files = self.path.rglob("*")

        # If is a single file, load it alone
        elif self.path.is_file():
            files = [self.path]

        # Else, there has to be an issue
        else:
            raise Exception(f"ERROR: {path} does not exist")

        # Divide between videos and images
        images = [path for path in files if path.suffix in image_formats]
        videos = [path for path in files if path.suffix in video_formats]

        # Unite images and videos
        files = images + videos

        # Retrieve shapes
        shapes = list(map(self.get_media_shape, files))

        # Store info
        self.files = files
        self.shapes = shapes
        self.n_files = len(files)

        # If no files are found, raise error
        assert self.n_files > 0, f"No images or videos found in {path}."

        # Store flag indicating if a file is a video or an image
        self.video_flag = [False] * len(images) + [True] * len(videos)

        # If any video is present
        if any(videos):

            # Initialise the first one, to be used when iterating
            self.new_video(videos[0])
        else:

            # If only images are present, set to None
            self.capture = None

    def __len__(self) -> int:

        """Return length of the Loader i.e. number of files"""

        return self.n_files

    def __iter__(self):

        """Set iteration number when looping"""

        self.count = 0

        return self

    def __next__(self):

        """Iterate through media"""

        # If we reach end of list, stop iteration
        if self.count == self.n_files:
            raise StopIteration

        # Select the path to media at hand
        path = self.files[self.count]

        # If the media is a video
        if self.video_flag[self.count]:

            # Read framemand store if operation is successful
            successful, image_BGR = self.capture.read()

            # If not successful, then video has ended
            if not successful:

                # Increase counter of media files upon completion
                self.count += 1

                # Release capture as we have completed the video
                self.capture.release()

                # If all videos have been processed
                if self.count == self.n_files:
                    raise StopIteration

                # If it is not the last video, start reading the next one
                else:
                    # Retrieve new path after incrementing count
                    path = self.files[self.count]

                    # Initialise new video
                    self.new_video(path)

                    # Read framem and store if operation is successful
                    successful, image_BGR = self.capture.read()

            # Increase frame number by one after reading
            self.frame += 1

        # If we have an image
        else:
            # Read BGR image
            image_BGR = cv2.imread(path, -1)

            # Increase counter
            self.count += 1

            # Check that the image exists
            assert image_BGR is not None, f"Can't read the image at {path}"

        # Convert to RGB
        image_RGB = image_BGR[:, :, ::-1]

        return path, image_RGB, image_BGR, self.capture

    def get_media_shape(self, media_path: Path) -> tuple:

        """Return (height, width) of media, either
        image or video"""

        # Parse the media
        media = MediaInfo.parse(media_path)

        # Only keep video or images
        track = [
            track for track in media.tracks if track.track_type in ["Video", "Image"]
        ][0]

        # Retrieve shape
        shape = (track.height, track.width)

        return shape

    def new_video(self, path) -> None:

        """Method to be called when a new video is processed"""

        # Initialise frame number
        self.frame = 0

        # Capture video using openCV
        self.capture = cv2.VideoCapture(path)

        # Get total number of frames for video at hand
        self.n_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
