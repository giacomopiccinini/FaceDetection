import cv2
import numpy as np
from pathlib import Path
from yaml import safe_load
from Code.Classes.Image import Image


class MediaLoader:

    """
    Load media files from disk or stream from webcam
    """

    def __init__(self, path: str) -> None:

        """
            Initialise MediaLoader class starting from a path (string).
            The path could be either:
            - a directory
            - a file
            - the string "webcam"

            If a directory, all files therein are load recursively.
            If a single file, only that file is processed.
            If "webcam", the device webcam is turned on and its recording is
            used as input

            Only files with extensions stored in "Settings/format.yaml" are retained.

        Input:
            path: The path to relevant file(s) or "webcam"

        Raises:
            Exception: The path does not exist
            Exception: All indicated files are not admissible
        """

        # Load admissible media formats (either videos or images)
        with open("Settings/format.yaml", "r") as file:
            formats = safe_load(file)
            self.formats = formats["image_formats"] + formats["video_formats"]

        # Class initialisation

        # Case of stream for device webcam
        if path == "webcam":
            self.path = None
            self.mode = "Stream"
            self.stream = cv2.VideoCapture(0)

        # Case of actual media (image, video)
        else:
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

            # Make sure only admissibile formats are retained
            files = [file for file in files if file.suffix in self.formats]

            # Raise error if no admissibile files are left
            if len(files) == 0:
                raise Exception(f"ERROR: no admissible files")

            # Assign files to class
            self.files = [Image.from_path(file.__str__()) for file in files]

    def __getitem__(self, i: int) -> tuple[np.array, str]:

        """Return the i-th element of the Loader, in the form (image, name)

        Input:
            int: Integer for index of file in the data loader

        Raises:
            Exception: Raise error when there is an issue with the stream from the webcam

        Returns:
            tuple: Tuple of NumPy array (the image/frame) and its name, together with suffix.
        """

        if self.mode == "Stream":

            try:
                # Capture the stream frame
                successful, frame = self.stream.read()
            except:
                raise Exception(f"ERROR loading the stream")

            return frame, "Stream"

        else:
            return self.files[i].tensor, self.files[i].name
