import cv2
from pathlib import Path
from Code.Classes.Image import Image


class MediaLoader:

    """
    Load media files from disk or stream from webcam
    """

    def __init__(self, path: str) -> None:

        if path == "webcam":
            self.path = None
            self.mode = "Stream"
            self.stream = cv2.VideoCapture(0)
        else:
            self.path = Path(path).absolute()
            self.mode = "Media"

            # Load files
            if self.path.is_dir():
                files = self.path.rglob(
                    "*"
                )  # If path is a directory, load all files recursively
            elif self.path.is_file():
                files = [self.path]  # If is a single file, load it alone
            else:
                raise Exception(f"ERROR: {path} does not exist")
            self.files = [Image.from_path(file.__str__()) for file in files]

    def __getitem__(self, i: int):

        """Return the i-th element of the Loader, in the form (image, name)"""

        if self.mode == "Stream":

            try:
                # Capture the stream frame
                successful, frame = self.stream.read()
            except:
                raise Exception(f"ERROR loading the stream")

            return frame, "Stream"

        else:
            return self.files[i].tensor, self.files[i].name
