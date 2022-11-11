import numpy as np
import cv2


def blur(image: np.array, boxes: list) -> np.array:

    """Blur the faces in a given image"""

    for box in boxes:

        # Crop image to ROI
        crop = image[
            box.rect.top() : box.rect.bottom(), box.rect.left() : box.rect.right()
        ]

        # Blur the ROI
        blurred = cv2.medianBlur(crop, 21)

        # Write blurred ROI
        image[
            box.rect.top() : box.rect.bottom(), box.rect.left() : box.rect.right()
        ] = blurred

    return image
