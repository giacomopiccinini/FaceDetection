import numpy as np
import cv2


def blur(image: np.array, boxes: list, blurring: int) -> np.array:

    """Blur the faces in a given image"""

    for box in boxes:

        # Crop image to ROI
        crop = image[
            box.rect.top() : box.rect.bottom(), box.rect.left() : box.rect.right()
        ]
        
        # If blurring is even, convert it to odd
        if blurring%2 == 0:
            blurring += 1
        
        # Blur the ROI
        blurred = cv2.medianBlur(crop, blurring)

        # Write blurred ROI
        image[
            box.rect.top() : box.rect.bottom(), box.rect.left() : box.rect.right()
        ] = blurred

    return image
