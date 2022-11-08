def convert_and_trim_bb(image, rectangle):

    """
    Convert the bounding box expression from
        (left, right, top, bottom) to
        (startX, startY, width, height)

    Returns:
        Tuple: (startX, startY, width, height)
    """
    # Extract the dlib bounding box features
    startX = rectangle.left()
    startY = rectangle.top()
    endX = rectangle.right()
    endY = rectangle.bottom()

    # Ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    # Compute the width and height of the bounding box
    width = endX - startX
    height = endY - startY

    return (startX, startY, width, height)
