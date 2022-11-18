import argparse
from argparse import ArgumentParser


def parse():

    """Parse command line arguments"""

    # Initiate argparser
    parser = ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--source",
        const="Input",
        default="Input",
        nargs="?",
        type=str,
        help="Directory where media is stored, or file or webcam",
    )

    parser.add_argument(
        "--model",
        const="Model/mmod_human_face_detector.dat",
        default="Model/mmod_human_face_detector.dat",
        nargs="?",
        type=str,
        help="Path to model to be loaded",
    )

    parser.add_argument(
        "--upsample",
        const=1,
        default=1,
        nargs="?",
        type=int,
        help="How many times to upsample the image in order to detect faces",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show (or not) the results of face detection",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save (or not) the results of face detection",
    )

    parser.add_argument(
        "--blur",
        const=None,
        default=None,
        nargs='?',
        type=int,
        help="Blur (or not) the detected faces, and specify level of (median) blur",
    )
    
    parser.add_argument(
        "--recognize",
        const=None,
        default=None,
        nargs='?',
        type=str,
        help="If not none, use the target directory to source images for comparison",
    )

    # Parse arguments
    args = parser.parse_args()

    return args
