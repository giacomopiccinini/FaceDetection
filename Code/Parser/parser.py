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

    # Parse arguments
    args = parser.parse_args()

    return args
