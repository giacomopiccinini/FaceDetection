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

    
    # Parse arguments
    args = parser.parse_args()

    return args