import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(
        description='COMP9517 20T1 Project - Pedestrian Detecting, Tracking and Clustering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-si',
        '--source-images',
        help='(REQUIRED) Source images, can be either a directory or wildcard files, e.g. sequence/**/*.jpg',
        dest='path',
        default=str(Path(__file__).resolve().parent.parent / 'sequence' / '*.jpg'),
    )
    parser.add_argument(
        '-sv',
        '--source-video',
        help='(REQUIRED) Source video, can be a camera index, a file or a url.',  # TODO: Unfinished
        dest='path_video',
        default='',
    )
    parser.add_argument(
        '-iw',
        '--image-width',
        help='Image width in pixels for resizing. '
             '0: take width from the first image; '
             '-1: same with 0 but keeps aspect ratio and creates black edges.',
        dest='width',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '-ih',
        '--image-height',
        help='Image height in pixels for resizing.'
             '0: take width from the first image; '
             '-1: same with 0 but keeps aspect ratio and creates black edges.',
        dest='height',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '-fps',
        '--frames-per-second',
        help='Playback frames per second.',
        dest='fps',
        type=float,
        default=10,
    )
    parser.add_argument(
        '--frame-skipping',
        help='Enable frame skipping when the processing speed cannot keep up.',
        dest='frame_skipping',
        action='store_true',
    )
    parser.add_argument(
        '--listening',
        help='Enable pulling listener.',
        dest='listening',
        action='store_true',
    )
    parser.add_argument(
        '--no-detecting',
        help='Disable pedestrian detecting (and all other processes)',
        dest='no_detecting',
        action='store_true',
    )
    parser.add_argument(
        '--no-tracking',
        help='Disable path tracking',
        dest='no_tracking',
        action='store_true',
    )
    parser.add_argument(
        '--no-clustering',
        help='Disable pedestrian clustering (group detection)',
        dest='no_clustering',
        action='store_true',
    )
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    return parser.parse_args()
