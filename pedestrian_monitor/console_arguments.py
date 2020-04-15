import argparse
from pathlib import Path
from os.path import basename, isfile, isdir, splitext
import glob


def _list_modules(folder_name):
    modules = glob.glob(str(Path(__file__).resolve().parent / folder_name / '*'))
    return list(
        splitext(basename(f))[0]
        for f in modules
        if (isfile(f) and splitext(f)[1] == '.py' and basename(f) != '__init__.py')
        or (isdir(f) and isfile(Path(f) / '__init__.py'))
    )


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
        '-pp',
        '--preprocessor',
        help='Use a specific preprocessor',
        dest='preprocessor',
        choices=_list_modules('preprocessors'),
        default='sample_preprocessor',
    )
    parser.add_argument(
        '--no-preprocessor',
        help='Disable image preprocessing',
        dest='no_preprocessor',
        action='store_true',
    )
    parser.add_argument(
        '-dt',
        '--detector',
        help='Use a specific detector',
        dest='detector',
        choices=_list_modules('detectors'),
        default='sample_detector',
    )
    parser.add_argument(
        '--no-detector',
        help='Disable pedestrian detecting',
        dest='no_detector',
        action='store_true',
    )
    parser.add_argument(
        '-tk',
        '--tracker',
        help='Use a specific tracker',
        dest='tracker',
        choices=_list_modules('trackers'),
        default='sample_tracker',
    )
    parser.add_argument(
        '--no-tracker',
        help='Disable pedestrian re-id and path tracking',
        dest='no_tracker',
        action='store_true',
    )
    parser.add_argument(
        '-cl',
        '--clusterer',
        help='Use a specific clusterer',
        dest='clusterer',
        choices=_list_modules('clusterers'),
        default='sample_clusterer',
    )
    parser.add_argument(
        '--no-clusterer',
        help='Disable pedestrian clustering (group detection)',
        dest='no_clusterer',
        action='store_true',
    )
    # if len(sys.argv) == 1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    return parser.parse_args()
