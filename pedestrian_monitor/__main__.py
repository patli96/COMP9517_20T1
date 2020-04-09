import sys

if not __package__ and not hasattr(sys, 'frozen'):
    # direct call of __main__.py
    import os.path

    path = os.path.realpath(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(path)))

import pedestrian_monitor  # noqa

if __name__ == '__main__':
    pedestrian_monitor.main()
