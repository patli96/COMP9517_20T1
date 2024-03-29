from typing import Dict, Tuple, List
from .sort import *
import numpy as np


def compute(  # This function will be called with named parameters, so please do not change the parameter name
        detections: List[Tuple[int, int, int, int]],  # List[ (y1, x1, y2, x2), ... ]
        detection_records: List[List[Tuple[int, int, int, int]]],  # List[ detections ], the current is [0]
        detection_frame_deltas: List[int],  # List[ detection_frame_delta ], the current is [0]
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_index: int,  # The current index of frame, started at 0
        image_records: List[np.ndarray],  # List[ images ], previously displayed images
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        previous_pedestrian_records: List[Dict[int, Tuple[int, int, int, int]]],  # List[ pedestrians ], previously computed
        previous_pedestrian_frame_deltas: List[int],  # List[ pedestrian_frame_delta ], for previously computed pedestrians
        previous_tracks: Dict[int, List[Tuple[int, int]]],  # Dict{ pedestrian_id: [(y, x), ...], ... }
        # The new tracks should be modified based on the previous_tracks
        # Both tracks and pedestrians shared the same frame_delta_records as they both came from trackers
         storage: Dict[str, List],  # It will be handed over to the next detector, please mutate this object directly
) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[int, List[Tuple[int, int]]]]:
    # define and update a global variable to store frame indexes have been tracked
    if len(previous_pedestrian_records) <= 1:
        storage['pedestrian_frame_indexes'] = [image_index]
        storage['tracked_id_list'] = []  # List[ List[frame_index, id1, id2, ...], ... ]
        global mot_tracker
        mot_tracker = Sort()
        print('-------------\nFIRST RUN\n--------------')
    else:
        storage['pedestrian_frame_indexes'].append(image_index)

    # update previous_pedestrian_frame_deltas
    previous_pedestrian_frame_deltas.insert(0, frame_delta)

    # initialize pedestrians and tracks
    pedestrians = {}
    tracks = {}

    # convert the coordinates from [y1, x1, y2, x2] to [x1, y1, x2, y2, 0]
    # the last column will be replaced by pedestrian_id
    # keep all converted coordinates in dets
    dets = []
    for bbox in detections:
        dets.append([bbox[1], bbox[0], bbox[3], bbox[2], 0])
    dets = np.array(dets)
    trackers = mot_tracker.update(dets)  # trackers: List[ List[x1, y1, x2, y2, pedestrian_id], ... ]
    id_list = [image_index]
    for bbox in trackers:
        ped_id = int(bbox[-1])
        id_list.append(ped_id)
        pedestrians[ped_id] = (int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2]))
        center_x = int(0.5 * (bbox[0] + bbox[2]))
        center_y = int(0.5 * (bbox[1] + bbox[3]))
        if ped_id in previous_tracks:
            trajectory = previous_tracks[ped_id]
            trajectory.insert(0, (center_y, center_x))
            tracks[ped_id] = trajectory
        else:
            tracks[ped_id] = [(center_y, center_x)]

    '''If no frame skipped'''
    if len(storage['tracked_id_list']) == len(storage['pedestrian_frame_indexes']):

        # update previous_pedestrian_records, previous_tracks, storage['tracked_id_list']
        previous_pedestrian_records.insert(0, pedestrians)
        for ped_id in tracks:
            previous_tracks[ped_id] = tracks[ped_id]
        storage['tracked_id_list'].append(id_list)

    # If detections frame runs faster then pedestrian frame
    elif len(storage['tracked_id_list']) > len(storage['pedestrian_frame_indexes']):

        # If same numbers of pedestrians detected as last time tracked
        if len(trackers) == len(storage['tracked_id_list'][-1][1:]):

            # Record all IDs not being detected this time
            unassigned = []  # List[(y1, x1), (y2, x2), ...]
            new_ids = []  # List[id1, id2, ...]
            last_time_tracked = storage['tracked_id_list'][-1][1:]

            for ped_id in tracks:
                if ped_id in last_time_tracked:
                    tracks[ped_id].insert(0, 'assigned')
                else:
                    unassigned.append(tracks[ped_id][-1])
                    new_ids.append(ped_id)

            frameskip = image_index - storage['pedestrian_frame_indexes'][-1]

            # Assign the rest IDs from last time to closest centers
            # calculate speed based on trajectory already calculated
            # estimate possible locations
            for ped_id in last_time_tracked:
                if tracks[ped_id][0] != 'assigned':
                    moved_distance = 0
                    for i in range(len(previous_tracks[ped_id])):
                        if i == 0:
                            moved_distance += np.sqrt(previous_tracks[ped_id][i][0] ** 2 + 
                                                      previous_tracks[ped_id][i][1] ** 2)
                        else:
                            moved_distance += np.sqrt(
                                (previous_tracks[ped_id][i][0] - previous_tracks[ped_id][i-1][0]) ** 2 + 
                                (previous_tracks[ped_id][i][1] - previous_tracks[ped_id][i-1][1]) ** 2)
                    speed_per_frame = moved_distance / len(previous_tracks[ped_id])
                    estimated_range = speed_per_frame * frameskip
                    dist_list = [np.abs(np.sqrt(np.abs(tracks[ped_id][-1][0] - loc[0])**2
                                                + np.abs(tracks[ped_id][-1][1] - loc[1])**2)
                                        - estimated_range)
                                 for loc in unassigned]
                    min_index = dist_list.index(min(dist_list))
                    # if unassigned IDs are far away from existing ones then don't assign them
                    if dist_list[min_index] < 0.5 * estimated_range:
                        tracks[ped_id].insert(0, unassigned[min_index])
                        tracks[ped_id].inset(0, 'assigned')
                        for line in trackers:
                            if line[-1] == new_ids[min_index]:
                                pedestrians[ped_id] = (line[1], line[0], line[3], line[2])
                        unassigned.pop(min_index)
                        for i in range(len(mot_tracker.trackers)):
                            if mot_tracker.trackers[i].id == new_ids[min_index]-1:
                                mot_tracker.pop(i)
                                break
                        KalmanBoxTracker.count -= 1
                        new_ids[min_index] = None

            # Delete mis-assigned new IDs
            for ped_id in new_ids:
                if ped_id is not None:
                    del(tracks[ped_id])
                    del(pedestrians[ped_id])

            # assign the new ID to those remain unassigned coordinates
            if len(unassigned) != 0:
                for ped_id in tracks:
                    if tracks[ped_id][0] != 'assigned':
                        tracks[ped_id].pop(0)

            new_dets = []
            for ped_id in new_ids:
                for bbox in trackers:
                    if bbox[-1] == ped_id:
                        t = (bbox[1], bbox[0], bbox[3], bbox[2])
                        for det in detections:
                            if det == t:
                                new_dets.append(bbox)
            new_trackers = mot_tracker.update(new_dets)  # trackers: List[ List[x1, y1, x2, y2, pedestrian_id], ... ]
            for bbox in new_trackers:
                for i in range(len(bbox)):
                    bbox[i] = round(bbox[i])
            id_list = [image_index]
            for ped_id in tracks:
                id_list.append(ped_id)
            for bbox in new_trackers:
                ped_id = bbox[-1]
                id_list.append(ped_id)
                pedestrians[ped_id] = (bbox[1], bbox[0], bbox[3], bbox[2])
                center_x = round(0.5 * (bbox[0] + bbox[2]))
                center_y = round(0.5 * (bbox[1] + bbox[3]))
                tracks[ped_id] = [(center_y, center_x)]
            storage['tracked_id_list'].append(id_list)

    # If pedestrians frame runs faster than detections
    else:
        # Take it as all pedestrians are keeping stationary
        # update previous_pedestrian_records, previous_tracks, storage['tracked_id_list']
        previous_pedestrian_records.insert(0, pedestrians)
        for ped_id in tracks:
            previous_tracks[ped_id] = tracks[ped_id]
        storage['tracked_id_list'].append(id_list)
    # pedestrians is a dict
    # its indexes are pedestrian ids, which should be stable between frames
    # the pedestrian id does not need to start at 0, but it needs to be unique and stable
    # Example: 0: (2, 4, 6, 8) -> 0: (3, 6, 9, 11), the index 0 means the same person moves
    # its values are tuples that define the bounding box of this pedestrian, and the format is (y1, x1, y2, x2)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    # pedestrians = {
      #   0: (2, 4, 80, 100),
    # }
    # tracks is a dict
    # its indexes are pedestrian ids, be aware that only currently detected pedestrians can have their tracks.
    # which means set(list(tracks.keys())) == set(list(pedestrians.keys()))
    # its values are lists that contain tuple of points, their format is (y, x)
    # please be aware that it is height-first, shape-like order instead of OpenCV's width-first order
    # tracks = {
      #   0: [(3, 3), (50, 50)]
    # }
    return pedestrians, tracks
