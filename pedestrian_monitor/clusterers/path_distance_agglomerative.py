from typing import Dict, Tuple, List, Any

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

def compute(  # This function will be called with named parameters, so please do not change the parameter name
        pedestrians: Dict[int, Tuple[int, int, int, int]],  # Dict{ pedestrian_id: (y1, x1, y2, x2), ... }
        pedestrian_records: List[Dict[int, Tuple[int, int, int, int]]],  # List[ pedestrians ], the current is [0]
        pedestrian_frame_deltas: List[int],  # List[ pedestrian_frame_delta ], the current is [0]
        tracks: Dict[int, List[Tuple[int, int]]],  # Dict{ pedestrian_id: [(y, x), ...], ... }
        # There is no track_records as each (y, x) is a record, and tracks have all pedestrians' tracks
        # Tracks of those who disappeared or moved out of the image will be removed, and they're useless
        # Both tracks and pedestrians shared the same frame_delta_records as they both came from trackers
        image: np.ndarray,  # The image, it is 3-channel BGR uint8 numpy array
        image_index: int,  # The current index of frame, started at 0
        image_records: List[np.ndarray],  # List[ images ], previously displayed images
        frame_delta: int,  # current_frame_index - last_computed_frame_index, will be >= 1
        previous_group_records: List[Dict[int, List[int]]],  # List[ groups ], previously computed groups
        previous_group_frame_deltas: List[int],  # List[ frame_delta ], for previously computed groups
        storage: Dict[str, Any],  # It will be handed over to the next detector, please mutate this object directly
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:

    depth = 3
    X = []
    pedestrian_ids = []
    if len(tracks) < 1:
        return {}, {}, {}
    for ped, track in tracks.items():
        if len(track) >= depth:
            pedestrian_ids.append(ped)
            x = []
            for p in track[:depth]:
                for i in range(len(p)):
                    x.append(p[i] / image.shape[i])
            X.append(x)

    X = np.array(X)

    if len(X) < 1:
        return {}, {}, {}

    db = AgglomerativeClustering(distance_threshold=0.1, compute_full_tree=True, n_clusters=None).fit(X)

    ax.clear()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.scatter(X[:, 0], X[:, 1], c=db.labels_)

    groups = {}
    for idx, cl in enumerate(db.labels_):
        if cl in groups:
            groups[cl].append(pedestrian_ids[idx])
        else:
            groups[cl] = [pedestrian_ids[idx]]

    single_groups = []
    for g in groups:
        if len(groups[g]) < 2:
            single_groups.append(g)

    for g in single_groups:
        del groups[g]

    # if len(groups.items()) > 0:
    #     print(groups)

    # groups is a dict
    # its indexes are group ids, which should be stable between frames
    # the group id does not need to start at 0, but it needs to be unique and stable
    # Example: 0: [0, 1] -> 0: [0, 1, 8] -> 0: [1, 8]
    # its values are lists that contain >=2 unique pedestrian id
    # Example: [1] is not a group and should not be returned

    # groups = {
    #     0: [0],
    #     1: [2, 4],
    #     2: [5, 3],
    # }
    # entering_members is a dict
    # it has the same structure as groups
    # the pedestrian_id inside will be marked as entering the group
    # due to the lazy removal, the pedestrian_ids inside may appear in groups or/and leaving_members
    # entering_members = {
    #     0: [1],
    #     1: [7],
    #     2: [9],
    # }
    # leaving_members is a dict
    # it has the same structure as groups
    # the pedestrian_id inside will be marked as leaving the group
    # due to the lazy removal, the pedestrian_ids inside may appear in groups or/and entering_members
    # leaving_members = {
    #     0: [1],
    #     1: [6],
    #     2: [8, 3],
    # }
    return groups, {}, {}
