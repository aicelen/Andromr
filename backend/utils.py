from typing import Tuple

import cv2
import numpy as np

from . import layers

from backend.staffline_extraction import Staff

def count(data, intervals):
    """Count elements in different intervals"""
    occur = []
    data = np.sort(data)
    intervals = np.insert(intervals, [0, len(intervals)], [np.min(data), np.max(data)])
    for idx in range(len(intervals[:-1])):
        sub = data[data>=intervals[idx]]
        sub = sub[sub<intervals[idx+1]]
        occur.append(len(sub))
    return occur


def find_closest_staffs(x: int, y: int) -> Tuple[Staff, Staff]:
    staffs = layers.get_layer('staffs')

    staffs = staffs.reshape(-1, 1).squeeze()
    diffs = sorted(staffs, key=lambda st: st - [x, y])
    if len(diffs) == 1:
        return diffs[0], diffs[0]
    elif len(diffs) == 2:
        return (diffs[0], diffs[1])

    # There are over three candidates
    first = diffs[0]
    second = diffs[1]
    third = diffs[2]
    if abs(first.y_lower - y) <= abs(first.y_upper - y):
        # Closer to the lower bound of the first candidate.
        if second.y_center > first.y_center:
            return first, second
        elif third.y_center > first.y_center:
            return first, third
        else:
            return first, first
    else:
        # Closer to the upper bound of the first candidate.
        if second.y_center < first.y_center:
            return first, second
        elif third.y_center < first.y_center:
            return first, third
        else:
            return first, first


def get_unit_size(x: int, y: int) -> float:
    st1, st2 = find_closest_staffs(x, y)
    if st1.y_center == st2.y_center:
        return float(st1.unit_size)

    # Within the stafflines
    if st1.y_upper <= y <= st1.y_lower:
        return float(st1.unit_size)

    # Outside stafflines.
    # Infer the unit size by linear interpolation.
    dist1 = abs(y - st1.y_center)
    dist2 = abs(y - st2.y_center)
    w1 = dist1 / (dist1 + dist2)
    w2 = dist2 / (dist1 + dist2)
    unit_size = w1 * st1.unit_size + w2 * st2.unit_size
    return float(unit_size)


def get_global_unit_size() -> float:
    staffs = layers.get_layer('staffs')
    usize = []
    for st in staffs.reshape(-1, 1).squeeze():
        usize.append(st.unit_size)
    return sum(usize) / len(usize)


def get_total_track_nums() -> int:
    staffs = layers.get_layer('staffs')
    tracks = [st.track for st in staffs.reshape(-1, 1).squeeze()]
    return len(set(tracks))


def remove_stems(data):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.dilate(cv2.erode(data.astype(np.uint8), ker), ker)


def slope_to_degree(y_diff: int, x_diff: int) -> float:
    return np.rad2deg(np.arctan2(y_diff, x_diff))
