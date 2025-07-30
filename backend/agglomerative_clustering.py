import numpy as np
import heapq
from pykdtree.kdtree import KDTree


class AgglomerativeClustering():
    """
    Implementation of Agglomerative Clustering using KDTree and heapq for better performance.
    Used because Scipy doesn't work on Android.

    Parameters
    -------
    data : np.ndarray
        Points in a 2D space. Array has shape (x, 2).
    distance_threshold : int
        Maximum distance between two points of the same group.

    Methods
    -------
    run() -> np.ndarray
        Run the algorithm. 
        Returns labels.
    get_labels() -> np.ndarray
        Get cluster labels.
        Returns labels.
    """

    def __init__(self, data, distance_threshold):
        self.data = data # np array num_elements, 2
        self.distance_threshold = distance_threshold
        self._fill_heapq()
        self.result = np.zeros(len(data), dtype=int)
        self.cur_group_idx = 0

    def _fill_heapq(self):
        """
        Fill the list with all relevant distances
        """

        distances, indices = KDTree(self.data).query(self.data, k=len(self.data))

        # Create a mask for all the points that are close enough to another one
        rows, cols = np.where((distances <= self.distance_threshold) & (distances > 0))

        # Remoce duplicates
        mask = rows < indices[rows, cols]

        # Creates tuples of (distance, point1_index, point2_index)
        self.dist_heapq = list(zip(distances[rows[mask], cols[mask]], rows[mask], indices[rows[mask], cols[mask]]))
        
        # Convert to heapq for better performance in future steps
        heapq.heapify(self.dist_heapq)
    
    def _add_idx_to_data(self, x, y):
        """
        Add the group_id to two points
        """

        group_idx = self.cur_group_idx
        if self.result[x] != 0 and self.result[y] != 0:
            # connect both groups
            self.result[self.result == self.result[y]] = self.result[x]
            return
        elif self.result[x] != 0:
            group_idx = self.result[x]
        elif self.result[y] != 0:
            group_idx = self.result[y]
           
        self.result[x] = group_idx
        self.result[y] = group_idx
        self.cur_group_idx += 1

    def _remove_zeros(self):
        """
        Changes zeros to a unique leabel from self.result
        """
        cur_label = np.max(self.result)
        for i in range(len(self.result)):
            if self.result[i] == 0:
                cur_label += 1
                self.result[i] = cur_label


    def _clean_labels_up(self):
        # Merging
        unique_labels, inverse = np.unique(self.result, return_inverse=True)
        self.result = inverse.reshape(self.result.shape)
                
    def run(self):
        """
        Run the algorithm. 
        Returns labels (np.ndarray).
        """
        while self.dist_heapq:
            self._add_idx_to_data(self.dist_heapq[0][1], self.dist_heapq[0][2]) # Grab from the first element the x and y 
            heapq.heappop(self.dist_heapq)


        self._remove_zeros()
        self._clean_labels_up()

        return self.result
    
    def get_labels(self):
        """
        Get cluster labels.
        Returns labels (np.ndarray).
        """
        return self.result