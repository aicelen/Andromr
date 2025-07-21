#custom integration of the agglomerative clustering without sklearn because Android doesn't work with scipy (10% Fortran)
#porbably not efficent; but is only active for short periods of time

import numpy as np
from math import dist #distance


class AgglomerativeClustering():
    def __init__(self, data, distance_threshold):
        self.data = data
        self.distance_threshold = distance_threshold
        self.distance_map = self.generate_dist_map(data)
        self.label_map = np.zeros((len(data)), #size
                            dtype = int #datatype
                            )

    def generate_dist_map(self, data):
        distances = np.zeros((len(data),len(data)), #size
                            dtype = float #datatype
                            )
        
        for x in range(len(data)):
            for y in range(len(data)):
                if not x==y:
                    distances[x][y] = dist(data[x], data[y])
                else:
                    distances[x][y] = np.inf

        return(distances)
    

    def update_dist_map(self, idx_x, idx_y):
        self.distance_map[idx_x][idx_y] = np.inf
        


    def min_element(self):
        #get min of arrayd
        return (np.unravel_index(np.argmin(self.distance_map), self.distance_map.shape))

    def replace_zeros(self):
        cur_label = np.max(self.label_map)
        for i in range(len(self.label_map)):
            if self.label_map[i] == 0:
                cur_label += 1
                self.label_map[i] = cur_label
            

    def generate_labels(self):
        min_idx_x, min_idx_y = self.min_element()
        min_idx_x, min_idx_y = int(min_idx_x), int(min_idx_y)

        label_idx = 1
        
        while self.distance_threshold > self.distance_map[min_idx_x][min_idx_y]:
            #check if we bind to cluster
            if self.label_map[min_idx_x] == 0 and self.label_map[min_idx_y] == 0:
                #no label
                self.label_map[min_idx_x] = label_idx
                self.label_map[min_idx_y] = label_idx
                label_idx += 1 #new label

            elif self.label_map[min_idx_x] != 0 and self.label_map[min_idx_y] == 0:
                #set y to label
                self.label_map[min_idx_y] = self.label_map[min_idx_x]

            elif self.label_map[min_idx_x] == 0 and self.label_map[min_idx_y] != 0:
                self.label_map[min_idx_x] = self.label_map[min_idx_y]

            elif self.label_map[min_idx_x] != 0 and self.label_map[min_idx_y] != 0:
                # Merge clusters
                cluster_label = self.label_map[min_idx_x]
                for i in range(len(self.label_map)):
                    if self.label_map[i] == self.label_map[min_idx_y]:
                        self.label_map[i] = cluster_label

            
            self.update_dist_map(min_idx_x, min_idx_y)
            min_idx_x, min_idx_y = self.min_element()
        self.replace_zeros()
    
    def get_labels(self):
        return self.label_map