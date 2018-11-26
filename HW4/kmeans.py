import numpy as np

class KMeans():
    def __init__(self, data, centers):
        self.data = data
        self.centers = centers
        

    def update_centers(self):
        pass

    def assign(self):
        distances = (self.data - self.centers)**2
        return distances.max(dim=1)

    def train(self):

