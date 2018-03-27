class Square:

    def __init__(self, cRange, maxima, status, rainThreshold, distThreshold, dist, id):
        self.cRange = cRange  # cRange
        self.maxima = maxima  # center point of the square and its rainfall intensity
        self.status = status  # status of the maximum
        self.rainThreshold = rainThreshold  # minimal rainthreshold
        self.distThreshold = distThreshold  # distance threshold to radarboundary
        self.dist = dist  # global distancefield
        self.id = id  # square identifier
        self.histX = []  # history of X displacements
        self.histY = []  # history of Y displacements
        self.histMaxima = []  # history of maxima locations
