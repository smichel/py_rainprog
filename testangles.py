import numpy as np
import math
def testangles(shiftX, shiftY, status, res):

    lengths = np.sqrt(np.square(shiftX) + np.square(shiftY)) * res

    shiftXex = shiftX[lengths <= 800]
    shiftYex = shiftY[lengths <= 800]

    meanXex = np.empty([len(shiftXex)])
    meanYex = np.empty([len(shiftYex)])

    for i in range(len(shiftXex)):
        meanXex[i] = np.mean(np.delete(shiftXex,i))
        meanYex[i] = np.mean(np.delete(shiftYex,i))

    shiftX = shiftXex[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]
    shiftY = shiftYex[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]

    return shiftX, shiftY
