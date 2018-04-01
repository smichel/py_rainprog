import numpy as np
from init import Square

def testangles(fields, status, res):

    shiftX = np.empty([len(fields)])
    shiftY = np.empty([len(fields)])
    status = np.arange(len(fields))

    for i in range(len(fields)):
        shiftX[i] = fields[i].shiftX
        shiftY[i] = fields[i].shiftY

    lengths = np.sqrt(np.square(shiftX) + np.square(shiftY)) * res

    shiftXex = shiftX[lengths <= 800]
    shiftYex = shiftY[lengths <= 800]
    status = status[lengths <= 800]

    meanXex = np.empty([len(shiftXex)])
    meanYex = np.empty([len(shiftYex)])

    for i in range(len(shiftXex)):
        meanXex[i] = np.mean(np.delete(shiftXex,i))
        meanYex[i] = np.mean(np.delete(shiftYex,i))

    shiftX = shiftXex[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]
    shiftY = shiftYex[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]
    status = status[(np.sign(meanXex) == np.sign(shiftXex)) | (np.sign(meanYex) == np.sign(shiftYex))]

    return shiftX, shiftY, status
