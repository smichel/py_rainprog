import numpy as np

def testangles(shiftX, shiftY, status):

    meanXex = np.empty([len(shiftX)])
    meanYex = np.empty([len(shiftY)])

    for i in range(len(shiftX)):
        meanXex[i] = np.mean(np.delete(shiftX,i))
        meanYex[i] = np.mean(np.delete(shiftY,i))

    shiftXex = shiftX[(np.sign(meanXex) == np.sign(shiftX)) & (np.sign(meanYex) == np.sign(shiftY))]
    shiftYex = shiftY[(np.sign(meanXex) == np.sign(shiftX)) & (np.sign(meanYex) == np.sign(shiftY))]

    return shiftXex, shiftYex