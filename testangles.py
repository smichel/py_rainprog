import numpy as np
import math
def testangles(shiftX, shiftY, status):

    lengths = np.sqrt(np.square(shiftX) + np.square(shiftY))
    lengthsEx = np.empty([len(lengths)])
    for i in range(len(lengths)):
        lengthsEx[i] = np.mean(np.delete(lengths, i))

    shiftXex = shiftX[(lengths * 2 <= lengthsEx) | (lengths * 0.5 >= lengthsEx)]
    shiftYex = shiftY[(lengths * 2 <= lengthsEx) | (lengths * 0.5 >= lengthsEx)]

    meanXex = np.empty([len(shiftXex)])
    meanYex = np.empty([len(shiftYex)])

    for i in range(len(shiftXex)):
        meanXex[i] = np.mean(np.delete(shiftXex,i))
        meanYex[i] = np.mean(np.delete(shiftYex,i))

    shiftX = shiftXex[(np.sign(meanXex) == np.sign(shiftXex)) & (np.sign(meanYex) == np.sign(shiftYex))]
    shiftY = shiftYex[(np.sign(meanXex) == np.sign(shiftXex)) & (np.sign(meanYex) == np.sign(shiftYex))]

    return shiftX, shiftY
