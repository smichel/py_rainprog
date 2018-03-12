import numpy as np

def createblob(u, v, x0, y0, d_s, res, amp, sigma, timeSteps):
    gaussBlob = np.zeros([timeSteps, d_s, d_s])
    x = np.arange(d_s)
    y = np.arange(d_s)
    [x, y] = np.meshgrid(x, y)

    def f(x, x0, y, y0, amp, sigma):
        return amp*np.exp(-(np.square(x-x0)/(2*sigma**2)+(np.square(y-y0)/(2*sigma**2))))



    for t in range(timeSteps):
        gaussBlob[t, :, :] = f(x, x0, y, y0, amp, sigma)
        x0 = x0 + u
        y0 = y0 + v

    return gaussBlob