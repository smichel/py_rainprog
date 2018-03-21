import numpy as np

def createblob(d_s, res, timeSteps):
    gaussBlob = np.zeros([timeSteps, d_s, d_s])
    x = np.arange(d_s)
    y = np.arange(d_s)
    [x, y] = np.meshgrid(x, y)
    u = 1
    v = 2
    x0 = 50
    x1 = 100
    y0 = 0
    amp = 10
    sigma = 10

    def f(x, x0, y, y0, amp, sigma):
        return amp*np.exp(-(np.square(x-x0)/(2*sigma**2)+(np.square(y-y0)/(2*sigma**2))))

    for t in range(timeSteps):
        gaussBlob[t, :, :] = f(x, x0, y, y0, amp, sigma)+f(x, x1, y, y0, amp*0.8, sigma)
        x0 = x0 + u
        x1 = x1 + u
        y0 = y0 + v

    return gaussBlob