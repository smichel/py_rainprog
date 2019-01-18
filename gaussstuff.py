bins = np.linspace(-800/res,800/res, num=9)
shiftX = np.hstack(shiftXlist[prog - trainTime:prog])

histX, bin_edges = np.histogram(np.hstack(shiftXlist[prog - trainTime:prog]), bins)
histY, _ = np.histogram(np.hstack(shiftYlist[prog - trainTime:prog]), bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
new_bin_centres = np.linspace(bin_centres[0], bin_centres[-1], 200)
# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

coeffX, var_matrix = curve_fit(gauss, bin_centres, histX, p0=p0, maxfev=2000)
coeffY, var_matrix = curve_fit(gauss, bin_centres, histY, p0=p0, maxfev=2000)
# Get the fitted curve
histX_fit = gauss(new_bin_centres, *coeffX)
histY_fit = gauss(new_bin_centres, *coeffY)

fig = plt.subplots()
plt.plot(bin_centres, histX, label='Data')
plt.plot(new_bin_centres, histX_fit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted mean = ', coeffX[1])
print('Fitted standard deviation = ', coeffX[2])

plt.show(block=False)

fig = plt.subplots()
plt.plot(bin_centres, histY, label='Data')
plt.plot(new_bin_centres, histY_fit, label='Fitted data')

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Fitted mean = ', coeffY[1])
print('Fitted standard deviation = ', coeffY[2])

plt.show(block=False)

def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/2.*sigma**2)


def KL(mu,epsilon,m,S,rho,r):
    return 1 / (2 * (1 - r ** 2)) * ((mu[0] - m[0]) ** 2 / S[0, 0] ** 2 - 2 * r * (
            ((mu[0] - m[0]) * (mu[1] - m[1])) / (S[0, 0] * S[1, 1])) + (mu[1] - m[1]) ** 2 / S[1, 1] ** 2) + 1 / (
                       2 * (1 - r ** 2)) * ((epsilon[0, 0] ** 2 - S[0, 0] ** 2) / S[0, 0] ** 2 - 2 * r * (
            rho * epsilon[0, 0] * epsilon[1, 1] - r * S[0, 0] * S[1, 1]) / (S[0, 0] * S[1, 1]) + (
                                                    epsilon[1, 1] ** 2 - S[1, 1] ** 2) / S[1, 1] ** 2) + np.log(
        (S[0, 0] * S[1, 1] * np.sqrt(1 - r ** 2)) / (epsilon[0, 0] * epsilon[1, 1] * np.sqrt(1 - rho ** 2)))

