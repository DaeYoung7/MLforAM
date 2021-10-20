from p60 import *
from sklearn.covariance import LedoitWolf

def simCovMu(mu0, cov0, nObs, shrink=False):
    x = np.random.multivariate_normal(mu0.flatten(), cov0, size=nObs)
    mu1 = x.mean(axis=0).reshape(-1, 1)
    if shrink:
        cov1 = LedoitWolf().fit(x).covariance_
    else:
        cov1 = np.cov(x, rowvar=0)
    return mu1, cov1

def deNoiseCov(cov0, q, bWidth):
    corr0 = cov2corr(cov0)
    eVal0, eVec0 = getPCA(corr0)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth)
    nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)
    corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    cov1 =corr2cov(corr1, np.diag(cov0)**.5)
    return cov1

def optPort(cov, mu=None):
    inv = np.linalg.inv(cov)
    ones = np.ones(shape=(inv.shape[0], 1))
    if mu is None:
        mu = ones
    w = np.dot(inv, mu)
    w /= np.dot(ones.T, w)
    return w

nObs, nTrials, bWidth, shrink, minVarPortf = 1000, 1000, .01, False, True
w1 = pd.DataFrame(columns=range(cov.shape[0]), index=range(nTrials), dtype=float)
w1_d = w1.copy(deep=True)
np.random.seed(0)
for i in range(1):
    mu1, cov1 = simCovMu(mu0, cov0, nObs, shrink=shrink)
    if minVarPortf:
        mu1 = None
    cov1_d = deNoiseCov(cov1, nObs * 1. / cov1.shape[1], bWidth)
    w1.loc[i] = optPort(cov1, mu1).flatten()
    w1_d.loc[i] = optPort(cov1_d, mu1).flatten()