from p51 import *
from p52 import *
from scipy.optimize import minimize


def errPDFs(var, eVal, q, bWidth, pts=1000):
    var = var[0]
    pdf0 = mpPDF(var, q, pts)
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)
    sse = np.sum((pdf1 - pdf0)**2)
    return sse

def findMaxEval(eVal, q, bWidth):
    out = minimize(lambda *x:errPDFs(*x), np.array([.5]), args=(eVal, q, bWidth), bounds=((1E-5, 1-1E-5),))
    if out['success']:
        var = out['x'][0]
    else:
        var = 1
    eMax = var * (1 + (1./q)**.5)**2
    return eMax, var

eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=.01)
print(np.diag(eVal0)[::-1].searchsorted(eMax0))
print(eMax0)
nFacts0 = eVal0.shape[0] - np.diag(eVal0)[::-1].searchsorted(eMax0)