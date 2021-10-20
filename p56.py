from p53 import *

def denoisedCorr(eVal, eVec,nFacts):
    eVal_ = np.diag(eVal).copy()
    eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0]-nFacts)
    eVal_ = np.diag(eVal_)
    corr1 = np.dot(eVec, eVal_).dot(eVec.T)
    corr1 = cov2corr(corr1)
    return corr1

corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
eVal1, eVec1 = getPCA(corr1)
plt.plot(np.diag(eVal0))
plt.plot(np.diag(eVal1))
plt.show()

def denoisedCorr2(eVal, eVec, nFacts, alpha=0.):
    eValL, eVecL = eVal[:nFacts, :nFacts], eVec[:, :nFacts]
    eValR, eVecR = eVal[nFacts:, nFacts:], eVec[:, nFacts:]
    corr0 = np.dot(eVecL, eValL).dot(eVecL.T)
    corr1 = np.dot(eVecR, eValR).dot(eVecR.T)
    corr2 = corr0 + alpha * corr1 + (1 - alpha) * np.diag(np.diag(corr1))
    return corr2

corr1 = denoisedCorr2(eVal0, eVec0, nFacts0, 0.5)
eVal1, eVec1 = getPCA(corr1)

plt.plot(np.diag(eVal0))
plt.plot(np.diag(eVal1))
plt.show()