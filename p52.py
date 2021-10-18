import numpy as np
import matplotlib.pyplot as plt

def getPCA(matrix):
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]
    eVal, eVec = eVal[indices], eVec[:,indices]
    eVal = np.diagflat(eVal)
    return eVal, eVec

def getRndCov(nCols, nFacts):
    w = np.random.normal(size=(nCols, nFacts))
    cov = np.dot(w, w.T)
    cov += np.diag(np.random.uniform(size=nCols))  # full rank matrix로 변경
    return cov

def cov2corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    corr[corr<-1], corr[corr>1] = -1, 1
    return corr

alpha, nCols, nFact, q = .995, 1000, 100, 10
cov = np.cov(np.random.normal(size=(nCols*q, nCols)), rowvar=0)  # nCols, nCols 형태가 됨
cov = alpha * cov + (1 - alpha) * getRndCov(nCols, nFact)
corr0 = cov2corr(cov)
eVal0, eVec0 = getPCA(corr0)  # 상관계수에서 고유값과 고유벡터구함..