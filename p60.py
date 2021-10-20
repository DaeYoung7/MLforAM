from p56 import *
from scipy.linalg import block_diag

def formBlockMatrix(nBlocks, bSize, bCorr):
    block = np.ones((bSize, bSize)) * bCorr
    block[range(bSize), range(bSize)] = 1
    corr = block_diag(*([block] * nBlocks))
    return corr

def corr2cov(corr, std):
    cov = corr * np.outer(std, std)
    return cov

def formTrueMatrix(nBlocks, bSize, bCorr):
    corr0 = formBlockMatrix(nBlocks, bSize, bCorr)
    corr0 = pd.DataFrame(corr0)
    cols = corr0.columns.tolist()
    np.random.shuffle(cols)
    # 열 -> 행 총 두번 섞는 효과
    corr0 = corr0[cols].loc[cols].copy(deep=True)
    std0 = np.random.uniform(.05, .2, corr0.shape[0])
    cov0 = corr2cov(corr0, std0)
    # 평균과 분산이 std0인 분포에서 랜덤 추출
    mu0 = np.random.normal(std0, std0, cov0.shape[0])
    return mu0, cov0

nBlocks, bSize, bCorr = 10, 50, 0.5
np.random.seed(0)
mu0, cov0 = formTrueMatrix(nBlocks, bSize, bCorr)