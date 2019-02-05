import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# sklearn準拠モデル
# 回帰なのでRegressorMixinを継承
# テキストのLassoではなく以下を用いた
# min { (1/2N) ||y - Xbeta||^2_2 + lambda||theta||_1 }
# such that beta - theta

class Admm(BaseEstimator, RegressorMixin):
    def __init__(self, lambda_=1.0, rho=1.0, max_iter=1000):
        self.lambda_ = lambda_
        self.rho = rho
        self.threshold = lambda_ / rho # しきい値
        self.max_iter = max_iter
        self.coef_ = None # 偏回帰係数
        self.intercept_ = 0.0 # 切片

    def _soft_thresholding_func(self, x: np.ndarray) -> np.ndarray: # 軟判定しきい値関数
        t = self.threshold

        # list of bool
        positive_index = x >= t
        negative_index = x <= t
        zero_index = abs(x) <= t

        y = np.zeros(x.shape)

        y[positive_index] = x[positive_index] - t
        y[negative_index] = x[negative_index] + t
        y[zero_index] = 0.0

        return y

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X: 計画行列
        # y: 観測ベクトル
        # beta: 回帰係数ベクトル

        N, M = X.shape

        inverse_mat = np.linalg.inv(np.dot(X.T, X) / N + self.rho * np.identity(M))
        # 初期化
        beta = np.dot(X.T, y) / N
        theta = beta.copy()
        mu = np.zeros(len(beta))

        for _ in range(self.max_iter):
           beta = np.dot(inverse_mat
                         , np.dot(X.T, y) / N + self.rho * theta - mu)
           theta = self._soft_thresholding_func(beta + mu / self.rho)
           mu += self.rho * (beta - theta)

        self.coef_ = beta

        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        return y
