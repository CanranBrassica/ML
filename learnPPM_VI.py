#! /usr/bin/python3
# -*- coding: utf-8 -*-

# 一次元ポアソン混合モデルの変分推論による学習

from matplotlib import pyplot as plt
from kyabepy.ml.pd import *


class PMM_Params:
    def __init__(self, K, lam, pi, max, min):
        self.K = K  # クラスター数
        self.lam = lam  # 各クラスタに対する観測モデルのパラメタの配列
        self.pi = pi  # 各クラスタの混合比率（配列）
        self.max = max  # データのおおよその最大値
        self.min = min  # データのおおよその最大値


def test_learnPPM_VI_1D():
    K = 2

    lam = rd.rand(K)
    lam_scale = 100
    lam *= lam_scale

    pi = rd.rand(K)
    pi /= np.sum(pi)

    pmmp = PMM_Params(K, lam, pi, max=lam_scale, min=0)
    N = 1000
    x, s = createData(pmmp, N)
    pmmp.max = x.max()
    pmmp.min = x.min()

    q_s, q_lam, q_pi = learn_VI(max_iter=100, x=x, pmmp=pmmp)
    print('real_lam = ' + str(lam))

    real =[[] ,[]]
    for n in range(N):
        real[s[n].argmax()].append(x[n])

    bins = 20

    weight = [[],[]]
    for n in range(N):
        for k in range(K):
            weight[k].append(q_s[n].ex[k])

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4),sharey=True)
    axR.hist(x, bins=bins, weights=weight[0], alpha=0.5, range=(x.min(),x.max()))
    axR.hist(x, bins=bins, weights=weight[1], alpha=0.5, range=(x.min(),x.max()))
    axR.set_title('learn')
    axL.hist(real[0], alpha=0.5, bins=bins, range=(x.min(),x.max()))
    axL.hist(real[1], alpha=0.5, bins=bins, range=(x.min(),x.max()))
    axL.set_title('real')

    fig.show()
    plt.show()


def createData(pmmp, N, D=1):
    data = np.zeros(N)
    s = np.zeros((N, pmmp.K))
    for n in range(N):
        k = rd.choice(np.array(range(pmmp.K)), p=pmmp.pi)
        data[n] = rd.poisson(pmmp.lam[k])
        s[n][k] = 1
    return data, s


def learn_VI(max_iter, x, pmmp):
    N = len(x)
    K = pmmp.K
    # q(lam), q(pi)を初期化
    # q_lamの初期化はshapeにpmmp.scaleをかけると良い
    q_lam = np.array([Gamma(shape=(pmmp.max - pmmp.min) * rd.random() + pmmp.min, scale=1) for k in range(K)])
    q_pi = Dirichlet(np.array([1.0 / K for k in range(K)]))

    for i in range(max_iter):
        # q(\vec{S})を更新
        q_s = np.array(
            [
                Categorical(
                    dist=np.array([
                            math.exp(x[n] * q_lam[k].ex_ln - q_lam[k].ex + q_pi.ex[k])
                        for k in range(K)])
                        / sum([
                            math.exp(x[n] * q_lam[k].ex_ln - q_lam[k].ex + q_pi.ex[k])
                        for k in range(K)])
                )
            for n in range(N)]
        )
        # q(\vec{lam})を更新
        q_lam = np.array(
            [
                Gamma(shape=q_lam[k].shape + sum([q_s[n].dist[k] * x[n] for n in range(N)]),
                      scale=q_lam[k].scale + sum([q_s[n].dist[k] for n in range(N)]))
             for k in range(K)]
        )
        # q(pi)を更新
        q_pi = Dirichlet(
            alpha=np.array(
                [
                    q_pi.alpha[k] + sum([q_s[n].dist[k] for n in range(N)])
                 for k in range(K)]
            )
        )

        print('<lam> = ' + str([q_lam[k].ex for k in range(K)]))

    return q_s, q_lam, q_pi


if __name__ == '__main__':
    test_learnPPM_VI_1D()
