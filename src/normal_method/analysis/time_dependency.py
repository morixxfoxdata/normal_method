import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# サンプルデータの生成（実際のデータに置き換えてください）
from src.normal_method.data import mnist_total

XX, Y = mnist_total()


def analyze_temporal_dependency(Y):
    n_images, n_timepoints = Y.shape

    n_images, n_timepoints = Y.shape

    # 1. 自己相関分析
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Autocorrelation Analysis", fontsize=16)
    max_lag = 100
    for i in range(n_images):
        acf_result = acf(Y[i], nlags=max_lag)
        plot_acf(Y[i], ax=axs[i // 5, i % 5], lags=max_lag, title=f"Image {i+1}")

        # Find the lag with the highest correlation (excluding lag 0)
        max_corr_lag = np.argmax(np.abs(acf_result[1:]))
        axs[i // 5, i % 5].axvline(
            x=max_corr_lag,
            color="r",
            linestyle="--",
            label=f"Max corr at lag {max_corr_lag}",
        )
        axs[i // 5, i % 5].legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.show()
    # 計算した自己相関結果を表示
    print("Maximum correlation lags for each image:")
    for i in range(n_images):
        acf_result = acf(Y[i], nlags=max_lag)
        max_corr_lag = np.argmax(np.abs(acf_result[1:]))
        print(f"Image {i+1}: Lag {max_corr_lag}")

    # 2. パワースペクトル密度の計算
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Power Spectral Density", fontsize=16)
    for i in range(n_images):
        f, Pxx = signal.periodogram(Y[i])
        axs[i // 5, i % 5].semilogy(f, Pxx)
        axs[i // 5, i % 5].set_xlabel("Frequency")
        axs[i // 5, i % 5].set_ylabel("PSD")
        axs[i // 5, i % 5].set_title(f"Image {i+1}")
    plt.tight_layout()
    plt.show()

    # 3. ウェーブレット変換
    fig, axs = plt.subplots(2, 5, figsize=(20, 12))
    fig.suptitle("Wavelet Transform", fontsize=16)
    for i in range(n_images):
        widths = np.arange(1, 31)
        cwtmatr = signal.cwt(Y[i], signal.ricker, widths)
        axs[i // 5, i % 5].imshow(
            cwtmatr,
            extent=[0, n_timepoints, 1, 31],
            cmap="PRGn",
            aspect="auto",
            vmax=abs(cwtmatr).max(),
            vmin=-abs(cwtmatr).max(),
        )
        axs[i // 5, i % 5].set_title(f"Image {i+1}")
        axs[i // 5, i % 5].set_ylabel("Scale")
        axs[i // 5, i % 5].set_xlabel("Time")
    plt.tight_layout()
    plt.show()

    # 4. サンプルエントロピーの計算
    def sample_entropy(time_series, m, r):
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [
                [time_series[j] for j in range(i, i + m - 1 + 1)]
                for i in range(N - m + 1)
            ]
            C = [
                len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r])
                for i in range(len(x))
            ]
            return sum(C)

        N = len(time_series)
        return -np.log(_phi(m + 1) / _phi(m))

    entropies = [sample_entropy(img, m=2, r=0.2 * np.std(img)) for img in Y]

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_images + 1), entropies)
    plt.xlabel("Image")
    plt.ylabel("Sample Entropy")
    plt.title("Sample Entropy of Each Image")
    plt.show()

    # 5. トレンド分析
    from scipy import stats

    trends = [stats.linregress(range(n_timepoints), img).slope for img in Y]

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_images + 1), trends)
    plt.xlabel("Image")
    plt.ylabel("Trend (Slope)")
    plt.title("Trend Analysis of Each Image")
    plt.show()


# 分析の実行
analyze_temporal_dependency(Y)
