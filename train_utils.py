import numpy as np
import pandas as pd
import math as m
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from typing import Tuple, List, Optional

# 1. 數據檔案路徑定義
# 假設這些檔案位於運行腳本的同一目錄或是相對路徑
FILE_4D = 'train4dAll.txt'
FILE_6D = 'train6dAll.txt'

# 2. 數據載入函數
# 根據模式載入並解析訓練數據
def load_and_parse_data(is_6d_mode:bool)-> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # is_6d_mode == true -> 6D (X, Y, D_f, D_r, D_l)
    # is_4d_mode == flase -> 4D (D_f, D_r, D_l)
    filename = FILE_6D if is_6d_mode else FILE_4D

    try:
        df = pd.read_csv(filename, header = None, sep = "\s+")
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {filename}，請檢查路徑。")
        return None, None
    
    # 刪除可能因為讀取空格導致的多餘 NaN 欄位
    df = df.dropna(axis=1, how='all')
    
    if df.empty or df.shape[1] < 2:
        print(f"錯誤：檔案 {filename} 讀取後數據不足。")
        return None, None

    # X 是前 n-1 欄，Y 是最後一欄
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values.reshape(-1, 1)

    return X, Y

# 3. RBFN模型類別
# RBF神經網路的實現，包含訓練和預測邏輯
class RBFN:
    def __init__(self, n_centers: int, sigma: float):
        self.n_centers = n_centers
        self.sigma = sigma
        self.centers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    def _gaussian_basis_func(self, c: np.ndarray, x: np.ndarray) -> float:
        """高斯徑向基函數"""
        distance_sq = np.sum((x - c) ** 2)
        return np.exp(-distance_sq / (2 * self.sigma ** 2))

    def _calculate_hidden_layer_output(self, X: np.ndarray) -> np.ndarray:
        """計算隱藏層 (RBF 輸出) 的輸出矩陣 H，包含偏置項 (Bias)。"""
        N = X.shape[0]
        H = np.zeros((N, self.n_centers))
        
        for i in range(N):
            for j in range(self.n_centers):
                H[i, j] = self._gaussian_basis_func(self.centers[j], X[i])
                
        # 加上偏置項 (Bias)
        H_with_bias = np.concatenate([H, np.ones((N, 1))], axis=1)
        return H_with_bias

    def train(self, X_train: np.ndarray, Y_train: np.ndarray):
        """RBFN 訓練步驟：K-Means 找中心點，偽逆矩陣找權重。"""
        
        if X_train is None or Y_train is None:
            raise ValueError("訓練數據無效，無法啟動訓練。")

        # 步驟 1: K-Means 確定中心點
        kmeans = KMeans(n_clusters=self.n_centers, random_state=0, n_init='auto')
        kmeans.fit(X_train)
        self.centers = kmeans.cluster_centers_

        # 步驟 2: 計算隱藏層輸出矩陣 H
        H = self._calculate_hidden_layer_output(X_train)

        # 步驟 3: 求解輸出權重 W (偽逆矩陣)
        H_pseudo_inverse = np.linalg.pinv(H)
        self.weights = H_pseudo_inverse @ Y_train

    def predict(self, X_input: List[float]) -> float:
        """預測最佳方向盤角度 phi。"""
        if self.weights is None or self.centers is None:
            # 確保模型已訓練
            raise ValueError("模型尚未訓練。")
            
        # 將輸入列表轉換為 NumPy 陣列 (1, n_features)
        X_input_array = np.array(X_input).reshape(1, -1)
        
        # 計算隱藏層輸出 H
        H = self._calculate_hidden_layer_output(X_input_array)
        
        # 輸出層輸出 Y_pred = H @ W
        Y_pred = H @ self.weights
        
        # 返回角度的單一數值
        return Y_pred[0, 0]

    def evaluate(self, X_data: np.ndarray, Y_data: np.ndarray) -> float:
        """計算訓練集或測試集的 RMSE。"""
        if self.weights is None:
            return float('inf')
            
        # 為了計算 RMSE，我們需要批次預測
        H = self._calculate_hidden_layer_output(X_data)
        Y_pred = H @ self.weights
        
        return np.sqrt(mean_squared_error(Y_data, Y_pred))