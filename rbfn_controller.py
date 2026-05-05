import numpy as np

class RBFNController:
    """
    RBFN 控制器類別，用於載入訓練好的參數並進行預測。
    """
    def __init__(self, model_file_path):
        """
        初始化控制器，並從指定路徑載入中心點、權重和 sigma。
        :param model_file_path: 儲存 RBFN 參數的 .npz 檔案路徑。
        """
        self.centers = None
        self.weights = None
        self.sigma = None
        
        try:
            # 載入 .npz 檔案，它包含了我們儲存的參數
            data = np.load(model_file_path)
            self.centers = data['centers']
            self.weights = data['weights']
            self.sigma = data['sigma'].item() # .item() 將單一元素陣列轉換為 Python float
            print(f"RBFN 模型載入成功：中心點數量={self.centers.shape[0]}, Sigma={self.sigma}")
        except FileNotFoundError:
            print(f"錯誤：找不到模型檔案 {model_file_path}。")
            raise
        except KeyError as e:
            print(f"錯誤：模型檔案 {model_file_path} 缺少必要的鍵：{e}。")
            raise

    def _gaussian_basis_func(self, c, x):
        """高斯徑向基函數 (Gaussian RBF)"""
        # 注意：這裡的計算必須與訓練時的邏輯完全一致
        distance_sq = np.sum((x - c) ** 2)
        return np.exp(-distance_sq / (2 * self.sigma ** 2))

    def _calculate_hidden_layer_output(self, X):
        """計算隱藏層 (RBF 輸出) 的輸出矩陣 H"""
        # X 必須是一個 (1, n_features) 的 NumPy 陣列
        
        # 建立一個包含所有 RBF 輸出值的列表
        h_values = []
        for j in range(self.centers.shape[0]):
            h_val = self._gaussian_basis_func(self.centers[j], X[0])
            h_values.append(h_val)
                
        # 轉換為 NumPy 陣列
        H = np.array(h_values).reshape(1, -1)
                
        # 加上偏置項 (Bias)，H 變成 (1, n_centers + 1)
        H_with_bias = np.concatenate([H, np.ones((1, 1))], axis=1)
        return H_with_bias

    def predict(self, input_features):
        """
        預測最佳方向盤角度 phi。
        :param input_features: 包含特徵的列表或 NumPy 陣列 (例如 [x, y, D_f, D_r, D_l])
        :return: 預測的方向盤角度 (float)
        """
        # 1. 確保輸入是 (1, n_features) 的 NumPy 陣列
        X_input = np.array(input_features).reshape(1, -1)
        
        # 2. 如果訓練時有進行數據歸一化，這裡必須使用相同的縮放器對 X_input 進行歸一化！
        # X_input_normalized = scaler.transform(X_input) 
        
        # 3. 計算隱藏層輸出 H
        H = self._calculate_hidden_layer_output(X_input)
        
        # 4. 輸出層輸出 Y_pred = H @ W
        Y_pred = H @ self.weights
        
        # 輸出結果是 (1, 1) 的矩陣，取其單一數值
        predicted_angle = Y_pred[0, 0]
        
        return predicted_angle

# --------------------------------------------------------------------------
# 輔助：如何儲存模型（在您的訓練腳本 .ipynb 中執行過）
# 
# # 假設 rbfn_model 是訓練好的模型實例
# model_data = {
#     'centers': rbfn_model.centers,
#     'weights': rbfn_model.weights,
#     'sigma': np.array(rbfn_model.sigma) # 將 sigma 包裝成 array 以便 np.savez 存取
# }
# np.savez('rbfn_model_weights.npz', **model_data)
# --------------------------------------------------------------------------