import sys
import os
import numpy as np
import math as m
import matplotlib.pyplot as plt
import warnings

# 引入PyQt5核心組件
from PyQt5 import QtWidgets, QtCore     # 引入布局管理器
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer     # 引入計時器

# 引入matplotlib嵌入組件
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 引入專案核心模組
from simple_playground import Playground
from train_utils import RBFN, load_and_parse_data

# 抑制numpy產生的除以0或是無效值警告
warnings.filterwarnings("ignore", "divide by zero encountered in scalar divide", RuntimeWarning, module='simple_geometry')
warnings.filterwarnings("ignore", "invalid value encountered in scalar divide", RuntimeWarning, module='simple_geometry')

# 全局常量
UI_FILE = "car_simulator.ui"   # UI檔案路徑
MODEL_WEIGHTS_FILE = "rbfn_model_weights.npz"
UPDATE_INTERVAL_MS = 50    # 模擬更新間隔（毫秒）
USE_6D_MODEL = True
RECORD_FILE = "successful_path.txt"

# --- 解決 Matplotlib 中文顯示問題 ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 1.載入UI介面
        try:
            loadUi(os.path.join(os.path.dirname(__file__), UI_FILE), self)
        
        except FileNotFoundError:
            print(f"錯誤：找不到UI檔案 {UI_FILE}。請確認檔案存在且路徑正確。")
            sys.exit(1)
        
        # 初始化data_selector的邏輯
        try:
            if hasattr(self, 'data_selector'):
                self.data_selector.clear()
                self.data_selector.addItem("4D")
                self.data_selector.addItem("6D")
                self.data_selector.setCurrentText("6D")
        except Exception as e:
            # 如果這裡出錯，通常是 UI 元件名稱拼寫錯誤或類型不對
            print(f"警告：初始化 data_selector 失敗。請檢查 UI 名稱。錯誤: {e}")
        
        # 2. 初始化核心組件
        self.playground = Playground()
        self.controller = None
        
        self.path_x = []
        self.path_y = []
        self.path_record = []
        self.is_playback_mode = False
        self.playback_step = 0
        self.playback_data = None
        self.is_6D_trained = False

        # 3. 初始化繪圖畫布
        self.init_canvas()

        # 4. 設置QTimer(模擬驅動)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_next_step)

        # 5. 連接信號槽(signals and slots)
        self.connect_signals()

        # 初始狀態設置
        self.reset_simulation()

    # 連接按鈕到對應的函數(slot)
    def connect_signals(self):
        self.start_button.clicked.connect(self.start_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.playback_button.clicked.connect(self.load_and_start_playback)
        self.train_button.clicked.connect(self.train_model)

    # 設置所有互動元件的啟用狀態
    def set_ui_interaction_enabled(self, enabled: bool):
        self.start_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.playback_button.setEnabled(enabled)
        self.data_selector.setEnabled(enabled)
        self.spinBox_nc.setEnabled(enabled)
        self.spinBox_sigma.setEnabled(enabled)
        # train_button 保持啟用，除非它本身正在執行訓練（我們在這裡暫時不禁用它，以避免邏輯循環）
        # self.train_button.setEnabled(enabled) 
    
        # 強制更新 GUI 狀態
        QtWidgets.QApplication.processEvents()

    def train_model(self):
        self.timer.stop()
        self.set_ui_interaction_enabled(False)
        self.train_button.setEnabled(False)

        # 1. 讀取用戶設定和超參數
        try:
            # 從 GUI 元件中讀取數據
            n_centers = int(self.spinBox_nc.text())
            sigma = float(self.spinBox_sigma.text())
            
            # 判斷數據模式 
            data_mode = self.data_selector.currentText()
            is_6d_mode = (data_mode == "6D")
        
        except ValueError:
            self.statusBar().showMessage("訓練錯誤：N_CENTERS 和 SIGMA 必須為有效數字。", 5000)
            return
        
        if n_centers <= 0 or sigma <= 0:
            self.statusBar().showMessage("訓練錯誤：中心點和 Sigma 必須大於零。", 5000)
            return
        
        self.statusBar().showMessage(f"開始訓練 RBFN 模型 ({data_mode} 模式, Centers={n_centers}, Sigma={sigma})...", 0)
        
        # 2. 數據載入
        X_train, Y_train = load_and_parse_data(is_6d_mode)
        
        if X_train is None:
            self.statusBar().showMessage("訓練失敗：數據載入錯誤。", 5000)
            return
        
        # 3. 執行訓練
        try:
            rbfn_model = RBFN(n_centers=n_centers, sigma=sigma)
            rbfn_model.train(X_train, Y_train) # 這是阻塞操作，可能會讓 GUI 短暫卡住
            
            # 4. 評估與更新控制器
            rmse = rbfn_model.evaluate(X_train, Y_train)
            self.controller = rbfn_model
            
            self.statusBar().showMessage(f"訓練成功！RMSE: {rmse:.4f}度。模型已準備就緒。", 0)
            
        except Exception as e:
            self.statusBar().showMessage(f"訓練失敗: 發生內部錯誤 - {e}", 8000)
            self.controller = None
        
        finally:
            self.set_ui_interaction_enabled(True)
            self.train_button.setEnabled(True)

        self.is_6D_trained = is_6d_mode       

    # 初始化Matplotlib畫布並嵌入到UI容器中
    def init_canvas(self):
        # 創建圖表和軸(用於繪製跑道)
        self.fig, self.ax = plt.subplots(figsize = (8, 8), tight_layout = True)
        self.canvas = FigureCanvas(self.fig)

        # 設置布局管理器並嵌入畫布
        layout = QtWidgets.QVBoxLayout(self.plot_container)
        layout.addWidget(self.canvas)

        # 添加導航工具欄(可選，用於縮放、平移等)
        # self.toolbar = NavigationToolbar(self.canvas, self.plot_container)
        # layout.addWidget(self.toolbar)

        # 設定繪圖初始限制
        # 假設跑道主要在X:[-10, 40], Y:[-10, 110]範圍
        self.ax.set_xlim(-10, 35)
        self.ax.set_ylim(-10, 60)
        # self.ax.set_aspect("equal", adjustable = "box")
        self.ax.set_title("RBFN模型車路徑模擬")

        # 繪製靜態跑道邊界(只執行一次)
        self.plot_track()

    # 繪製跑道邊界和終點區域
    def plot_track(self):
        # 清空軸，只保留靜態軌道
        self.ax.cla()
        self.ax.set_xlim(-10, 35)
        self.ax.set_ylim(-10, 60)
        self.ax.set_aspect("equal", adjustable = "box")
        self.ax.set_title("RBFN模型車路徑模擬")

        # 繪製跑道線段(從playground實例獲取邊界)
        for line in self.playground.lines:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], "k-", linewidth = 2)

        # 繪製終點區域(矩形)
        dp1 = self.playground.destination_line.p1
        dp2 = self.playground.destination_line.p2

        # 終點座標:(0, 90), (6, 99)
        rect_x = min(dp1.x, dp2.x)
        rect_y = min(dp1.y, dp2.y)
        rect_width = abs(dp1.x - dp2.x)
        rect_height = abs(dp1.y - dp2.y)

        from matplotlib.patches import Rectangle
        rect = Rectangle((rect_x, rect_y), rect_width, rect_height, 
                facecolor = "green", alpha = 0.3, label = "Destination")
        self.ax.add_patch(rect)
            
        # 重新呼叫legend()
        self.ax.legend(loc = "upper right", 
                    bbox_to_anchor = (1.05, 1.0), 
                    handles = [rect]) # 確保只顯示最後添加的patch標籤
        self.canvas.draw()

    # 重置模擬狀態，為下一次運行做準備
    def reset_simulation(self):
        self.timer.stop()
        self.playground.reset()

        # 清空軌跡
        self.path_x = []
        self.path_y = []
        self.path_record = []

        # 修正初始位置為作業要求(-6, 0, 90度)
        self.playground.setCarPosAndAngle(
            position = self.playground.car.getPosition("center").__class__(0.0, 1.0), 
            angle = 90.0
        )
        self.playground._checkDoneIntersects()
        initial_sensor_dists = self.playground.state

        # 設置初始軌跡點 (使用當前最新的 X/Y 坐標)
        current_x = self.playground.car.xpos
        current_y = self.playground.car.ypos
        self.path_x.append(current_x) # 確保這裡有初始點
        self.path_y.append(current_y) # 確保這裡有初始點

        # 清空並重繪圖表
        self.plot_track()

        # 設置初始車輛點和軌跡線(用於更新)
        self.car_plot, = self.ax.plot([self.path_x[0]], [self.path_y[0]], "ro", markersize = 8)
        self.path_line, = self.ax.plot(self.path_x, self.path_y, "b-", linewidth = 1)
        
        # 設置初始感測器射線(3個方向)
        self.sensor_plots = []
        for _ in range(3):
            # 初始時點都在(x, y)，線段長度為0
            line, = self.ax.plot([self.path_x[0]], [self.path_y[0]], 
                    "--", color = "gray", alpha = 0.5, linewidth = 1)
            self.sensor_plots.append(line)
            # 其中0 = 前, 1 = 右, 2 = 左

        self.canvas.draw()

        self._update_labels(0.0, initial_sensor_dists)
        self.statusBar().showMessage("模擬已重置。按'start'開始運行。")
        
    def load_and_start_playback(self):
        self.timer.stop()
        if not os.path.exists(RECORD_FILE):
            self.statusBar().showMessage("錯誤：找不到路徑記錄檔 (successful_path.txt)。請先成功運行一次模擬。", 5000)
            return

        # 1. 嘗試載入記錄檔
        try:
            data = np.loadtxt(RECORD_FILE, delimiter=',')
            if data.ndim == 1:
                # 處理只有一行數據的情況 (例如，只有一條記錄)
                data = data.reshape(1, -1)
            
            self.playback_data = data
            if not self.playback_data.size:
                self.statusBar().showMessage("錯誤：路徑記錄檔為空。請先成功運行一次模擬。", 5000)
                return
                
        except Exception as e:
            self.statusBar().showMessage(f"錯誤：載入路徑記錄檔失敗: {e}", 5000)
            return

        # 2. 初始化重放狀態
        self.reset_simulation()
        self.is_playback_mode = True
        self.playback_step = 0

        # 3. 連接定時器到重放函數
        self.timer.timeout.connect(self.run_next_step)

        self.statusBar().showMessage("重放模式:正在讀取並重現路徑(共 {len(self.playback_data)} 步)...", 0)
        self.timer.start(UPDATE_INTERVAL_MS)

    # QTimer 觸發的單一入口，根據模式執行對應邏輯。
    def run_next_step(self):
        if self.is_playback_mode:
            self._run_playback_step()
        else:
            self._run_simulation_step()
        
    # 即時控制模式：模型預測並推進運動學。
    def _run_simulation_step(self):
        # 1. 檢查結束條件
        self.playground._checkDoneIntersects()
        if self.playground.done:
            self.timer.stop()
            self.check_end_status(is_playback=False) # 區分模式
            return
                
        # 2. 獲取數據
        current_x = self.playground.car.xpos
        current_y = self.playground.car.ypos
        sensor_dists = self.playground.state # [D_f, D_r, D_l]
        full_input = [current_x, current_y] + sensor_dists
        
        # 3. 模型預測
        if self.is_6D_trained:
            input_features = full_input
        else:
            input_features = sensor_dists

        predicted_angle = self.controller.predict(input_features)

        # 4. 推進運動學
        self.playground.car.setWheelAngle(predicted_angle)
        self.playground.car.tick()
            
        # 5. 紀錄成功路徑 (僅在成功運行時)
        new_x = self.playground.car.xpos
        new_y = self.playground.car.ypos
        
        record_row = [
            new_x, new_y, sensor_dists[0], sensor_dists[1], sensor_dists[2], predicted_angle.item()
        ]
        self.path_record.append(record_row)

        # 6. 推進繪圖
        car_center = self.playground.car.getPosition("center")
        self._update_display(car_center, sensor_dists, predicted_angle.item())
        
    # 數據重放模式 : 讀取數據並設定位置
    def _run_playback_step(self):
        if self.playback_step >= len(self.playback_data):
            self.timer.stop()
            self.is_playback_mode = False
            self.statusBar().showMessage("重放結束：路徑已完整重現。", 0)
            return
            
        # 1. 從檔案讀取數據 (格式: X, Y, Df, Dr, Dl, Phi)
        record_row = self.playback_data[self.playback_step]
        x_new, y_new, sensor_f, sensor_r, sensor_l, phi_angle = record_row
        
        # 2. 直接設定位置 (忽略模型和運動學)
        from simple_geometry import Point2D
        self.playground.car.setPosition(Point2D(x_new, y_new))
        self.playground.car.setWheelAngle(phi_angle)
        
        # 3. 碰撞檢查 (滿足作業要求)
        self.playground._checkDoneIntersects() 
        if self.playground.done:
            self.timer.stop()
            self.is_playback_mode = False
            self.check_end_status(is_playback=True)
            return

        # 4. 推進繪圖 (使用檔案中的感測器數據)
        car_center = self.playground.car.getPosition("center")
        sensor_dists_playback = [sensor_f, sensor_r, sensor_l]
        
        # 修正 5：重放模式下的軌跡點添加
        self.path_x.append(x_new)
        self.path_y.append(y_new)
        
        self._update_display(car_center, sensor_dists_playback, phi_angle) 
        self.playback_step += 1
    
    # 統一繪圖和標籤更新(無論是模擬或是重放模式)
    def _update_display(self, car_center, sensor_dists, phi_angle):
        # 1. 獲取當前位置
        x, y = car_center.x, car_center.y

        # 2. 確保即時模式下獲取最新的交點數據來繪製射線
        intersections = [self.playground.front_intersects, self.playground.right_intersects, self.playground.left_intersects]
        
        # 3. 更新軌跡線
        self.car_plot.set_data([x], [y])
        self.path_line.set_data(self.path_x, self.path_y)

        # 4. 獲取一次車頭角度，並定義相對角度(only once)
        car_angle_head = self.playground.car.angle # 車頭的絕對角度
        SENSOR_RELATIVE_ANGLES = [0, -45, 45]

        # 5. 更新感測器射線
        # 根據模式決定是否顯示射線
        visibility = not self.is_playback_mode

        for i, current_intersect_list in enumerate(intersections):
            start_x, start_y = x, y

            # 確保angle每次都從car_angle_head重新計算
            current_abs_angle = car_angle_head + SENSOR_RELATIVE_ANGLES[i]
            
            # 將角度轉換為弧度，強制使用math.radians確保正確性
            rad = m.radians(current_abs_angle)
            
            # 判斷是否有交點
            # 如果有交點，則使用交點位置
            if current_intersect_list:
                end_x, end_y = current_intersect_list[0].x, current_intersect_list[0].y
            
            # 處理暢通或是距離有效但intersects為空的情況
            else:
                # 使用傳入的sensor_dists參數
                L = sensor_dists[i]
            
                if L == -1:   # 假設該方向暢通
                    # 為了快速實現，先假設不畫暢通的線，或將其長度設定為一個固定值
                    L = 50
                
                # 計算終點(不論暢通或是有讀數，公式都一樣)
                end_x = x + L * m.cos(rad)
                end_y = y + L * m.sin(rad)            
        
            # 更新射線的數據
            self.sensor_plots[i].set_data([start_x, end_x], [start_y, end_y])

            # 設置射線可見性
            self.sensor_plots[i].set_visible(visibility)
        
        # 6. 更新標籤
        self._update_labels(phi_angle, sensor_dists)

        # 7. 快速重繪(只重繪發生變化的部分)
        self.canvas.draw()
    
    # 開始或繼續模擬
    def start_simulation(self):
        if self.controller is None:
            self.statusBar().showMessage("錯誤：請先訓練模型！(按 Train 按鈕)", 5000)
            return

        if self.playground.done:
            # 如果已經結束，則先重置
            self.reset_simulation()
            
        if not self.timer.isActive():
            self.timer.start(UPDATE_INTERVAL_MS)
            self.statusBar().showMessage("模擬運行中...")

    # 更新QLabel顯示的即時數值
    def _update_labels(self, phi_angle, sensor_dists):
        # 顯示座標位置與方向盤角度
        self.label_xpos.setText(f"{self.playground.car.xpos:.2f}")
        self.label_ypos.setText(f"{self.playground.car.ypos:.2f}")
        self.label_phi.setText(f"{phi_angle:.2f}°")

        # 獲取 & 顯示感測器讀數
        # sensor_dists = self.playground.state

        self.label_sens_l.setText(f"{sensor_dists[2]:.2f}")
        self.label_sens_f.setText(f"{sensor_dists[0]:.2f}")
        self.label_sens_r.setText(f"{sensor_dists[1]:.2f}")
        
    # 檢查模擬結束的原因(成功或失敗)
    def check_end_status(self, is_playback):
        car_pos = self.playground.car.getPosition("center")
        dp1 = self.playground.destination_line.p1
        dp2 = self.playground.destination_line.p2

        is_success = car_pos.isInRect(dp1, dp2)

        if is_success:
            self.statusBar().showMessage("模擬結束：成功抵達終點！", 0)
            if not is_playback:
                if self.path_record:
                    try:
                        np.savetxt(RECORD_FILE, np.array(self.path_record), fmt = "%.4f", delimiter = ",")
                        # 確認記錄欄位數量
                        print(f"成功路徑已記錄到 {RECORD_FILE} (共 {len(self.path_record)} 步, 6 欄)")
                    except Exception as e:
                        print(f"❌ 路徑記錄失敗: {e}")
                else:
                    pass    
        else:
            self.statusBar().showMessage("模擬結束：車輛碰撞或偏離跑道！", 0)

# --- 主程式執行 ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())