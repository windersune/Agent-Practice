import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.font_manager as fm
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import time
import threading
import ttkbootstrap as tb  # 导入 ttkbootstrap

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 配置部分 =================
KIMI_API_KEY = "sk-iHo09Md7LCPPvmpDPRe6x9SYcFOBsk2q2nvFHFfMaSvW4M0K"  # 替换为实际API密钥
TARGET_STRAIN = 46.6
PLOT_STYLE = "bmh"

# ================= 核心函数 =================
class StressStrainAnalyzer:
    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.df = None
        self.strains = None
        self.stresses = None
        self.kimi_analysis = {}
        plt.style.use(PLOT_STYLE)

    def load_data(self):
        """加载并验证Excel数据"""
        try:
            if not self.input_path.exists():
                raise FileNotFoundError(f"输入文件 {self.input_path} 不存在")
            self.df = pd.read_excel(self.input_path, engine='openpyxl')
            self.df.columns = self.df.columns.str.strip()

            strain_col = None
            stress_col = None

            for col in self.df.columns:
                if "应变" in col.lower():
                    if strain_col is None:
                        strain_col = col
                if "应力" in col.lower():
                    if stress_col is None:
                        stress_col = col

            if strain_col is None or stress_col is None:
                missing = set()
                if strain_col is None:
                    missing.add("应变")
                if stress_col is None:
                    missing.add("应力")
                raise ValueError(f"缺少必要列 (或无法识别包含关键词 '应变'/'应力' 的列): {missing}")

            self.strains = self.df[strain_col].values.astype(float)
            self.stresses = self.df[stress_col].values.astype(float)
            self._validate_data()

        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise

    def _validate_data(self):
        """数据验证逻辑"""
        if len(self.strains) < 5:
            raise ValueError("至少需要10个数据点进行分析")

        if not np.all(np.diff(self.strains) > 0):
            raise ValueError("应变数据必须单调递增")

        if np.max(self.strains) >= TARGET_STRAIN:
            raise ValueError(f"应变数据已超过目标值 {TARGET_STRAIN}")

    def analyze_with_kimi(self):
        """使用Kimi大模型进行趋势分析"""
        prompt = f"""
        作为资深的材料力学专家，请分析以下应力-应变数据，并判断应变和应力的趋势，同时为我**建议一个最合适的数学模型**来拟合这些数据，用于后续的预测。
        **我的目标：**  理解应力-应变数据的趋势，并选择合适的模型进行外推预测。
        **已有的应力-应变数据 (最多最近5个点):**
        {self._format_recent_data()}
        **我的要求：**
        1.  **应变趋势分析:**  判断应变数据是呈现线性增长、指数增长还是多项式增长等趋势。
        2.  **应力趋势分析:**  判断应力数据是呈现线性增长、指数增长还是多项式增长等趋势。
        3.  **模型建议:**  从以下模型中选择一个最合适的模型来拟合应力-应变关系：`linear` (线性模型), `poly2` (二次多项式), `poly3` (三次多项式), `expo` (指数模型)。请给出模型名称即可。
        4.  **置信度评估:**  请对你建议的模型给出置信度评估，范围为 0-1 (1为最高置信度)。
        5.  **风险提示:**  请根据数据情况，给出任何可能的风险提示，例如外推预测可能不准确，数据量不足等等。
        **请用 JSON 格式返回结果，包含以下字段：**
        `strain_trend` (应变趋势), `stress_trend` (应力趋势), `suggested_model` (建议模型名称), `confidence` (置信度), `warning` (风险提示)
        **JSON 返回示例:**
        ```json
        {{
          "strain_trend": "线性",
          "stress_trend": "多项式",
          "suggested_model": "poly2",
          "confidence": 0.8,
          "warning": "数据量较少，外推预测可能存在一定误差，请注意风险。"
        }}
        ```
        请开始分析并给出 JSON 结果。
        """

        response = self._call_kimi_api(prompt)
        try:
            self.kimi_analysis = json.loads(response[response.find('{'):response.rfind('}')+1])
        except:
            print("Kimi分析结果解析失败，使用默认配置")
            self.kimi_analysis = {
                "strain_trend": "多项式",
                "stress_trend": "多项式",
                "suggested_model": "poly2",
                "confidence": 0.7,
                "warning": "注意外推预测风险"
            }

    def _format_recent_data(self):
        """格式化最近的几个数据点，用于 Kimi Prompt"""
        sample_size = min(5, len(self.strains))
        sample = []
        start_index = max(0, len(self.strains) - sample_size)
        for i in range(start_index, len(self.strains)):
            sample.append(f"应变: {self.strains[i]:.5f}%, 应力: {self.stresses[i]:.5f} MPa")
        return "\n".join(sample)

    def _call_kimi_api(self, prompt):
        """调用Kimi API"""
        headers = {
            "Authorization": f"Bearer {KIMI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "moonshot-v1-8k",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }

        try:
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            time.sleep(1)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return "{}"

    def build_prediction_models(self):
        """构建预测模型"""
        self.stress_model = self._create_model(
            x=self.strains.reshape(-1,1),
            y=self.stresses,
            model_type=self.kimi_analysis.get('suggested_model', 'poly2')
        )

    def _create_model(self, x, y, model_type):
        """动态创建预测模型"""
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type.startswith('poly'):
            degree = int(model_type[-1])
            model = make_pipeline(
                PolynomialFeatures(degree),
                LinearRegression()
            )
        elif model_type == 'expo':
            model = LinearRegression()
            y = np.log(y)
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        model.fit(x, y)
        return model

    def generate_data(self):
        """生成扩展数据，使用Kimi建议应变增量"""
        extended_data = []
        current_strain = self.strains[-1]
        step = 0
        max_steps = 10000
        tolerance = 0.0001
        min_increment = 0.00001
        default_increment = 0.0001

        while current_strain < (TARGET_STRAIN - tolerance) and step < max_steps:
            step += 1

            # ==== 调用Kimi获取建议应变增量 ====
            kimi_increment_prompt = f"""
                作为资深的材料力学专家，请根据以下应力-应变数据，为我预测并建议**下一个合理的应变增量值**，以便我能更平滑、更合理地外推应力-应变曲线至目标应变值 {TARGET_STRAIN}%。
                **我的目标：** 预测应力-应变曲线，直到应变达到 {TARGET_STRAIN}%。
                **已有的应力-应变数据 (最近100个点):**
                {self._format_recent_data()}
                **我的模型信息:**
                * 当前应力-应变关系模型: {self.kimi_analysis.get('suggested_model', 'poly2')} (Kimi建议)
                * 模型置信度: {self.kimi_analysis.get('confidence', 0)*100:.5f}%
                * 趋势分析 (Kimi分析): 应变趋势为 {self.kimi_analysis.get('strain_trend', '未知')}, 应力趋势为 {self.kimi_analysis.get('stress_trend', '未知')}
                **我的要求:**
                1.  请分析现有数据的趋势和模型信息，并**预测一个合理的应变增量值** (建议为万分比数值，例如 0.00001, 0.00003 等)。  **请尽量给出 0.00001 - 0.1 范围内的增量，避免过大或过小。**
                2.  请**简要解释你建议这个增量的原因** (例如，为了保持曲线平滑性，避免过快/过慢增长，考虑到现有数据趋势等)。
                3.  请以 **JSON 格式** 返回结果，包含字段: `suggested_strain_increment` (建议的应变增量数值), `reasoning` (建议原因)。
                **JSON 返回示例:**
                ```json
                {{
                  "suggested_strain_increment": 0.00001,
                  "reasoning": "根据现有数据呈多项式增长趋势，且模型置信度较高，建议采用较小的增量以保持曲线平滑并逐步接近目标应变。"
                }}
                ```
                请给出你的建议。
                """
            increment_response = self._call_kimi_api(kimi_increment_prompt)
            try:
                increment_analysis = json.loads(increment_response[increment_response.find('{'):increment_response.rfind('}')+1])
                suggested_increment = float(increment_analysis.get("suggested_strain_increment", default_increment))
                suggested_increment = max(suggested_increment, min_increment)
                print(f"Kimi建议应变增量: {suggested_increment}, 原因: {increment_analysis.get('reasoning', '无')}")
            except:
                print("Kimi应变增量建议解析失败，使用默认增量 0.00003")
                suggested_increment = default_increment

            next_strain = current_strain + suggested_increment
            next_stress = self.stress_model.predict([[next_strain]])[0]

            extended_data.append({
                '应变': round(next_strain, 5),
                '应力': round(next_stress, 5)
            })
            current_strain = next_strain
            print(f"Step {step}: current_strain = {current_strain:.5f}, next_strain = {next_strain:.5f}")

        return pd.DataFrame(extended_data)

    def save_results(self, extended_df, output_dir):
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存Excel
        output_path = output_dir / "预测结果.xlsx"
        full_df = pd.concat([self.df, extended_df], ignore_index=True)
        full_df.to_excel(output_path, index=False)

        # 保存图表
        plot_path = output_dir / "应力应变曲线.png"
        self._plot_curve(full_df, plot_path)

        return output_path, plot_path

    def _plot_curve(self, df, save_path):
        """绘制应力应变曲线"""
        plt.figure(figsize=(10, 6))

        colors = plt.cm.tab10.colors
        original_color = colors[0]
        predicted_color = colors[2]

        # 原始数据
        plt.plot(df['应变'][:len(self.strains)],
                df['应力'][:len(self.stresses)],
                linestyle='-',
                color=original_color,
                label='原始数据',
                linewidth=1)

        # 预测数据
        if len(df) > len(self.strains):
            plt.plot(df['应变'][len(self.strains):],
                    df['应力'][len(self.stresses):],
                    linestyle='-',
                    linewidth=1,
                    color=predicted_color,
                    label='预测数据')

        plt.xlabel('应变 (%)', fontsize=12)
        plt.ylabel('应力 (MPa)', fontsize=12)
        plt.title(f'应力-应变曲线（置信度: {self.kimi_analysis.get("confidence", 0)*100:.5f}%）', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


# ================= GUI 部分 =================
class StressStrainApp(tb.Window):
    def __init__(self):
        super().__init__(themename="morph")  # 使用 ttkbootstrap 主题
        self.title("应力应变分析Agent")
        self.geometry("800x750")
        self.minsize(800, 750)

        # 分析状态
        self.analysis_steps = 5
        self.current_step = 0
        self.stop_event = threading.Event() # 用于停止线程
        self.analysis_thread = None
        self.initial_image_text = "图像生成中..." #  设置初始提示文字

        # 获取 style 对象
        style = tb.Style()

        # 自定义按钮样式
        style.configure("TButton",
                        borderwidth=0,
                        relief="solid",
                        padding=5,
                        bordercolor = 'none',
                        focusthickness = 0,
                        focuscolor = 'none',
                        borderradius = 10,
                         shadowcolor="#808080",  # 添加阴影颜色
                        shadowoffset = (3, 3)
                        )
        # 添加阴影效果
        style.configure("TLabel",
                           shadowcolor="#808080",
                           shadowoffset = (3, 3)
                       )
        # 创建白色网格背景
        self.grid_bg_color = "white"


        # 界面组件
        self.create_widgets()

         # 分析器对象和路径
        self.analyzer = None
        self.output_path = None
        self.plot_path = None


    def create_widgets(self):
         # 1. 文件选择
        file_frame = tb.Frame(self, padding=10)
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        tb.Label(file_frame, text="Excel 文件:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_var = tk.StringVar()
        file_path_entry = tb.Entry(file_frame, textvariable=self.file_path_var, width=60)
        file_path_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        file_browse_button = tb.Button(file_frame, text="浏览", command=self.browse_file, bootstyle="info")
        file_browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # 2. 按钮 Frame
        button_frame = tb.Frame(self, padding=10)
        button_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        # 2.1 分析按钮
        self.analyze_button = tb.Button(button_frame, text="开始分析", command=self.start_analysis, bootstyle="success")
        self.analyze_button.grid(row=0, column=0, padx=5, pady=5)

        # 2.2 停止按钮
        self.stop_button = tb.Button(button_frame, text="停止运行", command=self.stop_analysis, state=tk.DISABLED, bootstyle="danger")
        self.stop_button.grid(row=0, column=1, padx=5, pady=5)

        # 3. 状态/结果显示
        self.result_label = tb.Label(self, text="请选择 Excel 文件并点击开始分析", wraplength=780, justify=tk.LEFT)
        self.result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        # 4. 进度条
        self.progress_bar = tb.Progressbar(self, orient=tk.HORIZONTAL, length=780, mode='determinate')
        self.progress_bar.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.progress_bar["maximum"] = self.analysis_steps  # 设置最大值
        self.progress_bar["value"] = 0  # 设置初始值

        # 5. 图表显示
        self.image_canvas = tb.Canvas(self, bg=self.grid_bg_color, highlightthickness=1)  # 使用 Canvas，设置背景
        self.image_canvas.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="news") # <--- 修改这里
        
         # 6. 保存结果按钮 (初始隐藏，在分析成功后显示)
        self.save_button = tb.Button(self, text="保存结果", command=self.save_output_file, state=tk.DISABLED, bootstyle="primary")
        self.save_button.grid(row=5, column=0, columnspan=2, pady=10)

        # 确保图表区域可以扩展
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)


    def draw_grid(self):
          """在 Canvas 上绘制白色网格"""
          canvas_width = self.image_canvas.winfo_width()
          canvas_height = self.image_canvas.winfo_height()

          if canvas_width > 0 and canvas_height > 0 : # 确保画布大小有效
            grid_size = 40  # 设置网格大小
            for x in range(grid_size, canvas_width, grid_size):
                self.image_canvas.create_line(x, 0, x, canvas_height, fill="lightgray", width = 1 ) # 使用浅灰色网格
            for y in range(grid_size, canvas_height, grid_size):
                self.image_canvas.create_line(0, y, canvas_width, y, fill="lightgray", width = 1) # 使用浅灰色网格


    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if filename:
            self.file_path_var.set(filename)

    def start_analysis(self):
        """初始化分析状态并开始分析线程"""
        if not self.file_path_var.get():
            self.show_message("错误", "请选择一个Excel文件")
            return

        # 初始化状态
        self.result_label.config(text="开始分析...", foreground="black")
        self.image_canvas.delete("all")  # 清空画布
        self.draw_grid() # 重绘白色网格
        
        self.image_canvas.create_text(self.image_canvas.winfo_width() / 2, self.image_canvas.winfo_height() / 2,
                                     text=self.initial_image_text, fill="gray", font = ('Arial', 14) , anchor="center") # 显示文字
        self.save_button.config(state=tk.DISABLED)
        self.progress_bar["value"] = 0
        self.current_step = 0

        # 禁用分析按钮，启用停止按钮
        self.analyze_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stop_event.clear()  # 清除停止标志

        # 启动分析线程
        self.analysis_thread = threading.Thread(target=self.analyze_data_thread)
        self.analysis_thread.start()


    def stop_analysis(self):
        """停止分析线程"""
        self.stop_event.set() # 设置停止标志
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(1)
        # 启用分析按钮，禁用停止按钮
        self.after(0, self.analyze_button.config, {"state": tk.NORMAL})
        self.after(0, self.stop_button.config, {"state": tk.DISABLED})
        self.after(0, self.result_label.config, {"text": "分析已停止", "foreground": "red"})
    
    def analyze_data_thread(self):
         """在线程中执行分析，防止GUI冻结"""
         try:
             # 调整输出路径，基于输入文件路径生成
            base_dir = Path(self.file_path_var.get()).parent
            output_dir = base_dir / "分析结果"
            
            # 初始化分析器
            self.analyzer = StressStrainAnalyzer(self.file_path_var.get())
            if self.stop_event.is_set():
               return  # 如果停止，则直接退出

            self.update_progress(1)  # 数据加载进度

            # 数据加载阶段
            self.update_status("加载数据...", "black")
            self.analyzer.load_data()
            if self.stop_event.is_set():
                return #  如果停止，则直接退出
            self.update_progress(2)  # 加载数据进度

            # 调用Kimi分析
            self.update_status("使用Kimi分析趋势...", "black")
            self.analyzer.analyze_with_kimi()
            if self.stop_event.is_set():
               return  # 如果停止，则直接退出
            print("Kimi 分析结果:", self.analyzer.kimi_analysis)
            self.update_progress(3)  # Kimi 分析进度

            # 构建预测模型
            self.update_status("构建预测模型...", "black")
            self.analyzer.build_prediction_models()
            if self.stop_event.is_set():
                 return  # 如果停止，则直接退出
            self.update_progress(4) # 模型构建进度

            # 生成扩展数据
            self.update_status("生成预测数据...", "black")
            extended_df = self.analyzer.generate_data()
            if self.stop_event.is_set():
               return #  如果停止，则直接退出
           
            # 保存结果
            self.update_status("保存结果...", "black")
            self.output_path, self.plot_path = self.analyzer.save_results(extended_df, output_dir)
            if self.stop_event.is_set():
               return  # 如果停止，则直接退出
            self.update_progress(5)  # 结果保存进度

             # 更新 GUI 显示最终结果
            self.after(0, self.show_analysis_results, "分析成功", "green")

         except Exception as e:
             self.after(0, self.show_analysis_results, f"分析失败: {str(e)}", "red")
         finally:
             # 分析结束时，启用分析按钮，禁用停止按钮
             self.after(0, self.analyze_button.config, {"state": tk.NORMAL})
             self.after(0, self.stop_button.config, {"state": tk.DISABLED})

    def update_status(self, message, color):
        """线程安全更新状态标签"""
        self.after(0, self.result_label.config, {"text": message, "foreground": color})

    def update_progress(self, step):
        """线程安全更新进度条"""
        self.current_step = step
        self.after(0, self.progress_bar.config, {"value": step})

    def show_analysis_results(self, message, color):
         """显示分析结果，包括文字信息和图像"""
         self.result_label.config(text=message, foreground=color)
         self.save_button.config(state=tk.NORMAL)  # 启用保存按钮

         if self.plot_path:
             try:
                  self.image_canvas.delete("all") # 清空画布
                  self.draw_grid() # 绘制网格

                  image = Image.open(self.plot_path)
                  image.thumbnail((600, 400))  # 确保图片在画布内
                  photo = ImageTk.PhotoImage(image)

                # 计算图像在画布中的中心位置
                  canvas_width = self.image_canvas.winfo_width()
                  canvas_height = self.image_canvas.winfo_height()
                  x = (canvas_width - photo.width()) // 2
                  y = (canvas_height - photo.height()) // 2

                  self.image_canvas.create_image(x, y, image=photo, anchor="nw") # 显示图像
                  self.image_canvas.image = photo # Keep a reference!
             except Exception as e:
                 self.show_message("错误", "图表显示失败")
                 self.result_label.config(text="图表加载失败", foreground="red")

    def save_output_file(self):
        """处理保存结果操作"""
        if not self.output_path:
            self.show_message("错误", "请先进行分析")
            return
        try:
            # 打开保存文件对话框
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                        filetypes=[("Excel files", "*.xlsx")])
            if file_path:
                # 读取已有的结果
                full_df = pd.read_excel(self.output_path)
                # 保存到用户选择的路径
                full_df.to_excel(file_path, index=False)
                self.show_message("保存成功", f"结果已保存至 {file_path}")
        except Exception as e:
            self.show_message("错误", f"保存失败: {str(e)}")

    def show_message(self, title, message):
        messagebox.showerror(title, message)


# ================= 主程序 =================
def main():
    try:
        app = StressStrainApp()                      #创建窗口
        app.mainloop()                                #响应窗口
    except KeyboardInterrupt:
        print("\n程序被中断，正在安全退出...")
    except Exception as e:
        print(f"\n程序运行出错：{e}")

if __name__ == "__main__":
    main()