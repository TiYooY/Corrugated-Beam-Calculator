# main.py

import customtkinter as ctk
import pandas as pd
import numpy as np
from elements import CorrugatedWebGirder
import traceback
from scipy.integrate import quad
from scipy.optimize import root_scalar
import sys  
import os   

def resource_path(relative_path: str) -> str:
    """
    获取资源的绝对路径, 解决打包后路径问题
    :param relative_path: 相对路径（如 'data/config.json'）
    :return: 绝对路径
    """
    if hasattr(sys, "_MEIPASS"):  # 打包后
        base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    else:  # 源码运行
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- 窗口基本设置 ---
        self.title("波纹腹板钢梁计算工具 v1.1")
        self.geometry("1000x750") # 增加了初始尺寸以容纳新布局
        self.minsize(900, 750)   # 设置最小窗口尺寸
        ctk.set_appearance_mode("System") # or "Light", "Dark"
        ctk.set_default_color_theme("blue")

        # --- 数据加载 ---
        try:
            csv_path = resource_path("sections_cwb.csv")
            self.db = pd.read_csv(csv_path)
            self.section_names = self.db["SectionName"].tolist()
        except FileNotFoundError:
            self.section_names = ["错误: 未找到 sections_cwb.csv"]
            self.db = None

        # --- 创建主框架 ---
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        main_frame.grid_columnconfigure(0, weight=1) # 配置列权重
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # --- 1. 模式选择区 ---
        mode_frame = ctk.CTkFrame(main_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.mode_var = ctk.StringVar(value="database")
        ctk.CTkLabel(mode_frame, text="选择模式:").pack(side="left", padx=10)
        ctk.CTkRadioButton(mode_frame, text="从数据库选择", variable=self.mode_var, value="database", command=self.on_mode_change).pack(side="left", padx=10)
        ctk.CTkRadioButton(mode_frame, text="手动输入参数", variable=self.mode_var, value="manual", command=self.on_mode_change).pack(side="left", padx=10)

        # --- 2. 参数输入区 (使用可滚动框架) ---
        input_scroll_frame = ctk.CTkScrollableFrame(main_frame, label_text="输入参数")
        input_scroll_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        input_scroll_frame.grid_columnconfigure(1, weight=1)


        ctk.CTkLabel(input_scroll_frame, text="截面数据库:").grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="w")
        self.section_dropdown = ctk.CTkComboBox(input_scroll_frame, width=250, values=self.section_names, command=self.on_section_select)
        self.section_dropdown.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        
        self.entries = {}
        # 拆分参数以插入波纹几何部分
        param_labels_part1 = {
            "hw": "截面深度 hw (mm)", 
            "tw": "腹板厚度 tw (mm)", 
            "bf": "翼缘宽度 bf (mm)",
            "tf": "翼缘厚度 tf (mm)", 
            "Fy": "钢材屈服强度 Fy (MPa)",
        }
        param_labels_part2 = {
            "E": "弹性模量 E (MPa)", 
            "G": "切变模量 G (MPa)", 
            "nu": "泊松比 ν",
            "phi_s": "抗剪折减系数 φs", 
            "phi_f": "抗弯折减系数 φf",
            "L": "跨度 L (mm)", 
            "omega2": "弯矩梯度系数 ω2", 
            "alpha_LT": "屈曲修正系数 α_LT"
        }

        # 创建第一部分输入框
        row_counter = 2
        for key, text in param_labels_part1.items():
            ctk.CTkLabel(input_scroll_frame, text=text).grid(row=row_counter, column=0, columnspan=2, padx=10, pady=5, sticky="w")
            entry = ctk.CTkEntry(input_scroll_frame, width=200)
            entry.grid(row=row_counter, column=2, padx=10, pady=5)
            self.entries[key] = entry
            row_counter += 1
            
        # --- 新增：波纹几何参数计算区 ---
        corrugation_frame = ctk.CTkFrame(input_scroll_frame)
        corrugation_frame.grid(row=row_counter, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        ctk.CTkLabel(corrugation_frame, text="波纹几何 (输入任意两者计算第三个)").grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        # a3
        ctk.CTkLabel(corrugation_frame, text="波纹高度 a3 (mm)").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entries['a3'] = ctk.CTkEntry(corrugation_frame)
        self.entries['a3'].grid(row=1, column=1, padx=5, pady=5)
        ctk.CTkButton(corrugation_frame, text="计算", width=50, command=lambda: self.calculate_geometric_param('a3')).grid(row=1, column=2, padx=5, pady=5)
        # s
        ctk.CTkLabel(corrugation_frame, text="半波纹曲线长 s (mm)").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entries['s'] = ctk.CTkEntry(corrugation_frame)
        self.entries['s'].grid(row=2, column=1, padx=5, pady=5)
        ctk.CTkButton(corrugation_frame, text="计算", width=50, command=lambda: self.calculate_geometric_param('s')).grid(row=2, column=2, padx=5, pady=5)
        # w
        ctk.CTkLabel(corrugation_frame, text="半波纹投影长 w (mm)").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entries['w'] = ctk.CTkEntry(corrugation_frame)
        self.entries['w'].grid(row=3, column=1, padx=5, pady=5)
        ctk.CTkButton(corrugation_frame, text="计算", width=50, command=lambda: self.calculate_geometric_param('w')).grid(row=3, column=2, padx=5, pady=5)
        row_counter += 1

        # 创建第二部分输入框
        for key, text in param_labels_part2.items():
            ctk.CTkLabel(input_scroll_frame, text=text).grid(row=row_counter, column=0, columnspan=2, padx=10, pady=5, sticky="w")
            entry = ctk.CTkEntry(input_scroll_frame, width=200)
            entry.grid(row=row_counter, column=2, padx=10, pady=5)
            self.entries[key] = entry
            row_counter += 1


        # --- 3. 计算选项区 ---
        calc_options_frame = ctk.CTkFrame(main_frame)
        calc_options_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkLabel(calc_options_frame, text="选择计算模型:").grid(row=0, column=0, padx=10, pady=5)
        self.calc_model_var = ctk.StringVar(value="EN剪力模型")
        self.model_dropdown = ctk.CTkComboBox(calc_options_frame, width=200, 
                                             values=["EN剪力模型", "EN弯曲模型", "CSA弯曲模型"], 
                                             variable=self.calc_model_var)
        self.model_dropdown.grid(row=0, column=1, padx=10, pady=5)

        # --- 4. 操作区 ---
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.calc_button = ctk.CTkButton(action_frame, text="计 算", command=self.on_calculate)
        self.calc_button.pack(side="left", padx=20, pady=10)
        self.clear_button = ctk.CTkButton(action_frame, text="清 除", command=self.clear_fields, fg_color="gray")
        self.clear_button.pack(side="left", padx=20, pady=10)

        # --- 5. 结果显示区 ---
        result_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        result_frame.grid(row=1, column=1, rowspan=3, padx=10, pady=10, sticky="nsew")
        result_frame.grid_rowconfigure(1, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(result_frame, text="计算结果").grid(row=0, column=0, pady=5, sticky="w")
        # 增加了字体大小
        self.result_textbox = ctk.CTkTextbox(result_frame, state="disabled", font=("Courier New", 16))
        self.result_textbox.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")

        # --- 初始化UI状态 ---
        self.on_mode_change()
        if self.section_names and self.section_names[0] and "错误" not in self.section_names[0]:
            self.section_dropdown.set(self.section_names[0])
            self.on_section_select(self.section_names[0])

    def _arc_length_integrand(self, x, a3, w):
        """
        The function inside the integral for calculating the arc length of the sine wave.
        y = (a3/2) * sin(pi*x/w)
        y' = (a3/2) * (pi/w) * cos(pi*x/w)
        Returns sqrt(1 + (y')^2)
        """
        # Avoid division by zero if w is somehow zero
        if w == 0:
            return 1.0
        
        term = (np.pi * a3) / (2 * w) * np.cos((np.pi * x) / w)
        return np.sqrt(1 + term**2)

    def _calculate_arc_length(self, a3, w):
        """
        Calculates the arc length 's' by performing numerical integration.
        Returns the calculated length and its estimated error.
        """
        # quad returns a tuple: (result, estimated_absolute_error)
        s_val, _ = quad(self._arc_length_integrand, 0, w, args=(a3, w))
        return s_val

    def calculate_geometric_param(self, target_param):
        """
        Calculates the target geometric parameter using high-precision numerical methods.
        - Integration for 's'.
        - Root-finding for 'a3' and 'w'.
        """
        try:
            a3_str = self.entries['a3'].get()
            s_str = self.entries['s'].get()
            w_str = self.entries['w'].get()

            a3 = float(a3_str) if a3_str else None
            s = float(s_str) if s_str else None
            w = float(w_str) if w_str else None

            if target_param == 's':
                if a3 is not None and w is not None:
                    val = self._calculate_arc_length(a3, w)
                    self.set_entry_text(self.entries['s'], f"{val:.4f}")
                else:
                    self.display_error("要计算 s, 请先输入 a3 和 w。")

            elif target_param == 'a3':
                if s is not None and w is not None:
                    if s < w:
                        self.display_error("输入错误: 波纹曲线长(s)不能小于投影长(w)。")
                        return

                    # Objective function: We want to find an 'a3' that makes this function zero.
                    def objective_a3(a3_guess, s_target, w_known):
                        return self._calculate_arc_length(a3_guess, w_known) - s_target

                    # Bracket for the search: a3 must be between 0 and some upper limit.
                    # A safe upper limit is derived from the approximation.
                    upper_bound = (4 * w / np.pi) * np.sqrt(s / w + 1) # Generous upper bound
                    sol = root_scalar(objective_a3, args=(s, w), method='brentq', bracket=[0, upper_bound])
                    
                    if sol.converged:
                        self.set_entry_text(self.entries['a3'], f"{sol.root:.4f}")
                    else:
                        self.display_error("计算 a3 未能收敛。请检查输入值。")
                else:
                    self.display_error("要计算 a3, 请先输入 s 和 w。")

            elif target_param == 'w':
                if a3 is not None and s is not None:
                    # Minimum possible 'w' is when the wave is a vertical line, s = a3. 
                    # But the wave is sinusoidal, so w must be > 0.
                    # A safe lower bound for w is derived from physical constraints (s must be > pi*a3/2)
                    min_w = 0.01 
                    if s < min_w:
                         self.display_error("输入错误: 曲线长(s)过小。")
                         return

                    # Objective function: We want to find a 'w' that makes this function zero.
                    def objective_w(w_guess, s_target, a3_known):
                        return self._calculate_arc_length(a3_known, w_guess) - s_target
                    
                    # Bracket for the search: w must be between some small value and s.
                    sol = root_scalar(objective_w, args=(s, a3), method='brentq', bracket=[min_w, s])
                    
                    if sol.converged:
                        self.set_entry_text(self.entries['w'], f"{sol.root:.4f}")
                    else:
                        self.display_error("计算 w 未能收敛。请检查输入值。")
                else:
                    self.display_error("要计算 w, 请先输入 a3 和 s。")

        except (ValueError, TypeError):
            self.display_error("参数计算错误: 请确保输入了有效的数字。")
        except Exception as e:
            self.display_error(f"发生未知错误: {e}\n{traceback.format_exc()}")


    def on_mode_change(self):
        """根据选择的模式，启用/禁用相应的控件。"""
        is_db_mode = (self.mode_var.get() == "database")
        
        self.section_dropdown.configure(state="normal" if is_db_mode else "disabled")
        for key, entry in self.entries.items():
            entry.configure(state="disabled" if is_db_mode else "normal")

    def on_section_select(self, selected_section):
        """当从数据库下拉菜单中选择一个截面时，自动填充参数。"""
        if self.db is None:
            self.display_error("数据库未加载，请检查 sections_cwb.csv 文件")
            return
            
        try:
            section_data = self.db[self.db["SectionName"] == selected_section].iloc[0]
            
            for key in self.entries.keys():
                if key in section_data.index:
                    self.set_entry_text(self.entries[key], section_data[key])
                else:
                    default_values = {
                        "E": 200000, "G": 77000, "nu": 0.3,
                        "phi_s": 0.9, "phi_f": 0.9, "omega2": 1.0,
                        "alpha_LT": 0.34, "L": 6000
                    }
                    if key in default_values:
                        self.set_entry_text(self.entries[key], default_values[key])
                    else:
                        # Clear fields that are not in the defaults or the CSV
                        self.set_entry_text(self.entries[key], "")
                        
        except Exception as e:
            self.display_error(f"读取截面数据时出错: {e}")

    def set_entry_text(self, entry, text):
        """一个辅助函数，用于设置输入框的文本。"""
        is_db_mode = self.mode_var.get() == 'database'
        current_state = "disabled" if is_db_mode else "normal"
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, str(text))
        entry.configure(state=current_state)

    def clear_fields(self):
        """清除所有输入框和结果。"""
        if self.mode_var.get() == 'manual':
            for entry in self.entries.values():
                entry.delete(0, "end")
        
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.configure(state="disabled")

    def on_calculate(self):
        """点击"计算"按钮时触发的核心函数。"""
        try:
            missing_params = []
            invalid_params = []
            
            for key, entry in self.entries.items():
                value = entry.get().strip()
                if not value:
                    missing_params.append(key)
                else:
                    try:
                        float(value)
                    except ValueError:
                        invalid_params.append(f"{key} (当前值: '{value}')")
            
            if missing_params:
                self.display_error(f"以下参数缺失:\n{', '.join(missing_params)}")
                return
            if invalid_params:
                self.display_error(f"以下参数不是有效数字:\n{', '.join(invalid_params)}")
                return

            params = {key: entry.get() for key, entry in self.entries.items()}
            calc_model = self.calc_model_var.get()

            beam = CorrugatedWebGirder(**params)
            results = beam.calculate(method=calc_model)

            self.result_textbox.configure(state="normal")
            self.result_textbox.delete("1.0", "end")
            
            result_text = f"--- 输入参数 ---\n"
            # 获取所有有效的参数用于显示
            params_to_display = {key: getattr(beam, key) for key in self.entries.keys() if hasattr(beam, key)}

            # 循环并格式化每个参数，使其独占一行
            for key, value in params_to_display.items():
                # 使用 f-string 格式化：key 左对齐占12个字符，value 保留2位小数
                result_text += f"{key:<12}: {value:.2f}\n"

            result_text += "\n--- 计算结果 ---\n"
            result_text += f"计算模型: {calc_model}\n"
            result_text += "--------------------\n"
            
            shear_cap = results.get('shear_capacity')
            moment_cap = results.get('moment_capacity')

            if shear_cap is not None:
                result_text += f"抗剪承载力 (Vr): {shear_cap:.2f} kN\n"
            if moment_cap is not None:
                result_text += f"抗弯承载力 (Mr): {moment_cap:.2f} kN·m\n"

            self.result_textbox.insert("1.0", result_text)
            self.result_textbox.configure(state="disabled")

        except ValueError as e:
            self.display_error(f"输入错误: 请确保所有参数均为有效数字。\n详细信息: {e}")
        except Exception as e:
            self.display_error(f"计算错误:\n{str(e)}\n\n详细跟踪:\n{traceback.format_exc()}")

    def display_error(self, message):
        """在结果区显示红色的错误信息。"""
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.insert("1.0", f"!!! 错误 !!!\n\n{message}")
        # You can add a tag to make the text red
        self.result_textbox.tag_add("error", "1.0", "1.12")
        self.result_textbox.tag_config("error", foreground="red", font=("Courier New", 16, "bold"))
        self.result_textbox.configure(state="disabled")

if __name__ == "__main__":
    app = App()
    app.mainloop()