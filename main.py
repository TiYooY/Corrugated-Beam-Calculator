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

# 语言字典
LANGUAGES = {
    'zh': {
        # 窗口标题
        'window_title': '波纹腹板钢梁计算工具 v1.4',
        
        # 模式选择
        'select_mode': '选择模式:',
        'from_database': '从数据库选择',
        'manual_input': '手动输入参数',
        
        # 数据库选择
        'input_params': '输入参数',
        'step_select': '分步选择截面:',
        'web_height': '1. 腹板高度 (mm):',
        'web_thickness': '腹板厚度 (mm):',
        'flange_width': '2. 翼缘宽度 (mm):',
        'flange_thickness': '翼缘厚度 (mm):',
        'select_section': '3. 选择截面:',
        'reset_selection': '重置选择',
        'placeholder_select': '--请选择--',
        
        # 参数标签
        'section_depth': '截面深度 hw (mm)',
        'web_thick': '腹板厚度 tw (mm)',
        'flange_w': '翼缘宽度 bf (mm)',
        'flange_t': '翼缘厚度 tf (mm)',
        'yield_strength': '钢材屈服强度 Fy (MPa)',
        'elastic_modulus': '弹性模量 E (MPa)',
        'shear_modulus': '切变模量 G (MPa)',
        'poisson_ratio': '泊松比 ν',
        'shear_reduction': '抗剪折减系数 φs',
        'flexural_reduction': '抗弯折减系数 φf',
        'span_length': '跨度 L (mm)',
        'moment_gradient': '弯矩梯度系数 ω2',
        'buckling_factor': '屈曲修正系数 α_LT',
        
        # 波纹几何
        'corrugation_geometry': '波纹几何 (输入任意两者计算第三个)',
        'corrugation_height': '波纹高度 a3 (mm)',
        'half_wave_length': '半波纹曲线长 s (mm)',
        'half_wave_projection': '半波纹投影长 w (mm)',
        'calculate': '计算',
        
        # 计算选项
        'select_model': '选择计算模型:',
        'en_shear': 'EN剪力模型',
        'en_flexure': 'EN弯曲模型',
        'csa_flexure': 'CSA弯曲模型',
        
        # 按钮
        'calc_button': '计 算',
        'clear_button': '清 除',
        
        # 结果显示
        'calc_results': '计算结果',
        'input_params_section': '--- 输入参数 ---',
        'calc_results_section': '--- 计算结果 ---',
        'calc_model': '计算模型:',
        'shear_capacity': '抗剪承载力 (Vr):',
        'moment_capacity': '抗弯承载力 (Mr):',
        'failure_mode': '失效模式:',
        'detailed_info': '--- 详细计算信息 ---',
        
        # EN剪力模型详细信息
        'local_capacity': '局部屈曲承载力:',
        'global_capacity': '整体屈曲承载力:',
        'local_stress': '局部临界应力:',
        'global_stress': '整体临界应力:',
        'local_slenderness': '局部长细比:',
        'global_slenderness': '整体长细比:',
        'local_reduction': '局部折减系数:',
        'global_reduction': '整体折减系数:',
        'fm_local_shear_buckling': '局部剪切屈曲控制',
        'fm_global_shear_buckling': '整体剪切屈曲控制',
        
        # EN/CSA弯曲模型详细信息
        'slenderness_ratio': '长细比',
        'ltb_limit': 'LTB界限长细比:',
        'ltb_occurs': '发生侧扭屈曲:',
        'critical_moment': '弹性临界弯矩:',
        'elastic_critical': '临界弯矩:',
        'ltb_reduction': 'LTB折减系数:',
        'elastic_section': '弹性截面模量:',
        'plastic_section': '塑性截面模量:',
        'buckling_type': '屈曲类型:',
        'plastic_moment': '塑性弯矩:',
        'buckling_limit': '屈曲界限(0.67Mp):',
        'fm_section_no_ltb': '截面强度控制（无LTB影响）',
        'fm_section_minor_ltb': '截面强度控制（轻微LTB影响）',
        'fm_ltb_control': '侧扭屈曲控制',
        'fm_inelastic_buckling': '非弹性屈曲控制', 'bt_inelastic_buckling': '非弹性屈曲',
        'fm_elastic_buckling': '弹性屈曲控制', 'bt_elastic_buckling': '弹性屈曲',

        # 错误消息
        'error': '错误',
        'error_header': '!!! 错误 !!!',
        'file_not_found': '错误: 未找到 sections_cwb.csv',
        'calc_s_error': '要计算 s, 请先输入 a3 和 w。',
        'calc_a3_error': '要计算 a3, 请先输入 s 和 w。',
        'calc_w_error': '要计算 w, 请先输入 a3 和 s。',
        'input_error': '输入错误:',
        'curve_length_error': '波纹曲线长(s)不能小于投影长(w)。',
        'curve_too_small': '曲线长(s)过小。',
        'calc_not_converged': '计算未能收敛。请检查输入值。',
        'param_calc_error': '参数计算错误: 请确保输入了有效的数字。',
        'unknown_error': '发生未知错误:',
        'missing_params': '以下参数缺失:',
        'invalid_params': '以下参数不是有效数字:',
        'current_value': '当前值:',
        'ensure_valid': '输入错误: 请确保所有参数均为有效数字。',
        'detailed_info_label': '详细信息:',
        'detailed_trace': '详细跟踪:',
        'calc_error': '计算错误:',
        
        # 其他
        'yes': '是',
        'no': '否',
        'units_kn': 'kN',
        'units_knm': 'kN·m',
        'units_mpa': 'MPa',
        'units_cm3': 'cm³',
        
        # 选择相关错误
        'init_dropdown_error': '初始化下拉框时出错:',
        'select_hw_error': '选择腹板高度时出错:',
        'select_tw_error': '选择腹板厚度时出错:',
        'select_bf_error': '选择翼缘宽度时出错:',
        'select_tf_error': '选择翼缘厚度时出错:',
        'read_section_error': '读取截面数据时出错:',
    },
    'en': {
        # Window title
        'window_title': 'Corrugated Web Girder Calculator v1.3',
        
        # Mode selection
        'select_mode': 'Select Mode:',
        'from_database': 'Select from Database',
        'manual_input': 'Manual Input',
        
        # Database selection
        'input_params': 'Input Parameters',
        'step_select': 'Step-by-step Section Selection:',
        'web_height': '1. Web Height (mm):',
        'web_thickness': 'Web Thickness (mm):',
        'flange_width': '2. Flange Width (mm):',
        'flange_thickness': 'Flange Thickness (mm):',
        'select_section': '3. Select Section:',
        'reset_selection': 'Reset',
        'placeholder_select': '--Select--',
        
        # Parameter labels
        'section_depth': 'Section Depth hw (mm)',
        'web_thick': 'Web Thickness tw (mm)',
        'flange_w': 'Flange Width bf (mm)',
        'flange_t': 'Flange Thickness tf (mm)',
        'yield_strength': 'Steel Yield Strength Fy (MPa)',
        'elastic_modulus': 'Elastic Modulus E (MPa)',
        'shear_modulus': 'Shear Modulus G (MPa)',
        'poisson_ratio': 'Poisson\'s Ratio ν',
        'shear_reduction': 'Shear Reduction Factor φs',
        'flexural_reduction': 'Flexural Reduction Factor φf',
        'span_length': 'Span Length L (mm)',
        'moment_gradient': 'Moment Gradient Factor ω2',
        'buckling_factor': 'Buckling Correction Factor α_LT',
        
        # Corrugation geometry
        'corrugation_geometry': 'Corrugation Geometry (input any two to calculate the third)',
        'corrugation_height': 'Corrugation Height a3 (mm)',
        'half_wave_length': 'Half-wave Arc Length s (mm)',
        'half_wave_projection': 'Half-wave Projection w (mm)',
        'calculate': 'Calculate',
        
        # Calculation options
        'select_model': 'Select Calculation Model:',
        'en_shear': 'EN Shear Model',
        'en_flexure': 'EN Flexural Model',
        'csa_flexure': 'CSA Flexural Model',
        
        # Buttons
        'calc_button': 'Calculate',
        'clear_button': 'Clear',
        
        # Results display
        'calc_results': 'Calculation Results',
        'input_params_section': '--- Input Parameters ---',
        'calc_results_section': '--- Calculation Results ---',
        'calc_model': 'Calculation Model:',
        'shear_capacity': 'Shear Capacity (Vr):',
        'moment_capacity': 'Moment Capacity (Mr):',
        'failure_mode': 'Failure Mode:',
        'detailed_info': '--- Detailed Calculation Info ---',
        
        # EN Shear model details
        'local_capacity': 'Local Buckling Capacity:',
        'global_capacity': 'Global Buckling Capacity:',
        'local_stress': 'Local Critical Stress:',
        'global_stress': 'Global Critical Stress:',
        'local_slenderness': 'Local Slenderness:',
        'global_slenderness': 'Global Slenderness:',
        'local_reduction': 'Local Reduction Factor:',
        'global_reduction': 'Global Reduction Factor:',
        'fm_local_shear_buckling': 'Local Shear Buckling Control',
        'fm_global_shear_buckling': 'Global Shear Buckling Control',

        # EN/CSA Flexural model details
        'slenderness_ratio': 'Slenderness Ratio',
        'ltb_limit': 'LTB Limit Slenderness:',
        'ltb_occurs': 'LTB Occurs:',
        'critical_moment': 'Elastic Critical Moment:',
        'elastic_critical': 'Critical Moment:',
        'ltb_reduction': 'LTB Reduction Factor:',
        'elastic_section': 'Elastic Section Modulus:',
        'plastic_section': 'Plastic Section Modulus:',
        'buckling_type': 'Buckling Type:',
        'plastic_moment': 'Plastic Moment:',
        'buckling_limit': 'Buckling Limit (0.67Mp):',
        'fm_section_no_ltb': 'Section Strength Control (No LTB Effect)',
        'fm_section_minor_ltb': 'Section Strength Control (Minor LTB Effect)',
        'fm_ltb_control': 'Lateral-Torsional Buckling Control',
        'fm_inelastic_buckling': 'Inelastic Buckling Control', 'bt_inelastic_buckling': 'Inelastic Buckling',
        'fm_elastic_buckling': 'Elastic Buckling Control', 'bt_elastic_buckling': 'Elastic Buckling',

        # Error messages
        'error': 'Error',
        'error_header': '!!! ERROR !!!',
        'file_not_found': 'Error: sections_cwb.csv not found',
        'calc_s_error': 'To calculate s, please input a3 and w first.',
        'calc_a3_error': 'To calculate a3, please input s and w first.',
        'calc_w_error': 'To calculate w, please input a3 and s first.',
        'input_error': 'Input Error:',
        'curve_length_error': 'Arc length (s) cannot be less than projection (w).',
        'curve_too_small': 'Arc length (s) is too small.',
        'calc_not_converged': 'Calculation did not converge. Please check input values.',
        'param_calc_error': 'Parameter calculation error: Please ensure valid numbers are entered.',
        'unknown_error': 'Unknown error occurred:',
        'missing_params': 'Missing parameters:',
        'invalid_params': 'Invalid parameters (not valid numbers):',
        'current_value': 'Current value:',
        'ensure_valid': 'Input error: Please ensure all parameters are valid numbers.',
        'detailed_info_label': 'Details:',
        'detailed_trace': 'Detailed trace:',
        'calc_error': 'Calculation Error:',
        
        # Others
        'yes': 'Yes',
        'no': 'No',
        'units_kn': 'kN',
        'units_knm': 'kN·m',
        'units_mpa': 'MPa',
        'units_cm3': 'cm³',
        
        # Selection errors
        'init_dropdown_error': 'Error initializing dropdowns:',
        'select_hw_error': 'Error selecting web height:',
        'select_tw_error': 'Error selecting web thickness:',
        'select_bf_error': 'Error selecting flange width:',
        'select_tf_error': 'Error selecting flange thickness:',
        'read_section_error': 'Error reading section data:',
    }
}

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
        
        # 初始化语言（默认中文）
        self.current_lang = 'zh'
        self.lang = LANGUAGES[self.current_lang]

        # --- 窗口基本设置 ---
        self.title(self.lang['window_title'])
        self.geometry("1200x800")
        self.minsize(1100, 800)
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # --- 数据加载 ---
        try:
            csv_path = resource_path("sections_cwb.csv")
            self.db = pd.read_csv(csv_path)
            self.section_names = self.db["SectionName"].tolist()
        except FileNotFoundError:
            self.section_names = [self.lang['file_not_found']]
            self.db = None

        # 创建UI
        self.create_ui()
        
        # 初始化UI状态
        self.initialize_dropdowns()
        self.geometric_params = {'hw', 'tw', 'bf', 'tf', 'a3', 's', 'w'}
        self.on_mode_change()

    def create_ui(self):
        """创建用户界面"""
        # --- 创建主框架 ---
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)

        # --- 0. 语言切换区 ---
        lang_frame = ctk.CTkFrame(self.main_frame)
        lang_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="ew")
        
        ctk.CTkLabel(lang_frame, text="Language / 语言:").pack(side="left", padx=10)
        self.lang_var = ctk.StringVar(value="中文")
        lang_menu = ctk.CTkSegmentedButton(
            lang_frame,
            values=["中文", "English"],
            variable=self.lang_var,
            command=self.on_language_change
        )
        lang_menu.pack(side="left", padx=10)

        # --- 1. 模式选择区 ---
        self.mode_frame = ctk.CTkFrame(self.main_frame)
        self.mode_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        self.mode_var = ctk.StringVar(value="database")
        self.mode_label = ctk.CTkLabel(self.mode_frame, text=self.lang['select_mode'])
        self.mode_label.pack(side="left", padx=10)
        
        self.radio_db = ctk.CTkRadioButton(
            self.mode_frame, 
            text=self.lang['from_database'], 
            variable=self.mode_var, 
            value="database", 
            command=self.on_mode_change
        )
        self.radio_db.pack(side="left", padx=10)
        
        self.radio_manual = ctk.CTkRadioButton(
            self.mode_frame, 
            text=self.lang['manual_input'], 
            variable=self.mode_var, 
            value="manual", 
            command=self.on_mode_change
        )
        self.radio_manual.pack(side="left", padx=10)

        # --- 2. 参数输入区 ---
        self.input_scroll_frame = ctk.CTkScrollableFrame(
            self.main_frame, 
            label_text=self.lang['input_params']
        )
        self.input_scroll_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.input_scroll_frame.grid_columnconfigure(1, weight=1)

        # --- 2.1 数据库选择区域 ---
        self.database_frame = ctk.CTkFrame(self.input_scroll_frame)
        self.database_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        self.database_frame.grid_columnconfigure(1, weight=1)
        self.database_frame.grid_columnconfigure(3, weight=1)
        
        self.db_title_label = ctk.CTkLabel(
            self.database_frame, 
            text=self.lang['step_select'], 
            font=ctk.CTkFont(weight="bold")
        )
        self.db_title_label.grid(row=0, column=0, columnspan=4, padx=5, pady=(5,10), sticky="w")
        
        # 第一步：选择腹板参数
        self.hw_label = ctk.CTkLabel(self.database_frame, text=self.lang['web_height'])
        self.hw_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.hw_dropdown = ctk.CTkComboBox(self.database_frame, width=120, command=self.on_hw_select)
        self.hw_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.tw_label = ctk.CTkLabel(self.database_frame, text=self.lang['web_thickness'])
        self.tw_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.tw_dropdown = ctk.CTkComboBox(self.database_frame, width=120, command=self.on_tw_select)
        self.tw_dropdown.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        
        # 第二步：选择翼缘参数
        self.bf_label = ctk.CTkLabel(self.database_frame, text=self.lang['flange_width'])
        self.bf_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.bf_dropdown = ctk.CTkComboBox(self.database_frame, width=120, command=self.on_bf_select)
        self.bf_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        self.tf_label = ctk.CTkLabel(self.database_frame, text=self.lang['flange_thickness'])
        self.tf_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.tf_dropdown = ctk.CTkComboBox(self.database_frame, width=120, command=self.on_tf_select)
        self.tf_dropdown.grid(row=2, column=3, padx=5, pady=5, sticky="w")
        
        # 第三步：最终选择截面
        self.section_label = ctk.CTkLabel(self.database_frame, text=self.lang['select_section'])
        self.section_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.final_section_dropdown = ctk.CTkComboBox(
            self.database_frame, 
            width=250, 
            command=self.on_final_section_select
        )
        self.final_section_dropdown.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # 重置按钮
        self.reset_button = ctk.CTkButton(
            self.database_frame, 
            text=self.lang['reset_selection'], 
            width=80, 
            command=self.reset_selection
        )
        self.reset_button.grid(row=3, column=3, padx=5, pady=5, sticky="e")

        # --- 2.2 手动参数输入区域 ---
        self.entries = {}
        self.entry_labels = {}
        
        # 参数定义
        param_keys_part1 = ['hw', 'tw', 'bf', 'tf', 'Fy']
        param_keys_part2 = ['E', 'G', 'nu', 'phi_s', 'phi_f', 'L', 'omega2', 'alpha_LT']
        
        # 参数标签映射
        self.param_label_keys = {
            'hw': 'section_depth',
            'tw': 'web_thick',
            'bf': 'flange_w',
            'tf': 'flange_t',
            'Fy': 'yield_strength',
            'E': 'elastic_modulus',
            'G': 'shear_modulus',
            'nu': 'poisson_ratio',
            'phi_s': 'shear_reduction',
            'phi_f': 'flexural_reduction',
            'L': 'span_length',
            'omega2': 'moment_gradient',
            'alpha_LT': 'buckling_factor'
        }

        # 创建第一部分输入框
        row_counter = 2
        for key in param_keys_part1:
            label = ctk.CTkLabel(
                self.input_scroll_frame, 
                text=self.lang[self.param_label_keys[key]]
            )
            label.grid(row=row_counter, column=0, columnspan=2, padx=10, pady=5, sticky="w")
            self.entry_labels[key] = label
            
            entry = ctk.CTkEntry(self.input_scroll_frame, width=200)
            entry.grid(row=row_counter, column=2, padx=10, pady=5)
            self.entries[key] = entry
            row_counter += 1
            
        # --- 波纹几何参数计算区 ---
        self.corrugation_frame = ctk.CTkFrame(self.input_scroll_frame)
        self.corrugation_frame.grid(row=row_counter, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        
        self.corr_title = ctk.CTkLabel(
            self.corrugation_frame, 
            text=self.lang['corrugation_geometry']
        )
        self.corr_title.grid(row=0, column=0, columnspan=3, pady=(0, 5))
        
        # a3
        self.a3_label = ctk.CTkLabel(self.corrugation_frame, text=self.lang['corrugation_height'])
        self.a3_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.entries['a3'] = ctk.CTkEntry(self.corrugation_frame)
        self.entries['a3'].grid(row=1, column=1, padx=5, pady=5)
        self.a3_calc_btn = ctk.CTkButton(
            self.corrugation_frame, 
            text=self.lang['calculate'], 
            width=50, 
            command=lambda: self.calculate_geometric_param('a3')
        )
        self.a3_calc_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # s
        self.s_label = ctk.CTkLabel(self.corrugation_frame, text=self.lang['half_wave_length'])
        self.s_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.entries['s'] = ctk.CTkEntry(self.corrugation_frame)
        self.entries['s'].grid(row=2, column=1, padx=5, pady=5)
        self.s_calc_btn = ctk.CTkButton(
            self.corrugation_frame, 
            text=self.lang['calculate'], 
            width=50, 
            command=lambda: self.calculate_geometric_param('s')
        )
        self.s_calc_btn.grid(row=2, column=2, padx=5, pady=5)
        
        # w
        self.w_label = ctk.CTkLabel(self.corrugation_frame, text=self.lang['half_wave_projection'])
        self.w_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.entries['w'] = ctk.CTkEntry(self.corrugation_frame)
        self.entries['w'].grid(row=3, column=1, padx=5, pady=5)
        self.w_calc_btn = ctk.CTkButton(
            self.corrugation_frame, 
            text=self.lang['calculate'], 
            width=50, 
            command=lambda: self.calculate_geometric_param('w')
        )
        self.w_calc_btn.grid(row=3, column=2, padx=5, pady=5)
        row_counter += 1

        # 创建第二部分输入框
        for key in param_keys_part2:
            label = ctk.CTkLabel(
                self.input_scroll_frame, 
                text=self.lang[self.param_label_keys[key]]
            )
            label.grid(row=row_counter, column=0, columnspan=2, padx=10, pady=5, sticky="w")
            self.entry_labels[key] = label
            
            entry = ctk.CTkEntry(self.input_scroll_frame, width=200)
            entry.grid(row=row_counter, column=2, padx=10, pady=5)
            self.entries[key] = entry
            row_counter += 1

        # --- 3. 计算选项区 ---
        self.calc_options_frame = ctk.CTkFrame(self.main_frame)
        self.calc_options_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        self.model_label = ctk.CTkLabel(self.calc_options_frame, text=self.lang['select_model'])
        self.model_label.grid(row=0, column=0, padx=10, pady=5)
        
        self.calc_model_var = ctk.StringVar(value=self.lang['en_shear'])
        self.model_dropdown = ctk.CTkComboBox(
            self.calc_options_frame, 
            width=200,
            values=[self.lang['en_shear'], self.lang['en_flexure'], self.lang['csa_flexure']],
            variable=self.calc_model_var
        )
        self.model_dropdown.grid(row=0, column=1, padx=10, pady=5)

        # --- 4. 操作区 ---
        self.action_frame = ctk.CTkFrame(self.main_frame)
        self.action_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

        self.calc_button = ctk.CTkButton(
            self.action_frame, 
            text=self.lang['calc_button'], 
            command=self.on_calculate
        )
        self.calc_button.pack(side="left", padx=20, pady=10)
        
        self.clear_button = ctk.CTkButton(
            self.action_frame, 
            text=self.lang['clear_button'], 
            command=self.clear_fields, 
            fg_color="gray"
        )
        self.clear_button.pack(side="left", padx=20, pady=10)


        # --- 5. 结果显示区 ---
        result_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        # 调整了rowspan以更好地匹配左侧布局
        result_frame.grid(row=1, column=1, rowspan=4, padx=10, pady=10, sticky="nsew")
        result_frame.grid_rowconfigure(1, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)
        
        # 为标题标签命名，以便后续更新文本
        self.result_title_label = ctk.CTkLabel(result_frame, text=self.lang['calc_results'], font=ctk.CTkFont(weight="bold"))
        self.result_title_label.grid(row=0, column=0, pady=5, sticky="w")
        
        # 调整了字体大小
        self.result_textbox = ctk.CTkTextbox(result_frame, state="disabled", font=("Courier New", 14))
        self.result_textbox.grid(row=1, column=0, padx=0, pady=5, sticky="nsew")

    def initialize_dropdowns(self):
        """初始化分步选择的下拉框"""
        if self.db is None:
            return
            
        try:
            # 1. 设置腹板高度 (hw) - 这一部分逻辑正确，保持不变
            hw_values = [self.lang['placeholder_select']] + [str(int(val)) for val in sorted(self.db['hw'].unique())]
            self.hw_dropdown.configure(values=hw_values)
            self.hw_dropdown.set(self.lang['placeholder_select'])
            
            # 2. 设置其他被禁用的下拉框 (tw, bf, tf, final_section)
            placeholder_list = [self.lang['placeholder_select']]
            
            # 对每一个待选的下拉框进行更稳健的初始化
            # 先配置选项和默认值，最后再禁用，确保渲染正确
            
            self.tw_dropdown.configure(values=placeholder_list)
            self.tw_dropdown.set(placeholder_list[0])
            self.tw_dropdown.configure(state="disabled")

            self.bf_dropdown.configure(values=placeholder_list)
            self.bf_dropdown.set(placeholder_list[0])
            self.bf_dropdown.configure(state="disabled")

            self.tf_dropdown.configure(values=placeholder_list)
            self.tf_dropdown.set(placeholder_list[0])
            self.tf_dropdown.configure(state="disabled")
            
            self.final_section_dropdown.configure(values=placeholder_list)
            self.final_section_dropdown.set(placeholder_list[0])
            self.final_section_dropdown.configure(state="disabled")
            
        except Exception as e:
            self.display_error(f"{self.lang['init_dropdown_error']} {e}")

    def on_hw_select(self, selected_hw):
        """选择腹板高度后，更新腹板厚度选项"""
        placeholder_list = [self.lang['placeholder_select']]
        if selected_hw == placeholder_list[0]:
            self.tw_dropdown.configure(values=placeholder_list, state="disabled"); self.tw_dropdown.set(placeholder_list[0])
            self.bf_dropdown.configure(values=placeholder_list, state="disabled"); self.bf_dropdown.set(placeholder_list[0])
            self.tf_dropdown.configure(values=placeholder_list, state="disabled"); self.tf_dropdown.set(placeholder_list[0])
            self.final_section_dropdown.configure(values=placeholder_list, state="disabled"); self.final_section_dropdown.set(placeholder_list[0])
            return

        if self.db is None: return
        try:
            hw_value = float(selected_hw)
            filtered_db = self.db[self.db['hw'] == hw_value]
            tw_values = placeholder_list + [str(val) for val in sorted(filtered_db['tw'].unique())]
            self.tw_dropdown.configure(values=tw_values, state="normal")
            self.tw_dropdown.set(placeholder_list[0])
            
            self.bf_dropdown.configure(values=placeholder_list, state="disabled"); self.bf_dropdown.set(placeholder_list[0])
            self.tf_dropdown.configure(values=placeholder_list, state="disabled"); self.tf_dropdown.set(placeholder_list[0])
            self.final_section_dropdown.configure(values=placeholder_list, state="disabled"); self.final_section_dropdown.set(placeholder_list[0])
        except Exception as e:
            self.display_error(f"{self.lang['select_hw_error']} {e}")

    def on_tw_select(self, selected_tw):
        """选择腹板厚度后，更新翼缘宽度选项"""
        placeholder_list = [self.lang['placeholder_select']]
        if selected_tw == placeholder_list[0]:
            self.bf_dropdown.configure(values=placeholder_list, state="disabled"); self.bf_dropdown.set(placeholder_list[0])
            self.tf_dropdown.configure(values=placeholder_list, state="disabled"); self.tf_dropdown.set(placeholder_list[0])
            self.final_section_dropdown.configure(values=placeholder_list, state="disabled"); self.final_section_dropdown.set(placeholder_list[0])
            return

        if self.db is None: return
        try:
            hw_value = float(self.hw_dropdown.get())
            tw_value = float(selected_tw)
            filtered_db = self.db[(self.db['hw'] == hw_value) & (self.db['tw'] == tw_value)]
            bf_values = placeholder_list + [str(int(val)) for val in sorted(filtered_db['bf'].unique())]
            self.bf_dropdown.configure(values=bf_values, state="normal")
            self.bf_dropdown.set(placeholder_list[0])
            
            self.tf_dropdown.configure(values=placeholder_list, state="disabled"); self.tf_dropdown.set(placeholder_list[0])
            self.final_section_dropdown.configure(values=placeholder_list, state="disabled"); self.final_section_dropdown.set(placeholder_list[0])
        except Exception as e:
            self.display_error(f"{self.lang['select_tw_error']} {e}")

    def on_bf_select(self, selected_bf):
        """选择翼缘宽度后，更新翼缘厚度选项"""
        placeholder_list = [self.lang['placeholder_select']]
        if selected_bf == placeholder_list[0]:
            self.tf_dropdown.configure(values=placeholder_list, state="disabled"); self.tf_dropdown.set(placeholder_list[0])
            self.final_section_dropdown.configure(values=placeholder_list, state="disabled"); self.final_section_dropdown.set(placeholder_list[0])
            return

        if self.db is None: return
        try:
            hw_value = float(self.hw_dropdown.get())
            tw_value = float(self.tw_dropdown.get())
            bf_value = float(selected_bf)
            filtered_db = self.db[(self.db['hw'] == hw_value) & (self.db['tw'] == tw_value) & (self.db['bf'] == bf_value)]
            tf_values = placeholder_list + [str(val) for val in sorted(filtered_db['tf'].unique())]
            self.tf_dropdown.configure(values=tf_values, state="normal")
            self.tf_dropdown.set(placeholder_list[0])
            
            self.final_section_dropdown.configure(values=placeholder_list, state="disabled")
            self.final_section_dropdown.set(placeholder_list[0])
        except Exception as e:
            self.display_error(f"{self.lang['select_bf_error']} {e}")

    def on_tf_select(self, selected_tf):
        """选择翼缘厚度后，更新最终截面选项"""
        placeholder_list = [self.lang['placeholder_select']]
        if selected_tf == placeholder_list[0]:
            self.final_section_dropdown.configure(values=placeholder_list, state="disabled"); self.final_section_dropdown.set(placeholder_list[0])
            return

        if self.db is None: return
        try:
            hw_value = float(self.hw_dropdown.get())
            tw_value = float(self.tw_dropdown.get())
            bf_value = float(self.bf_dropdown.get())
            tf_value = float(selected_tf)
            filtered_db = self.db[(self.db['hw'] == hw_value) & (self.db['tw'] == tw_value) & (self.db['bf'] == bf_value) & (self.db['tf'] == tf_value)]
            
            section_names = filtered_db['SectionName'].tolist()
            final_values = placeholder_list + section_names
            self.final_section_dropdown.configure(values=final_values, state="normal")
            
            if len(section_names) == 1:
                self.final_section_dropdown.set(section_names[0])
                self.on_final_section_select(section_names[0])
            else:
                self.final_section_dropdown.set(placeholder_list[0])
        except Exception as e:
            self.display_error(f"{self.lang['select_tf_error']} {e}")

    def on_final_section_select(self, selected_section):
        """最终选择截面后，填充所有参数"""
        if self.db is None or not selected_section:
            return
        try:
            section_data = self.db[self.db["SectionName"] == selected_section].iloc[0]
            for key in self.entries.keys():
                if key in section_data.index:
                    self.set_entry_text(self.entries[key], section_data[key])
                else:
                    default_values = {
                        "E": 200000, "G": 77000, "nu": 0.3,
                        "phi_s": 0.75, "phi_f": 0.9, "omega2": 1.0,
                        "alpha_LT": 0.34, "L": 6000
                    }
                    if key in default_values:
                        self.set_entry_text(self.entries[key], default_values[key])
                    else:
                        self.set_entry_text(self.entries[key], "")
            self.on_mode_change()
        except Exception as e:
            self.display_error(f"{self.lang['read_section_error']} {e}")

    def reset_selection(self):
        """重置所有选择"""
        self.hw_dropdown.set(self.lang['placeholder_select'])
        self.tw_dropdown.set(self.lang['placeholder_select'])
        self.bf_dropdown.set(self.lang['placeholder_select'])
        self.tf_dropdown.set(self.lang['placeholder_select'])
        self.final_section_dropdown.set(self.lang['placeholder_select'])
        
        # 重新初始化下拉框
        self.initialize_dropdowns()

    def _arc_length_integrand(self, x, a3, w):
        if w == 0: return 1.0
        term = (np.pi * a3) / (2 * w) * np.cos((np.pi * x) / w)
        return np.sqrt(1 + term**2)

    def _calculate_arc_length(self, a3, w):
        s_val, _ = quad(self._arc_length_integrand, 0, w, args=(a3, w))
        return s_val

    def calculate_geometric_param(self, target_param):
        """使用高精度数值方法计算目标几何参数。"""
        try:
            a3 = float(self.entries['a3'].get()) if self.entries['a3'].get() else None
            s = float(self.entries['s'].get()) if self.entries['s'].get() else None
            w = float(self.entries['w'].get()) if self.entries['w'].get() else None

            if target_param == 's':
                if a3 is not None and w is not None:
                    val = self._calculate_arc_length(a3, w)
                    self.set_entry_text(self.entries['s'], f"{val:.4f}")
                else: self.display_error(self.lang['calc_s_error'])
            elif target_param == 'a3':
                if s is not None and w is not None:
                    if s < w:
                        self.display_error(self.lang['curve_length_error']); return
                    def objective_a3(a3_guess, s_target, w_known):
                        return self._calculate_arc_length(a3_guess, w_known) - s_target
                    upper_bound = (4 * w / np.pi) * np.sqrt(s / w + 1)
                    sol = root_scalar(objective_a3, args=(s, w), method='brentq', bracket=[0, upper_bound])
                    if sol.converged: self.set_entry_text(self.entries['a3'], f"{sol.root:.4f}")
                    else: self.display_error(self.lang['calc_not_converged'])
                else: self.display_error(self.lang['calc_a3_error'])
            elif target_param == 'w':
                if a3 is not None and s is not None:
                    min_w = 0.01 
                    if s < min_w:
                         self.display_error(self.lang['curve_too_small']); return
                    def objective_w(w_guess, s_target, a3_known):
                        return self._calculate_arc_length(a3_known, w_guess) - s_target
                    sol = root_scalar(objective_w, args=(s, a3), method='brentq', bracket=[min_w, s])
                    if sol.converged: self.set_entry_text(self.entries['w'], f"{sol.root:.4f}")
                    else: self.display_error(self.lang['calc_not_converged'])
                else: self.display_error(self.lang['calc_w_error'])
        except (ValueError, TypeError): self.display_error(self.lang['param_calc_error'])
        except Exception as e: self.display_error(f"{self.lang['unknown_error']} {e}\n{traceback.format_exc()}")

    def on_mode_change(self):
        """根据选择的模式，启用/禁用相应的控件。"""
        is_db_mode = (self.mode_var.get() == "database")
        
        # 控制数据库选择区域的显示和状态
        db_state = "normal" if is_db_mode else "disabled"
        if is_db_mode:
            self.database_frame.grid()
        else:
            self.database_frame.grid_remove()
        
        self.hw_dropdown.configure(state=db_state)
        self.tw_dropdown.configure(state=db_state)
        self.bf_dropdown.configure(state=db_state)
        self.tf_dropdown.configure(state=db_state)
        self.final_section_dropdown.configure(state=db_state)
        self.reset_button.configure(state=db_state)
        
        # --- 核心修改：精细化控制输入框状态 ---
        for key, entry in self.entries.items():
            if is_db_mode:
                # 在数据库模式下，只锁定几何参数
                if key in self.geometric_params:
                    entry.configure(state="disabled")
                else:
                    entry.configure(state="normal")
            else:
                # 在手动模式下，所有输入框都可用
                entry.configure(state="normal")

    def set_entry_text(self, entry, text):
        """一个辅助函数，用于设置输入框的文本。"""
        # 无论输入框当前是什么状态，都必须先设为 normal 才能修改文字
        entry.configure(state="normal")
        entry.delete(0, "end")
        entry.insert(0, str(text))
        # 注意：修改后不再将它设置回 disable，状态管理完全交给 on_mode_change

    def clear_fields(self):
        """清除所有输入框和结果。"""
        if self.mode_var.get() == 'manual':
            for entry in self.entries.values():
                entry.delete(0, "end")
        else:
            self.reset_selection()
        
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.configure(state="disabled")

    def on_calculate(self):
        """点击"计算"按钮时触发的核心函数。"""
        try:
            missing_params, invalid_params = [], []
            for key, entry in self.entries.items():
                value = entry.get().strip()
                if not value:
                    # --- FIX 1: START ---
                    # 更安全地获取参数的显示名称
                    param_label_key = self.param_label_keys.get(key)
                    display_name = self.lang.get(param_label_key, key) if param_label_key else key
                    missing_params.append(display_name)
                    # --- FIX 1: END ---
                else:
                    try: float(value)
                    except ValueError:
                        # --- FIX 2: START ---
                        # 同样, 更安全地获取参数的显示名称
                        param_label_key = self.param_label_keys.get(key)
                        p_name = self.lang.get(param_label_key, key) if param_label_key else key
                        # --- FIX 2: END ---
                        invalid_params.append(f"{p_name} ({self.lang['current_value']}: '{value}')")
            
            if missing_params: self.display_error(f"{self.lang['missing_params']}\n{', '.join(missing_params)}"); return
            if invalid_params: self.display_error(f"{self.lang['invalid_params']}\n{', '.join(invalid_params)}"); return

            params = {key: entry.get() for key, entry in self.entries.items()}
            
            selected_model_text = self.calc_model_var.get()
            model_for_backend = None
            if selected_model_text == self.lang['en_shear']: model_for_backend = LANGUAGES['zh']['en_shear']
            elif selected_model_text == self.lang['en_flexure']: model_for_backend = LANGUAGES['zh']['en_flexure']
            elif selected_model_text == self.lang['csa_flexure']: model_for_backend = LANGUAGES['zh']['csa_flexure']
            
            beam = CorrugatedWebGirder(**params)
            results = beam.calculate(method=model_for_backend or selected_model_text)

            self.result_textbox.configure(state="normal")
            self.result_textbox.delete("1.0", "end")
            
            result_text = f"{self.lang['input_params_section']}\n"
            params_to_display = {k: getattr(beam, k) for k in self.entries.keys() if hasattr(beam, k)}
            for key, value in params_to_display.items():
                # --- FIX 3: START ---
                # 使用同样安全的逻辑来获取用于显示的标签文本
                param_label_key = self.param_label_keys.get(key)
                if param_label_key:
                    label_text = self.lang.get(param_label_key, key)
                else:
                    label_text = key # Fallback for keys like 'a3', 's', 'w'
                # --- FIX 3: END ---
                result_text += f"{label_text:<28}: {value:.2f}\n"

            result_text += f"\n{self.lang['calc_results_section']}\n"
            result_text += f"{self.lang['calc_model']} {selected_model_text}\n--------------------\n"
            
            if (sc := results.get('shear_capacity')) is not None: result_text += f"{self.lang['shear_capacity']} {sc:.2f} {self.lang['units_kn']}\n"
            if (mc := results.get('moment_capacity')) is not None: result_text += f"{self.lang['moment_capacity']} {mc:.2f} {self.lang['units_knm']}\n"
            failure_mode_key = results.get('failure_mode') 
            if failure_mode_key:
                # 使用这个键去语言字典里查找对应的翻译文本
                # .get() 方法比 [] 更安全，如果找不到翻译，它会直接显示键本身，方便调试
                translated_fm = self.lang.get(failure_mode_key, failure_mode_key)
                result_text += f"{self.lang['failure_mode']} {translated_fm}\n"
            result_text += f"\n{self.lang['detailed_info']}\n"
            
        
            if model_for_backend == LANGUAGES['zh']['en_shear']:
                if (v:=results.get('local_capacity')) is not None: result_text += f"{self.lang['local_capacity']} {v:.2f} {self.lang['units_kn']}\n"
                if (v:=results.get('global_capacity')) is not None: result_text += f"{self.lang['global_capacity']} {v:.2f} {self.lang['units_kn']}\n"
                if (v:=results.get('local_buckling_stress')) is not None: result_text += f"{self.lang['local_stress']} {v:.2f} {self.lang['units_mpa']}\n"
                if (v:=results.get('global_buckling_stress')) is not None: result_text += f"{self.lang['global_stress']} {v:.2f} {self.lang['units_mpa']}\n"
                if (v:=results.get('local_slenderness')) is not None: result_text += f"{self.lang['local_slenderness']} {v:.3f}\n"
                if (v:=results.get('global_slenderness')) is not None: result_text += f"{self.lang['global_slenderness']} {v:.3f}\n"
                if (v:=results.get('local_reduction_factor')) is not None: result_text += f"{self.lang['local_reduction']} {v:.3f}\n"
                if (v:=results.get('global_reduction_factor')) is not None: result_text += f"{self.lang['global_reduction']} {v:.3f}\n"
            elif model_for_backend == LANGUAGES['zh']['en_flexure']:
                if (v:=results.get('slenderness_ratio')) is not None: result_text += f"{self.lang['slenderness_ratio']} (λ_LT): {v:.3f}\n"
                if (v:=results.get('ltb_limit')) is not None: result_text += f"{self.lang['ltb_limit']} {v:.3f}\n"
                if (v:=results.get('ltb_occurs')) is not None: result_text += f"{self.lang['ltb_occurs']} {self.lang['yes'] if v else self.lang['no']}\n"
                if (v:=results.get('critical_moment')) is not None: result_text += f"{self.lang['critical_moment']} {v:.2f} {self.lang['units_knm']}\n"
                if (v:=results.get('reduction_factor')) is not None: result_text += f"{self.lang['ltb_reduction']} {v:.3f}\n"
                if (v:=results.get('elastic_modulus')) is not None: result_text += f"{self.lang['elastic_section']} {v:.0f} {self.lang['units_cm3']}\n"
                if (v:=results.get('plastic_modulus')) is not None: result_text += f"{self.lang['plastic_section']} {v:.0f} {self.lang['units_cm3']}\n"
            elif model_for_backend == LANGUAGES['zh']['csa_flexure']:
                # --- 从这里开始修改 ---
                buckling_type_key = results.get('buckling_type') # 获取键, e.g., 'bt_inelastic_buckling'
                if buckling_type_key:
                    # 使用键查找翻译
                    translated_bt = self.lang.get(buckling_type_key, buckling_type_key)
                    result_text += f"{self.lang['buckling_type']} {translated_bt}\n"
                # --- 到这里结束修改 ---
                if (v:=results.get('slenderness_ratio')) is not None: result_text += f"{self.lang['slenderness_ratio']} {v:.3f}\n"
                if (v:=results.get('ltb_occurs')) is not None: result_text += f"{self.lang['ltb_occurs']} {self.lang['yes'] if v else self.lang['no']}\n"
                if (v:=results.get('plastic_moment')) is not None: result_text += f"{self.lang['plastic_moment']} {v:.2f} {self.lang['units_knm']}\n"
                if (v:=results.get('critical_moment')) is not None: result_text += f"{self.lang['elastic_critical']} {v:.2f} {self.lang['units_knm']}\n"
                if (v:=results.get('buckling_limit')) is not None: result_text += f"{self.lang['buckling_limit']} {v:.2f} {self.lang['units_knm']}\n"
                if (v:=results.get('elastic_modulus')) is not None: result_text += f"{self.lang['elastic_section']} {v:.0f} {self.lang['units_cm3']}\n"
                if (v:=results.get('plastic_modulus')) is not None: result_text += f"{self.lang['plastic_section']} {v:.0f} {self.lang['units_cm3']}\n"

            self.result_textbox.insert("1.0", result_text)
            self.result_textbox.configure(state="disabled")
        except ValueError as e: self.display_error(f"{self.lang['ensure_valid']}\n{self.lang['detailed_info_label']} {e}")
        except Exception as e: self.display_error(f"{self.lang['calc_error']}\n{str(e)}\n\n{self.lang['detailed_trace']}\n{traceback.format_exc()}")
    def display_error(self, message):
        """在结果区显示红色的错误信息。"""
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.insert("1.0", f"{self.lang['error_header']}\n\n{message}")
        self.result_textbox.tag_add("error", "1.0", "1.12")
        self.result_textbox.tag_config("error", foreground="red", font=("Courier New", 16, "bold"))
        self.result_textbox.configure(state="disabled")

    def on_language_change(self, lang_string):
        """切换UI语言"""
        self.current_lang = 'en' if lang_string == "English" else 'zh'
        self.lang = LANGUAGES[self.current_lang]
        self.update_ui_text()

    def update_ui_text(self):
        """更新所有UI组件的文本"""
        self.title(self.lang['window_title'])
        self.input_scroll_frame.configure(label_text=self.lang['input_params'])
        self.result_title_label.configure(text=self.lang['calc_results'])
        self.mode_label.configure(text=self.lang['select_mode'])
        self.radio_db.configure(text=self.lang['from_database'])
        self.radio_manual.configure(text=self.lang['manual_input'])
        self.db_title_label.configure(text=self.lang['step_select'])
        self.hw_label.configure(text=self.lang['web_height'])
        self.tw_label.configure(text=self.lang['web_thickness'])
        self.bf_label.configure(text=self.lang['flange_width'])
        self.tf_label.configure(text=self.lang['flange_thickness'])
        self.section_label.configure(text=self.lang['select_section'])
        self.reset_button.configure(text=self.lang['reset_selection'])
        for key, label in self.entry_labels.items():
            label.configure(text=self.lang[self.param_label_keys[key]])
        self.corr_title.configure(text=self.lang['corrugation_geometry'])
        self.a3_label.configure(text=self.lang['corrugation_height'])
        self.s_label.configure(text=self.lang['half_wave_length'])
        self.w_label.configure(text=self.lang['half_wave_projection'])
        for btn in [self.a3_calc_btn, self.s_calc_btn, self.w_calc_btn]:
            btn.configure(text=self.lang['calculate'])
        self.model_label.configure(text=self.lang['select_model'])
        
        current_model_text = self.calc_model_var.get()
        new_model_values = [self.lang['en_shear'], self.lang['en_flexure'], self.lang['csa_flexure']]
        self.model_dropdown.configure(values=new_model_values)

        if current_model_text in [LANGUAGES['zh']['en_shear'], LANGUAGES['en']['en_shear']]:
             self.calc_model_var.set(self.lang['en_shear'])
        elif current_model_text in [LANGUAGES['zh']['en_flexure'], LANGUAGES['en']['en_flexure']]:
             self.calc_model_var.set(self.lang['en_flexure'])
        elif current_model_text in [LANGUAGES['zh']['csa_flexure'], LANGUAGES['en']['csa_flexure']]:
             self.calc_model_var.set(self.lang['csa_flexure'])
        else: self.calc_model_var.set(new_model_values[0])

        self.calc_button.configure(text=self.lang['calc_button'])
        self.clear_button.configure(text=self.lang['clear_button'])
        if self.db is None: self.section_names = [self.lang['file_not_found']]
        self.initialize_dropdowns() # 直接调用初始化函数，它会用新的语言刷新所有下拉框
if __name__ == "__main__":
    app = App()
    app.mainloop()