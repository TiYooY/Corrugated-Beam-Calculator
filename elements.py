# elements.py

from abc import ABC, abstractmethod
import numpy as np

class StructuralElement(ABC):
    """一个抽象基类，定义所有结构构件的通用接口。"""
    @abstractmethod
    def calculate(self, method: str):
        # 强制所有子类（如波纹腹板钢梁）都必须实现一个名为 calculate 的方法
        pass

class CorrugatedWebGirder(StructuralElement):
    """波纹腹板钢梁的具体实现类。"""
    def __init__(self, hw, tw, bf, tf, Fy, a3, s, w, E, G, nu, phi_s, phi_f, L, omega2, alpha_LT):
        # 初始化对象时，存储所有几何和材料属性
        self.hw = float(hw) # 腹板高度
        self.tw = float(tw) # 腹板厚度
        self.bf = float(bf) # 翼缘宽度
        self.tf = float(tf) # 翼缘厚度
        self.Fy = float(Fy) # 屈服强度
        self.a3 = float(a3) # 波纹高度
        self.s = float(s) # 半波纹曲线长
        self.w = float(w) # 半波纹投影长
        self.E = float(E) # 弹性模量
        self.G = float(G) # 切变模量
        self.nu = float(nu) # 泊松比
        self.phi_s = float(phi_s) # 抗剪折减系数
        self.phi_f = float(phi_f) # 抗弯折减系数
        self.L = float(L) # 跨度
        self.omega2 = float(omega2) # 弯矩梯度系数
        self.alpha_LT = float(alpha_LT) # 屈曲修正系数

    def calculate(self, method: str):
        """
        公共计算方法，根据传入的 method 参数，调用相应的私有计算函数。
        这是面向对象设计中"封装"的体现。
        """
        if method == "EN剪力模型":
            return self._calculate_EN_shear()
        elif method == "EN弯曲模型":
            return self._calculate_EN_flexural()
        elif method == "CSA弯曲模型":
            return self._calculate_CSA_flexural()
        else:
            # 如果传入一个未知的模型名称，则引发错误
            raise ValueError(f"未知的计算模型: {method}")

    def _calculate_section_properties(self):
        """计算截面几何特性"""
        # 惯性矩
        Iy = 2 * (self.tf * self.bf **3 / 12) 
        Ix = 1/12 * (self.bf * (self.hw + 2 * self.tf)**3 - self.bf * self.hw**3)
        
        # 弹性截面模量
        Wx = Ix / (self.hw / 2 + self.tf)
        
        # 塑性截面模量（简化计算）
        Zx = self.bf * self.tf * (self.hw + self.tf) + self.hw**2 * self.tw / 4
        
        # 扭转和翘曲惯性矩
        It = (1/3) * (2 * self.bf * self.tf**3) 
        Iw = Iy * self.hw**2 / 4
        
        return {
            'Ix': Ix, 'Iy': Iy, 'Wx': Wx, 'Zx': Zx,
            'It': It, 'Iw': Iw
        }

    def _calculate_EN_flexural(self):
        """EN弯曲模型计算，包含详细的LTB判断"""
        section_props = self._calculate_section_properties()
        
        # 特征抗弯承载力
        M_Rk = section_props['Wx'] * self.Fy
        
        # 弹性临界弯矩
        term1_sqrt = self.E * section_props['Iy'] * self.G * section_props['It']
        term2_sqrt = (np.pi**2 * self.E**2 * section_props['Iy'] * section_props['Iw']) / self.L**2
        M_cr = (self.omega2 * np.pi / self.L) * np.sqrt(term1_sqrt + term2_sqrt)
    
        if M_cr <= 0:
            return {
                "shear_capacity": None,
                "moment_capacity": 0.0,
                "failure_mode": "计算错误：临界弯矩为负值"
            }

        # 1. 计算相对长细比 lambda_LT
        lambda_LT = np.sqrt(M_Rk / M_cr)
        
        # 2. 判断是否发生LTB
        # 一般来说，当lambda_LT > 0.4时开始考虑LTB效应
        ltb_limit = 0.4
        ltb_occurs = lambda_LT > ltb_limit
        
        # 3. 计算辅助值 phi_LT
        phi_LT = 0.5 * (1 + self.alpha_LT * (lambda_LT - 0.2) + lambda_LT**2)
        
        # 4. 计算折减系数 chi_LT
        inner_sqrt = phi_LT**2 - lambda_LT**2
        if inner_sqrt < 0:
            inner_sqrt = 0
            
        chi_LT = 1 / (phi_LT + np.sqrt(inner_sqrt))
        chi_LT = min(chi_LT, 1.0)
        
        # 5. 确定失效模式
        if lambda_LT <= 0.2:
            failure_mode = "截面强度控制（无LTB影响）"
        elif lambda_LT <= ltb_limit:
            failure_mode = "截面强度控制（轻微LTB影响）"
        else:
            failure_mode = "侧扭屈曲控制"
        
        # 6. 计算最终的抗弯承载力
        moment_capacity = self.phi_f * chi_LT * M_Rk

        return {
            "shear_capacity": None,
            "moment_capacity": moment_capacity / 1e6,
            "failure_mode": failure_mode,
            "slenderness_ratio": lambda_LT,
            "ltb_limit": ltb_limit,
            "ltb_occurs": ltb_occurs,
            "elastic_modulus": section_props['Wx'] / 1e3,  # 转换为cm³
            "plastic_modulus": section_props['Zx'] / 1e3,  # 转换为cm³
            "critical_moment": M_cr / 1e6,  # 转换为kN·m
            "reduction_factor": chi_LT,
            "phi_LT": phi_LT
        }

    def _calculate_CSA_flexural(self):
        """CSA弯曲模型计算，包含详细信息"""
        section_props = self._calculate_section_properties()
        
        # 弹性临界弯矩计算（与EN相同）
        term1_sqrt = self.E * section_props['Iy'] * self.G * section_props['It']
        term2_sqrt = (np.pi**2 * self.E**2 * section_props['Iy'] * section_props['Iw']) / self.L**2
        M_cr = (self.omega2 * np.pi / self.L) * np.sqrt(term1_sqrt + term2_sqrt)
        
        # 1. 计算塑性矩 Mp
        Mp = self.Fy * section_props['Zx']
        
        # 2. 将 Mcr 作为 Mu
        Mu = M_cr
        
        # 3. 判断屈曲范围并计算 Mr
        buckling_limit = 0.67 * Mp
        
        if Mu > buckling_limit:
            # 非弹性屈曲范围
            moment_capacity = 1.15 * self.phi_f * Mp * (1 - (0.28 * Mp / Mu))
            moment_capacity = min(moment_capacity, self.phi_f * Mp)
            failure_mode = "非弹性屈曲控制"
            buckling_type = "非弹性屈曲"
        else:
            # 弹性屈曲范围
            moment_capacity = self.phi_f * Mu
            failure_mode = "弹性屈曲控制"
            buckling_type = "弹性屈曲"
        
        # 4. 计算长细比（基于CSA方法）
        slenderness_ratio = np.sqrt(Mp / Mu) if Mu > 0 else float('inf')
        ltb_occurs = slenderness_ratio > 0.4

        return {
            "shear_capacity": None,
            "moment_capacity": moment_capacity / 1e6,
            "failure_mode": failure_mode,
            "buckling_type": buckling_type,
            "slenderness_ratio": slenderness_ratio,
            "ltb_occurs": ltb_occurs,
            "plastic_moment": Mp / 1e6,
            "critical_moment": Mu / 1e6,
            "buckling_limit": buckling_limit / 1e6,
            "elastic_modulus": section_props['Wx'] / 1e3,
            "plastic_modulus": section_props['Zx'] / 1e3
        }

    def _calculate_local_shear_capacity(self):
        """计算局部剪切屈曲承载力"""
        # 计算 τ_cr,l (局部临界剪应力)
        tau_crl = (5.34 + (self.a3 * self.s) / (self.hw * self.tw)) * \
                  (np.pi**2 * self.E / (12 * (1 - self.nu**2))) * (self.tw / self.s)**2
        
        # 计算 λ_c,l (相对长细比)
        lambda_crl = np.sqrt(self.Fy / (tau_crl * np.sqrt(3)))
        
        # 计算 χ_c,l (折减系数)
        chi_crl = np.minimum(1.15 / (0.9 + lambda_crl), 1.0)
        
        # 计算局部剪切承载力
        vl = self.phi_s * chi_crl * self.Fy / np.sqrt(3) * self.hw * self.tw
        
        return {
            'capacity': vl,
            'critical_stress': tau_crl,
            'slenderness': lambda_crl,
            'reduction_factor': chi_crl
        }

    def _calculate_global_shear_capacity(self):
        """计算整体剪切屈曲承载力"""
        # 计算截面几何参数
        Iz = (self.w * self.tw**3) / 12 + (self.w * self.tw * self.a3**2) / 8
        
        # 计算 Dx 和 Dz
        Dx = (self.E * self.tw**3) / (12 * (1 - self.nu**2)) * (self.w / self.s)
        Dz = self.E * Iz / self.w

        # 计算 τ_cr,g
        tau_crg = (32.4 / (self.tw * self.hw**2)) * (Dx * Dz**3)**(1/4)

        # 计算 λ_c,g 和 χ_c,g
        lambda_crg = np.sqrt(self.Fy / (tau_crg * np.sqrt(3)))
        chi_crg = np.minimum(1.5 / (0.5 + lambda_crg**2), 1.0)

        # 计算整体剪切承载力
        vg = self.phi_s * chi_crg * self.Fy / np.sqrt(3) * self.hw * self.tw
        
        return {
            'capacity': vg,
            'critical_stress': tau_crg,
            'slenderness': lambda_crg,
            'reduction_factor': chi_crg,
            'Dx': Dx,
            'Dz': Dz,
            'Iz': Iz
        }

    def _calculate_EN_shear(self):
        """EN剪力模型计算，包含详细的失效模式判断"""
        # 计算局部和整体剪切承载力
        local_results = self._calculate_local_shear_capacity()
        global_results = self._calculate_global_shear_capacity()
        
        vl = local_results['capacity']
        vg = global_results['capacity']
        
        # 判断失效模式
        if vl < vg:
            failure_mode = "局部剪切屈曲控制"
            controlling_capacity = vl
            controlling_stress = local_results['critical_stress']
            controlling_slenderness = local_results['slenderness']
            controlling_reduction = local_results['reduction_factor']
        else:
            failure_mode = "整体剪切屈曲控制"
            controlling_capacity = vg
            controlling_stress = global_results['critical_stress']
            controlling_slenderness = global_results['slenderness']
            controlling_reduction = global_results['reduction_factor']

        return {
            "shear_capacity": controlling_capacity / 1000,  # 转换为kN
            "moment_capacity": None,
            "failure_mode": failure_mode,
            "local_buckling_stress": local_results['critical_stress'],
            "global_buckling_stress": global_results['critical_stress'],
            "local_capacity": vl / 1000,
            "global_capacity": vg / 1000,
            "controlling_stress": controlling_stress,
            "controlling_slenderness": controlling_slenderness,
            "controlling_reduction_factor": controlling_reduction,
            "local_slenderness": local_results['slenderness'],
            "global_slenderness": global_results['slenderness'],
            "local_reduction_factor": local_results['reduction_factor'],
            "global_reduction_factor": global_results['reduction_factor']
        }