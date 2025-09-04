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
    

    def _calculate_EN_flexural(self):
        
        Iy = 2 * (self.tf * self.bf **3 / 12) 
        Ix =  1/12 * (self.bf * (self.hw + 2 * self.tf)**3 - self.bf * self.hw**3)
        # 强轴截面模量
        Wx = Ix / (self.hw / 2 + self.tf)
        # 特征抗弯承载力 (已修正为使用 Wx)
        M_Rk = Wx * self.Fy
        It = (1/3) * (2 * self.bf *self.tf**3) 
        Iw = Iy * self.hw**2 / 4 
        term1_sqrt = self.E * Iy * self.G * It
        term2_sqrt = (np.pi**2 * self.E**2 * Iy * Iw) /self.L**2
        M_cr = (self.omega2 * np.pi / self.L) * np.sqrt(term1_sqrt + term2_sqrt)
    
        if M_cr <= 0:
            # 修正：确保返回格式一致
            return {
                "shear_capacity": None,
                "moment_capacity": 0.0
            }

        # 1. 计算相对长细比 lambda_LT
        lambda_LT = np.sqrt(M_Rk / M_cr)
        
        # 2. 计算辅助值 phi_LT
        phi_LT = 0.5 * (1 + self.alpha_LT * (lambda_LT - 0.2) + lambda_LT**2)
        
        # 3. 计算折减系数 chi_LT
        # 增加一个检查防止 phi_LT**2 - lambda_LT**2 为负数
        inner_sqrt = phi_LT**2 - lambda_LT**2
        if inner_sqrt < 0:
            inner_sqrt = 0 # 防止计算错误
            
        chi_LT = 1 / (phi_LT + np.sqrt(inner_sqrt))
        chi_LT = min(chi_LT, 1.0) # 折减系数不能大于1
        
        # 4. 计算最终的抗弯屈曲承载力
        moment_capacity = chi_LT * M_Rk
    
        return {
            "shear_capacity": None,  # 未计算剪力
            "moment_capacity": moment_capacity / 1e6,
        }
    

    def _calculate_CSA_flexural(self):
        Iy = 2 * (self.tf * self.bf **3 / 12) 
        Ix =  1/12 * (self.bf * (self.hw + 2 * self.tf)**3 - self.bf * self.hw**3)
        # 强轴截面模量
        Wx = Ix / (self.hw / 2 + self.tf)
        # 特征抗弯承载力 (已修正为使用 Wx)
        M_Rk = Wx * self.Fy
        It = (1/3) * (2 * self.bf *self.tf**3) 
        Iw = Iy * self.hw**2 / 4 
        term1_sqrt = self.E * Iy * self.G * It
        term2_sqrt = (np.pi**2 * self.E**2 * Iy * Iw) /self.L**2
        M_cr = (self.omega2 * np.pi / self.L) * np.sqrt(term1_sqrt + term2_sqrt)
        
        # 1. 计算强轴塑性截面模量 Zx
        Zx = 1/12 * (self.bf * (self.hw + 2 * self.tf)**3 - self.bf * self.hw**3) / 0.5 / (self.hw + 2 * self.tf)
    
        # 2. 计算塑性矩 Mp
        Mp = self.Fy * Zx
        
        # 3. 将 Mcr 作为 Mu
        Mu = M_cr
        
        # 4. 判断屈曲范围并选择公式计算 Mr
        if Mu > 0.67 * Mp:
            # 非弹性屈曲范围
            moment_capacity = 1.15 * self.phi_f * Mp * (1 - (0.28 * Mp / Mu))
            # 规范规定，计算结果不能超过 phi * Mp
            moment_capacity = min(moment_capacity, self.phi_f * Mp)
            print(f"    [CSA S16] 范围: 非弹性屈曲 (Mu > 0.67*Mp)")
        else:
            # 弹性屈曲范围
            moment_capacity = self.phi_f * Mu
            print(f"    [CSA S16] 范围: 弹性屈曲 (Mu <= 0.67*Mp)")

        return {
            "shear_capacity": None,  # 未计算剪力
            "moment_capacity": moment_capacity / 1e6,
        }
    

    def _calculate_EN_shear(self):
        
        # 步骤 2.2: 计算 τ_cr,l (局部临界剪应力)
        tau_crl = (5.34 + (self.a3 * self.s) / (self.hw * self.tw)) * (np.pi**2 * self.E / (12 * (1 - self.nu**2))) * (self.tw / self.s)**2
    
        # 步骤 2.3: 计算 λ_c,l (相对长细比)
        lambda_crl = np.sqrt(self.Fy / (tau_crl * np.sqrt(3)))
    
        # 步骤 2.4: 计算 χ_c,l (折减系数)
        chi_crl = np.minimum(1.15 / (0.9 + lambda_crl), 1.0)
    
        # 步骤 2.5: 计算最终的 V_r
        vl_final = self.phi_s * chi_crl * self.Fy / np.sqrt(3) * self.hw * self.tw

        Iz = (self.w * self.tw**3) / 12 + (self.w * self.tw * self.a3**2) / 8
        # 步骤 4.2: 计算 Dx 和 Dz
        Dx = (self.E * self.tw**3) / (12 * (1 - self.nu**2)) * (self.w / self.s)
        Dz = self.E * Iz / self.w

        # 步骤 4.3: 计算 τ_cr,g
        tau_crg = (32.4 / (self.tw * self.hw**2))* (Dx * Dz**3)**(1/4)

        # 步骤 4.4: 计算 λ_c,g 和 χ_c,g
        lambda_crg = np.sqrt(self.Fy / (tau_crg * np.sqrt(3)))
        chi_crg = np.minimum(1.5 / (0.5 + lambda_crg**2), 1.0)

        # 步骤 4.5: 计算最终的 Vg
        vg_final = self.phi_s * chi_crg * self.Fy / np.sqrt(3) * self.hw * self.tw

        if vl_final < vg_final:
            shear_capacity = vl_final  
        else:
            shear_capacity = vg_final

        return {
            "shear_capacity": shear_capacity / 1000,
            "moment_capacity": None,  # 未计算弯矩
        }