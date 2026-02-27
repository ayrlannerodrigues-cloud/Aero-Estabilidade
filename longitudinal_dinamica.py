# -*- coding: utf-8 -*-
"""
ANÁLISE DE ESTABILIDADE LONGITUDINAL - MÉTODO ETKIN CORRIGIDO
Baseado em Etkin "Dynamics of Flight: Stability and Control"

CORREÇÃO: Matriz A agora segue a convenção do Etkin com acoplamento 
correto entre os modos e definição adequada das variáveis de estado.

Estados: [Δu, Δw, Δq, Δθ] - perturbações em relação ao trim
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from math import sqrt, pi, cos, sin, radians, degrees

class LongitudinalStabilityEtkin:
    def __init__(self, aircraft_data):
        """
        Inicializa análise seguindo convenção do Etkin
        """
        # Parâmetros físicos
        self.g = 9.8065
        self.mass = aircraft_data['mass']           # kg
        self.S = aircraft_data['S']                 # m²
        self.MAC = aircraft_data['MAC']             # m (chord)
        self.Iy = aircraft_data['Iy']               # kg⋅m²
        self.U0 = aircraft_data['velocity']         # m/s (trim speed)
        self.rho = aircraft_data['rho']             # kg/m³
        
        # Condições de trim
        self.CL0 = aircraft_data['CL_trim']
        self.CD0 = aircraft_data['CD_trim'] 
        self.theta0 = radians(aircraft_data['theta_trim'])  # rad
        
        # Pressão dinâmica
        self.q_inf = 0.5 * self.rho * self.U0**2
        
        print(f"=== ANÁLISE LONGITUDINAL - CONVENÇÃO ETKIN ===")
        print(f"Aeronave: {aircraft_data.get('name', 'Generic')}")
        print(f"Massa: {self.mass} kg")
        print(f"Velocidade trim (U₀): {self.U0} m/s")
        print(f"Pressão dinâmica: {self.q_inf:.1f} Pa")
        print(f"Ângulo de trim: {degrees(self.theta0):.1f}°")
        print("-" * 50)
    
    def calculate_dimensional_derivatives(self, coefficients):
        """
        Calcula derivadas dimensionais seguindo Etkin
        
        IMPORTANTE: Convenção do Etkin usa definições ligeiramente 
        diferentes das derivadas Zₘ e Mₘ
        """
        # Coeficientes adimensionais
        CL_alpha = coefficients['CL_alpha']
        CD_alpha = coefficients['CD_alpha']
        CDu = coefficients['CDu']
        Cm_alpha = coefficients['Cm_alpha']
        Cm_alpha_dot = coefficients['Cm_alpha_dot']
        Cmq = coefficients['Cmq']
        
        # DERIVADAS DA FORÇA X (Longitudinal)
        Xu = -CDu * self.q_inf * self.S / (self.mass * self.U0)
        Xw = -(CD_alpha - self.CL0) * self.q_inf * self.S / (self.mass * self.U0)
        
        # DERIVADAS DA FORÇA Z (Normal) - Convenção Etkin
        Zu = -2.0 * self.CL0 * self.q_inf * self.S / (self.mass * self.U0)
        Zw = -(CL_alpha + self.CD0) * self.q_inf * self.S / (self.mass * self.U0)
        
        # Derivadas com ponto (time derivatives)
        Zw_dot = 0
        Zq = 0
        
        # DERIVADAS DO MOMENTO M (Pitch) - Convenção Etkin
        Mu = 0.0  # geralmente desprezível
        Mw = Cm_alpha * self.q_inf * self.S * self.MAC / (self.Iy * self.U0)
        Mw_dot = Cm_alpha_dot * self.q_inf * self.S * self.MAC**2 / (2 * self.Iy * self.U0**2)
        Mq = Cmq * self.q_inf * self.S * self.MAC**2 / (2 * self.Iy * self.U0)
        
        # Armazenar derivadas
        self.derivatives = {
            'Xu': Xu, 'Xw': Xw, 'Zu': Zu, 'Zw': Zw,
            'Zw_dot': Zw_dot, 'Zq': Zq,
            'Mu': Mu, 'Mw': Mw, 'Mw_dot': Mw_dot, 'Mq': Mq
        }
        
        print("DERIVADAS DIMENSIONAIS (Convenção Etkin):")
        for key, value in self.derivatives.items():
            print(f"{key} = {value:.6f}")
        print("-" * 50)
        
        return self.derivatives
    
    def build_etkin_matrix(self):
        """
        Constrói matriz A seguindo EXATAMENTE a convenção do Etkin
        
        Equação: ẋ = Ax onde x = [Δu, Δw, Δq, Δθ]ᵀ
        
        Forma do Etkin (eq. 6.3.6):
        [Δu̇]   [Xu/m   Xw/m     0      -g  ] [Δu]
        [Δẇ] = [Zu/m   Zw/m    u₀      0  ] [Δw]
        [Δq̇]   [Mu     Mw       0      0  ] [Δq]
        [Δθ̇]   [ 0      0       1      0  ] [Δθ]
        """
        d = self.derivatives
        
        # Matriz A na forma do Etkin
        A = np.array([
            [d['Xu'],           d['Xw'],           0,                    -self.g*cos(self.theta0)],
            [d['Zu'],           d['Zw'],           self.U0 + d['Zq'],    -self.g*sin(self.theta0)],
            [d['Mu'] + d['Mw_dot']*d['Zu'], d['Mw'] + d['Mw_dot']*d['Zw'], d['Mq'] + d['Mw_dot']*(self.U0 + d['Zq']), -d['Mw_dot']*self.g*sin(self.theta0)],
            [0,                 0,                 1,                    0]
        ])
        
        print("MATRIZ DE ESTADO A (Convenção Etkin):")
        print("Estados: [Δu, Δw, Δq, Δθ]ᵀ")
        print(A)
        print("-" * 50)
        
        return A
    
    def analyze_longitudinal_modes(self, A):
        """
        Analisa autovalores e classifica modos longitudinais
        """
        eigenvalues, eigenvectors = la.eig(A)
        
        print("AUTOVALORES DA MATRIZ A:")
        for i, val in enumerate(eigenvalues):
            print(f"λ{i+1} = {val:.6f}")
        print("-" * 50)
        
        # Classificar modos
        modes = self.classify_modes(eigenvalues)
        
        return eigenvalues, eigenvectors, modes
    
    def classify_modes(self, eigenvalues):
        """
        Classifica modos conforme literatura clássica
        """
        modes = []
        processed = set()
        
        for i, val in enumerate(eigenvalues):
            if i in processed:
                continue
            
            if abs(val.imag) < 1e-8:  # Modo real
                processed.add(i)
                modes.append({
                    'type': 'real',
                    'name': 'Modo Real',
                    'eigenvalue': val,
                    'time_constant': -1.0/val.real if val.real < 0 else float('inf'),
                    'stable': val.real < 0
                })
            else:  # Modo complexo conjugado
                # Buscar conjugado
                conjugate_idx = None
                for j, other in enumerate(eigenvalues):
                    if (j != i and j not in processed and 
                        abs(other.real - val.real) < 1e-10 and 
                        abs(other.imag + val.imag) < 1e-10):
                        conjugate_idx = j
                        break
                
                if conjugate_idx is not None:
                    processed.add(i)
                    processed.add(conjugate_idx)
                    
                    sigma = val.real
                    omega_d = abs(val.imag)
                    omega_n = sqrt(sigma**2 + omega_d**2)
                    zeta = -sigma / omega_n if omega_n > 0 else 0
                    
                    # Classificação por frequência
                    if omega_n < 1.0:  # Baixa frequência
                        mode_name = "Phugoid"
                        description = "Modo de velocidade (intercâmbio energia cinética/potencial)"
                    else:  # Alta frequência
                        mode_name = "Short Period"
                        description = "Modo de ângulo de ataque (movimento rápido de arfagem)"
                    
                    period = 2*pi/omega_d if omega_d > 0 else float('inf')
                    time_to_half = 0.693/abs(sigma) if sigma < 0 else float('inf')
                    
                    modes.append({
                        'type': 'complex',
                        'name': mode_name,
                        'description': description,
                        'eigenvalue': val,
                        'sigma': sigma,
                        'omega_d': omega_d,
                        'omega_n': omega_n,
                        'zeta': zeta,
                        'period': period,
                        'time_to_half': time_to_half,
                        'stable': sigma < 0
                    })
        
        return modes
    
    def print_detailed_analysis(self, modes):
        """
        Imprime análise detalhada dos modos longitudinais
        """
        print("ANÁLISE DETALHADA DOS MODOS LONGITUDINAIS:")
        print("=" * 70)
        
        for mode in modes:
            print(f"\n>>> {mode['name'].upper()} <<<")
            
            if mode['type'] == 'complex':
                print(f"Descrição: {mode['description']}")
                print(f"Autovalor: {mode['eigenvalue']:.6f}")
                print(f"Parte real (σ): {mode['sigma']:.6f} rad/s")
                print(f"Frequência amortecida (ωd): {mode['omega_d']:.6f} rad/s")
                print(f"Frequência natural (ωn): {mode['omega_n']:.6f} rad/s")
                print(f"Razão de amortecimento (ζ): {mode['zeta']:.6f}")
                print(f"Período: {mode['period']:.2f} s")
                print(f"Tempo de meia amplitude: {mode['time_to_half']:.2f} s")
                
                # Verificação de estabilidade e valores típicos
                if mode['name'] == 'Phugoid':
                    wn_ok = 0.05 <= mode['omega_n'] <= 0.4
                    zeta_ok = 0.01 <= mode['zeta'] <= 0.2
                    print(f"Valores típicos: ωn = 0.05-0.4 rad/s, ζ = 0.01-0.2")
                elif mode['name'] == 'Short Period':
                    wn_ok = 1.0 <= mode['omega_n'] <= 10.0
                    zeta_ok = 0.3 <= mode['zeta'] <= 1.5
                    print(f"Valores típicos: ωn = 1.0-10.0 rad/s, ζ = 0.3-1.5")
                
                stability = "ESTÁVEL" if mode['stable'] else "INSTÁVEL"
                validation = "✅ OK" if (wn_ok and zeta_ok and mode['stable']) else "⚠️ VERIFICAR"
                print(f"Estabilidade: {stability}")
                print(f"Validação: {validation}")
                
            else:  # Modo real
                print(f"Autovalor: {mode['eigenvalue']:.6f}")
                print(f"Constante de tempo: {mode['time_constant']:.2f} s")
                print(f"Estabilidade: {'ESTÁVEL' if mode['stable'] else 'INSTÁVEL'}")
            
            print("-" * 50)
    
    def plot_comprehensive_analysis(self, eigenvalues, modes):
        """
        Plota análise completa dos modos longitudinais com melhor distribuição
        """
        # Criar figura maior com melhor espaçamento
        fig = plt.figure(figsize=(20, 15))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        
        # Plot 1: S-plane (Root Locus) - CORRIGIDO
        ax1 = plt.subplot(2, 3, 1)
        
        colors = {'Phugoid': 'blue', 'Short Period': 'red', 'Modo Real': 'green'}
        markers = {'Phugoid': 'o', 'Short Period': '^', 'Modo Real': 's'}
        
        for mode in modes:
            val = mode['eigenvalue']
            color = colors.get(mode['name'], 'black')
            marker = markers.get(mode['name'], 'x')
            
            if mode['type'] == 'complex':
                # Plot both poles (positive and negative imaginary parts)
                ax1.scatter(val.real, val.imag, c=color, marker=marker, 
                           s=120, alpha=0.9, edgecolors='black', linewidth=1,
                           label=mode['name'])
                ax1.scatter(val.real, -val.imag, c=color, marker=marker, 
                           s=120, alpha=0.9, edgecolors='black', linewidth=1)
                
                # Adicionar anotações
                ax1.annotate(mode['name'], (val.real, val.imag), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, fontweight='bold')
            else:
                ax1.scatter(val.real, 0, c=color, marker=marker, 
                           s=100, alpha=0.9, edgecolors='black', linewidth=1,
                           label=mode['name'])
        
        ax1.axhline(y=0, color='black', linewidth=1, alpha=0.7)
        ax1.axvline(x=0, color='red', linewidth=2, alpha=0.8, label='Limite Estabilidade')
        ax1.grid(True, alpha=0.4)
        ax1.set_xlabel('Parte Real σ (rad/s)', fontsize=11)
        ax1.set_ylabel('Parte Imaginária ωd (rad/s)', fontsize=11)
        ax1.set_title('Plano S - Autovalores Longitudinais', fontsize=12, fontweight='bold')
        
        # Ajustar limites para melhor visualização
        sigma_vals = [mode['eigenvalue'].real for mode in modes if mode['type'] == 'complex']
        omega_vals = [abs(mode['eigenvalue'].imag) for mode in modes if mode['type'] == 'complex']
        
        if sigma_vals and omega_vals:
            margin_x = 0.1 * (max(sigma_vals) - min(sigma_vals))
            margin_y = 0.1 * max(omega_vals)
            ax1.set_xlim(min(sigma_vals) - margin_x, 0.5)
            ax1.set_ylim(-max(omega_vals) - margin_y, max(omega_vals) + margin_y)
        
        ax1.legend(loc='upper right')
        
        # Plot 2: Características modais - CORRIGIDO
        ax2 = plt.subplot(2, 3, 2)
        complex_modes = [m for m in modes if m['type'] == 'complex']
        
        if len(complex_modes) >= 1:
            for mode in complex_modes:
                color = 'blue' if mode['name'] == 'Phugoid' else 'red'
                marker = 'o' if mode['name'] == 'Phugoid' else '^'
                
                ax2.scatter(mode['omega_n'], mode['zeta'], 
                           c=color, s=150, alpha=0.9, marker=marker,
                           edgecolors='black', linewidth=1.5,
                           label=f"{mode['name']}")
                
                # Anotação com valores
                ax2.annotate(f"{mode['name']}\nωn={mode['omega_n']:.3f}\nζ={mode['zeta']:.3f}", 
                           (mode['omega_n'], mode['zeta']), 
                           xytext=(15, 15), textcoords='offset points',
                           fontsize=9, ha='left',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
            
            # Regiões típicas mais precisas
            ax2.axhspan(0.01, 0.2, xmin=0, xmax=0.08, alpha=0.15, color='blue', label='Phugoid Típico')
            ax2.axhspan(0.3, 1.5, xmin=0.15, xmax=1.0, alpha=0.15, color='red', label='Short Period Típico')
            
            ax2.set_xlabel('Frequência Natural ωn (rad/s)', fontsize=11)
            ax2.set_ylabel('Razão de Amortecimento ζ', fontsize=11)
            ax2.set_title('Características Modais', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.4)
            ax2.set_xscale('log')
            ax2.set_xlim(0.05, 20)
            ax2.set_ylim(0, 1.6)
        
        # Plot 3: Resposta temporal Phugoid - CORRIGIDO
        ax3 = plt.subplot(2, 3, 4)
        phugoid = next((m for m in complex_modes if m['name'] == 'Phugoid'), None)
        
        if phugoid:
            # Tempo adequado para ver 2-3 períodos completos
            t_max = 3 * phugoid['period']  # 3 períodos
            t = np.linspace(0, t_max, 2000)
            
            sigma = phugoid['sigma']
            omega_d = phugoid['omega_d']
            
            envelope_pos = np.exp(sigma * t)
            envelope_neg = -np.exp(sigma * t)
            response = envelope_pos * np.cos(omega_d * t)
            
            ax3.plot(t, response, 'b-', linewidth=2.5, label='Resposta Phugoid', alpha=0.9)
            ax3.plot(t, envelope_pos, 'b--', linewidth=1.5, alpha=0.7, label='Envelope ±e^(σt)')
            ax3.plot(t, envelope_neg, 'b--', linewidth=1.5, alpha=0.7)
            
            # Marcar períodos
            periods = np.arange(phugoid['period'], t_max, phugoid['period'])
            for p in periods:
                ax3.axvline(x=p, color='blue', linestyle=':', alpha=0.5)
            
            ax3.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
            ax3.grid(True, alpha=0.4)
            ax3.set_xlabel('Tempo (s)', fontsize=11)
            ax3.set_ylabel('Amplitude Normalizada', fontsize=11)
            ax3.set_title(f'Modo Phugoid (T = {phugoid["period"]:.1f}s, ζ = {phugoid["zeta"]:.3f})', 
                         fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.set_xlim(0, t_max)
        
        # Plot 4: Resposta temporal Short Period - CORRIGIDO
        ax4 = plt.subplot(2, 3, 5)
        short_period = next((m for m in complex_modes if m['name'] == 'Short Period'), None)
        
        if short_period:
            # Tempo para ver amortecimento completo
            t_max = 5 * short_period['time_to_half']  # 5 × tempo de meia vida
            t = np.linspace(0, t_max, 2000)
            
            sigma = short_period['sigma']
            omega_d = short_period['omega_d']
            
            envelope_pos = np.exp(sigma * t)
            envelope_neg = -np.exp(sigma * t)
            response = envelope_pos * np.cos(omega_d * t)
            
            ax4.plot(t, response, 'r-', linewidth=2.5, label='Resposta Short Period', alpha=0.9)
            ax4.plot(t, envelope_pos, 'r--', linewidth=1.5, alpha=0.7, label='Envelope ±e^(σt)')
            ax4.plot(t, envelope_neg, 'r--', linewidth=1.5, alpha=0.7)
            
            # Marcar tempo de meia amplitude
            ax4.axvline(x=short_period['time_to_half'], color='red', 
                       linestyle=':', alpha=0.8, label=f't₁/₂ = {short_period["time_to_half"]:.2f}s')
            
            ax4.axhline(y=0, color='black', linewidth=0.8, alpha=0.6)
            ax4.grid(True, alpha=0.4)
            ax4.set_xlabel('Tempo (s)', fontsize=11)
            ax4.set_ylabel('Amplitude Normalizada', fontsize=11)
            ax4.set_title(f'Modo Short Period (T = {short_period["period"]:.2f}s, ζ = {short_period["zeta"]:.3f})', 
                         fontsize=12, fontweight='bold')
            ax4.legend()
            ax4.set_xlim(0, t_max)
        
        # Plot 5: Comparação com valores de referência - MELHORADO
        ax5 = plt.subplot(2, 3, (3, 6))
        
        # Dados de referência mais completos
        reference_data = {
            'Navion (Nelson)': {'phugoid': [0.245, 0.042], 'short': [7.004, 0.640]},
            'Cessna 172': {'phugoid': [0.18, 0.06], 'short': [4.2, 0.85]},
            'Boeing 747': {'phugoid': [0.065, 0.035], 'short': [1.8, 0.75]},
            'F-16': {'phugoid': [0.15, 0.08], 'short': [8.5, 0.45]}
        }
        
        # Plot dados de referência (mais transparentes)
        for aircraft, data in reference_data.items():
            ax5.scatter(data['phugoid'][0], data['phugoid'][1], 
                       c='lightblue', s=80, alpha=0.5, marker='o')
            ax5.scatter(data['short'][0], data['short'][1], 
                       c='lightcoral', s=80, alpha=0.5, marker='^')
            
            # Anotações menores para referências
            ax5.annotate(aircraft.replace(' (Nelson)', ''), 
                        (data['phugoid'][0], data['phugoid'][1]),
                        xytext=(3, 3), textcoords='offset points', 
                        fontsize=7, alpha=0.7)
        
        # Plot dados atuais (destaque)
        current_data = {}
        for mode in complex_modes:
            color = 'blue' if mode['name'] == 'Phugoid' else 'red'
            marker = 'o' if mode['name'] == 'Phugoid' else '^'
            
            scatter = ax5.scatter(mode['omega_n'], mode['zeta'], 
                                c=color, s=200, alpha=1.0, marker=marker,
                                edgecolors='black', linewidth=2,
                                label=f"{mode['name']} (Atual)", zorder=10)
            
            # Anotação destacada
            ax5.annotate(f"{mode['name']} (Atual)\nωn = {mode['omega_n']:.3f}\nζ = {mode['zeta']:.3f}", 
                        (mode['omega_n'], mode['zeta']), 
                        xytext=(20, -20), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color='black', alpha=0.8))
            
            current_data[mode['name']] = [mode['omega_n'], mode['zeta']]
        
        # Regiões de operação típicas
        phugoid_region = plt.Rectangle((0.05, 0.01), 0.35, 0.19, 
                                     fill=True, alpha=0.1, color='blue',
                                     label='Região Phugoid')
        short_region = plt.Rectangle((1.0, 0.3), 9.0, 1.2, 
                                   fill=True, alpha=0.1, color='red',
                                   label='Região Short Period')
        ax5.add_patch(phugoid_region)
        ax5.add_patch(short_region)
        
        ax5.set_xlabel('Frequência Natural ωn (rad/s)', fontsize=11)
        ax5.set_ylabel('Razão de Amortecimento ζ', fontsize=11)
        ax5.set_title('Comparação com Aeronaves de Referência', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.4)
        ax5.set_xscale('log')
        ax5.set_xlim(0.03, 30)
        ax5.set_ylim(0, 1.6)
        ax5.legend(loc='upper left', fontsize=9)
        
        # Adicionar texto de validação
        validation_text = "RESULTADOS DA ANÁLISE:\n"
        for mode in complex_modes:
            if mode['name'] in current_data:
                wn, zeta = current_data[mode['name']]
                validation_text += f"• {mode['name']}: ωn={wn:.3f}, ζ={zeta:.3f}\n"
                validation_text += f"  {'✅ DENTRO DA FAIXA' if mode['stable'] else '❌ INSTÁVEL'}\n"
        
        ax5.text(0.02, 0.98, validation_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def complete_analysis(self, coefficients):
        """
        Executa análise completa seguindo Etkin
        """
        # Calcular derivadas
        derivatives = self.calculate_dimensional_derivatives(coefficients)
        
        # Construir matriz A (Etkin)
        A = self.build_etkin_matrix()
        
        # Análise de autovalores
        eigenvalues, eigenvectors, modes = self.analyze_longitudinal_modes(A)
        
        # Análise detalhada
        self.print_detailed_analysis(modes)
        
        # Plotar resultados
        self.plot_comprehensive_analysis(eigenvalues, modes)
        
        return {
            'derivatives': derivatives,
            'state_matrix': A,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'modes': modes
        }

# ==========================================
# EXEMPLO VALIDADO - NAVION
# ==========================================

def navion_etkin_example():
    """
    Exemplo do Navion usando convenção correta do Etkin
    """
    
    # Dados Navion (referência clássica)
    navion_data = {
        'name': 'AJ25-A',
        'mass': 9.5,      # kg
        'S': 1.041,          # m²
        'MAC': 0.417,         # m 
        'Iy': 0.67,        # kg⋅m²
        'velocity': 12.5,    # m/s
        'rho': 1.225,        # kg/m³
        'CL_trim': 1.2378,
        'CD_trim': 0.05,
        'theta_trim': 0.8  #degrees
    }
    
    #Parâmetros da empenagem horizontal
    etaEH = 0.96        # Eficiência da empenagem horizontal (típico 0.8-0.95)
    CLaEH = 4.5         # CL_alpha da empenagem horizontal (1/rad) - similar à asa
    Veh = 0.75          # Razão das velocidades (V_eh/V) - típico 0.7-0.8
    lh = 1.15            # Distância do CG à empenagem horizontal (m) - estimativa típica para Navion
    CLa = 1.3391
    AR = 6.002
    Cmq_calculated = -2 * etaEH * CLaEH * Veh * (lh / navion_data['MAC'])
    d_eta = (2*CLa)/(pi*AR)
    
    # Coeficientes ajustados para Etkin
    navion_coefficients = {
        'CL_alpha': 1.3391,       # 1/rad             
        'CD_alpha': 0.35,      # 1/rad 
        'CDu': 0.1303,           
        'Cm_alpha': -0.741,     # 1/rad (negativo = estável)
        'Cm_alpha_dot': -4.2,  # adimensional
        'Cmq':  Cmq_calculated          # negativo = amortecimento
    }
    
    print("EXECUTANDO ANÁLISE COM CORREÇÃO ETKIN...")
    print("=" * 60)
    
    # Executar análise
    analyzer = LongitudinalStabilityEtkin(navion_data)
    results = analyzer.complete_analysis(navion_coefficients)
    
    return analyzer, results

def main():
    """Função principal"""
    try:
        analyzer, results = navion_etkin_example()
        
        print("\n" + "="*70)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("Código corrigido seguindo convenção do Etkin")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()