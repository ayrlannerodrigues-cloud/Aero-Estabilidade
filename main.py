import pandas as pd
from src.analysis.estatica_longitudinal import AircraftStabilityAnalyzer
from src.analysis.estatica_direcional import EstabilidadeDirecional
from src.analysis.estatica_lateral import EstabilidadeLateral
from src.database.db_manager import DBManager
from src.visualization.plotter import StabilityPlotter
from src.models.enums import FlightPhase
from src.models.dataclasses import FlightPhaseData

def main():
    # Criar gerenciador do banco de dados
    db_manager = DBManager()
    
    # Criar e executar o analisador
    analyzer = AircraftStabilityAnalyzer()
    direcional = EstabilidadeDirecional()
    lateral = EstabilidadeLateral()
    
    # Salvar resultados no banco de dados
    db_manager.save_stability_analysis(analyzer)
    
    # Salvar coeficientes aerodinâmicos detalhados
    db_manager.save_detailed_aero_coefficients(analyzer)
    
    # Recuperar e mostrar resultados de estabilidade
    results = db_manager.get_analysis_results()
    print("\nResultados da análise por fase de voo:")
    print("Fase | Velocidade | Cma_min | Cma_max | Cm0_min | Cm0_max | ME_min | ME_max | αt_min | αt_max")
    print("-" * 100)
    
    for result in results:
        print(f"{result[0]:<8} | {result[1]:>9.1f} | {result[2]:>7.3f} | {result[3]:>7.3f} | "
              f"{result[4]:>7.3f} | {result[5]:>7.3f} | {result[6]:>6.3f} | {result[7]:>6.3f} | "
              f"{result[8]:>6.1f} | {result[9]:>6.1f}")
    
    # Recuperar e mostrar coeficientes aerodinâmicos detalhados
    coef_results = db_manager.get_detailed_aero_coefficients()
    print("\nCoeficientes Aerodinâmicos Detalhados por Fase de Voo e Condição de Carga:")
    print("Fase | Carga | CL0 | eps0 | deda | Cm0_asa | Cma_asa | Cm0_eh | Cma_eh | Cm0_fus | Cma_fus | Cm0_aero | Cma_aero")
    print("-" * 120)
    
    for coef in coef_results:
        phase, load, cl0, eps0, deda = coef[0:5]
        cm0_wing, cma_wing, cm0_eh, cma_eh = coef[5:9]
        cm0_fus, cma_fus, cm0_aero, cma_aero = coef[9:13]
        
        print(f"{phase:<8} | {load:<4} | {cl0:>5.3f} | {eps0:>5.3f} | {deda:>5.3f} | "
              f"{cm0_wing:>7.3f} | {cma_wing:>7.3f} | {cm0_eh:>7.3f} | {cma_eh:>7.3f} | "
              f"{cm0_fus:>7.3f} | {cma_fus:>7.3f} | {cm0_aero:>8.3f} | {cma_aero:>8.3f}")
    
    # Exportar para CSV
    export_detailed_coefficients_to_csv(coef_results, 'coeficientes_detalhados.csv')
    
    # Criar visualizações
    plotter = StabilityPlotter()
    
    # Executar análise para todas as fases
    plotter.plot_all_phases()
    
    # Plotar comparação de margem estática e trimagem
    plotter.plot_static_margin_trim_comparison()
    plotter.plot_aircraft_points_max_load()
    plotter.plot_aircraft_points_min_load()
    plotter.plot_all()
    plotter.plot_max_load()
    plotter.plot_min_load()
    plotter.plot_stability_curves()
    
    # Flight phase data
    # Flight phase data
    phases = {
        FlightPhase.TAKEOFF: FlightPhaseData(velocity=11, Cla=0.1085, a0=-14.62, Cmac=-0.28, CLaeh=3.91, alphaf=0),
        FlightPhase.CLIMB: FlightPhaseData(velocity=13.5, Cla=0.1025, a0=-13.92, Cmac=-0.27, CLaeh=3.426, alphaf=0),
        FlightPhase.CRUISE: FlightPhaseData(velocity=15, Cla=0.0902, a0=-9.32, Cmac=-0.26, CLaeh=0.0544*57.3, alphaf=0),
        FlightPhase.DESCENT: FlightPhaseData(velocity=23.0, Cla=0.1015, a0=-13.82, Cmac=-0.27, CLaeh=3.91, alphaf=0),
        FlightPhase.LANDING: FlightPhaseData(velocity=18.0, Cla=0.1095, a0=-14.72, Cmac=-0.29, CLaeh=3.91, alphaf=0)
}
    
    # Analyze each flight phase
    for phase, data in phases.items():
        print(f"\nAnalyzing {phase.value} phase...")
        results = lateral.calculate_flight_phase(phase, data)
        plotter.plot_stability_lateral(results, phase)

def export_detailed_coefficients_to_csv(results, filename):
    """Exporta os coeficientes aerodinâmicos detalhados para um arquivo CSV"""
    data = []
    
    for result in results:
        data.append({
            'Fase de Voo': result[0],
            'Condição de Carga': result[1],
            'CL0': result[2],
            'eps0': result[3],
            'deda': result[4],
            'Cm0_asa': result[5],
            'Cma_asa': result[6],
            'Cm0_eh': result[7],
            'Cma_eh': result[8],
            'Cm0_fus': result[9],
            'Cma_fus': result[10],
            'Cm0_aero': result[11],
            'Cma_aero': result[12]
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Coeficientes exportados para {filename}")

def salvar_coeficientes_Cma(analyzer, nome_arquivo='coeficientes_Cma.csv'):
    dados = []
    
    for phase in FlightPhase:
        print(f"Analisando {phase.value}...")
        analyzer.set_flight_phase(phase)
        analyzer.run_analysis()
        
        dados.append({
            'Fase de Voo': phase.value,
            'Velocidade (m/s)': analyzer.velocity,
            'Cma Asa': analyzer.Cma_asa,
            'Cma Estabilizador': analyzer.Cma_eh,
            'Cma Fuselagem': analyzer.Cma_fus,
            'Cma Total': analyzer.Cma_aero  # Assumindo que Cma_total é Cma_aero
        })
    
    df = pd.DataFrame(dados)
    df.to_csv(nome_arquivo, index=False)
    return df

if __name__ == "__main__":
    main()