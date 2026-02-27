# main.py

from analysis.estatica_longitudinal import AircraftStabilityAnalyzer
from models.enums import FlightPhase

def main():
    # Inicializa o analisador de estabilidade
    analyzer = AircraftStabilityAnalyzer()
    
    # 1. ESCOLHA A FASE DE VOO A SER ANALISADA
    # Você pode alterar a fase aqui. As opções são:
    # FlightPhase.TAKEOFF, FlightPhase.CLIMB, FlightPhase.CRUISE, FlightPhase.DESCENT, FlightPhase.LANDING
    phase_to_analyze = FlightPhase.CRUISE

    # Configura o analisador para a fase de voo escolhida
    analyzer.set_flight_phase(phase_to_analyze)

    # Executa a análise para a fase selecionada
    print(f"--- Análise de Estabilidade Estática Longitudinal para a fase: {phase_to_analyze.name} ---")

    # Obtém e mostra os coeficientes aerodinâmicos
    analyzer.get_aerodynamic_coefficients()

    # Calcula e mostra o centro de gravidade
    analyzer.calculate_and_show_cg()

    # Realiza a análise de estabilidade e calcula a margem estática
    analyzer.stability_analysis()
    
    # Adiciona os resultados da análise estática ao banco de dados
    analyzer.add_static_analysis_results_to_db()

    print(f"\nAnálise para a fase {phase_to_analyze.name} concluída.")

if __name__ == "__main__":
    main()