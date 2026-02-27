# Aero Estabilidade

Projeto para análise de estabilidade longitudinal pelo método de Etkin.

## Estrutura do repositório

- `longitudinal_dinamica.py`: implementação principal do modelo, com classes de dados (`AircraftData` e `AerodynamicCoefficients`) e rotina de análise.
- `tests/test_longitudinal_dinamica.py`: teste de fumaça da execução do exemplo do Navion sem renderização de gráfico.
- `main.py`, `main2.py`, `teste.py`, `a.py`: scripts legados/experimentais.

## Executar análise

```bash
python longitudinal_dinamica.py
```

## Rodar testes

```bash
python -m pytest -q
```

> Observação: para automações (CI ou testes), use `navion_etkin_example(plot=False)` para evitar abrir janelas gráficas.
