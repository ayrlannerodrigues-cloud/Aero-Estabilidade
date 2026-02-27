from longitudinal_dinamica import navion_etkin_example


def test_navion_example_without_plot_returns_structured_results():
    _, results = navion_etkin_example(plot=False)

    assert "derivatives" in results
    assert "state_matrix" in results
    assert "eigenvalues" in results
    assert "modes" in results
    assert len(results["modes"]) >= 2
