from __future__ import annotations

from collections.abc import Callable

import pytest

from omnivault.dsa.graphs.flow import FlowNetwork, MaxFlowResult, dinic, edmonds_karp, ford_fulkerson, push_relabel

type MaxFlowAlgorithm = Callable[[FlowNetwork[str], str, str], MaxFlowResult[str]]


def _line_graph() -> FlowNetwork[str]:
    network: FlowNetwork[str] = FlowNetwork()
    network.add_edge("s", "a", 5.0)
    network.add_edge("a", "t", 3.0)
    return network


def _diamond_graph() -> FlowNetwork[str]:
    network: FlowNetwork[str] = FlowNetwork()
    network.add_edge("s", "a", 10.0)
    network.add_edge("s", "b", 10.0)
    network.add_edge("a", "c", 4.0)
    network.add_edge("b", "c", 6.0)
    network.add_edge("c", "t", 10.0)
    return network


@pytest.mark.unit
class TestFlowNetwork:
    def test_add_edge_registers_vertices(self) -> None:
        network: FlowNetwork[str] = FlowNetwork()
        network.add_edge("s", "t", 7.0)
        assert "s" in network.vertices
        assert "t" in network.vertices
        assert network.get_capacity("s", "t") == pytest.approx(7.0)

    def test_get_capacity_missing_edge_is_zero(self) -> None:
        network: FlowNetwork[str] = FlowNetwork()
        network.add_edge("s", "t", 1.0)
        assert network.get_capacity("t", "s") == pytest.approx(0.0)


@pytest.mark.unit
class TestMaxFlowAlgorithms:
    @pytest.mark.parametrize("algorithm", [ford_fulkerson, edmonds_karp, dinic, push_relabel])
    def test_line_graph_bottleneck_is_three(self, algorithm: MaxFlowAlgorithm) -> None:
        result = algorithm(_line_graph(), "s", "t")
        assert result.max_flow_value == pytest.approx(3.0)

    @pytest.mark.parametrize("algorithm", [ford_fulkerson, edmonds_karp, dinic, push_relabel])
    def test_diamond_graph_max_flow_is_ten(self, algorithm: MaxFlowAlgorithm) -> None:
        result = algorithm(_diamond_graph(), "s", "t")
        assert result.max_flow_value == pytest.approx(10.0)
