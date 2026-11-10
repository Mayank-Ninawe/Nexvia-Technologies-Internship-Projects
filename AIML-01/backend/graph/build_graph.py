"""Build an explainable account relationship graph from PaySim transactions."""

from __future__ import annotations

import argparse
import json

import networkx as nx
import pandas as pd
from fraud_detection.config import PROJECT_ROOT, RAW_DATA_PATH

GRAPH_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "graph"
GRAPH_SUMMARY_PATH = GRAPH_OUTPUT_DIR / "graph_summary.json"
EDGE_SUMMARY_PATH = GRAPH_OUTPUT_DIR / "edge_summary.csv"
NODE_SUMMARY_PATH = GRAPH_OUTPUT_DIR / "node_summary.csv"

REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "nameDest",
    "isFraud",
]


def load_graph_window(
    max_step: int,
    max_rows: int | None,
) -> pd.DataFrame:
    """Load a chronological PaySim window for graph analysis."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at: {RAW_DATA_PATH}"
        )

    dataframe = pd.read_csv(RAW_DATA_PATH, usecols=REQUIRED_COLUMNS)
    dataframe = dataframe[dataframe["step"] <= max_step].copy()

    if max_rows is not None:
        dataframe = dataframe.head(max_rows).copy()

    if dataframe.empty:
        raise ValueError("No transactions available for the requested graph window.")

    return dataframe


def aggregate_relationships(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into directed sender-receiver relationships."""
    relationships = (
        dataframe.groupby(["nameOrig", "nameDest"], as_index=False)
        .agg(
            transaction_count=("amount", "size"),
            total_amount=("amount", "sum"),
            average_amount=("amount", "mean"),
            fraud_transaction_count=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            first_step=("step", "min"),
            last_step=("step", "max"),
        )
    )

    return relationships


def build_account_graph(relationships: pd.DataFrame) -> nx.DiGraph:
    """Create a directed graph from aggregated account relationships."""
    graph = nx.DiGraph()

    for row in relationships.itertuples(index=False):
        graph.add_edge(
            row.nameOrig,
            row.nameDest,
            transaction_count=int(row.transaction_count),
            total_amount=float(row.total_amount),
            average_amount=float(row.average_amount),
            fraud_transaction_count=int(row.fraud_transaction_count),
            fraud_rate=float(row.fraud_rate),
            first_step=int(row.first_step),
            last_step=int(row.last_step),
        )

    return graph


def build_node_summary(
    graph: nx.DiGraph,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate explainable node-level relationship metrics."""
    outgoing = (
        relationships.groupby("nameOrig", as_index=False)
        .agg(
            outgoing_transaction_count=("transaction_count", "sum"),
            outgoing_total_amount=("total_amount", "sum"),
            outgoing_fraud_transaction_count=(
                "fraud_transaction_count",
                "sum",
            ),
        )
        .rename(columns={"nameOrig": "account_id"})
    )

    incoming = (
        relationships.groupby("nameDest", as_index=False)
        .agg(
            incoming_transaction_count=("transaction_count", "sum"),
            incoming_total_amount=("total_amount", "sum"),
            incoming_fraud_transaction_count=(
                "fraud_transaction_count",
                "sum",
            ),
        )
        .rename(columns={"nameDest": "account_id"})
    )

    account_ids = pd.DataFrame({"account_id": list(graph.nodes)})
    node_summary = account_ids.merge(outgoing, on="account_id", how="left")
    node_summary = node_summary.merge(incoming, on="account_id", how="left")
    node_summary = node_summary.fillna(0)

    node_summary["out_degree"] = node_summary["account_id"].map(
        dict(graph.out_degree())
    )
    node_summary["in_degree"] = node_summary["account_id"].map(
        dict(graph.in_degree())
    )

    node_summary["total_transaction_count"] = (
        node_summary["outgoing_transaction_count"]
        + node_summary["incoming_transaction_count"]
    )
    node_summary["total_amount"] = (
        node_summary["outgoing_total_amount"]
        + node_summary["incoming_total_amount"]
    )
    node_summary["fraud_linked_transaction_count"] = (
        node_summary["outgoing_fraud_transaction_count"]
        + node_summary["incoming_fraud_transaction_count"]
    )

    node_summary["fraud_linked_ratio"] = (
        node_summary["fraud_linked_transaction_count"]
        / node_summary["total_transaction_count"].where(
            node_summary["total_transaction_count"] > 0,
            1,
        )
    )

    return node_summary.sort_values(
        ["fraud_linked_transaction_count", "total_transaction_count"],
        ascending=False,
    )


def create_graph_summary(
    graph: nx.DiGraph,
    dataframe: pd.DataFrame,
    relationships: pd.DataFrame,
    node_summary: pd.DataFrame,
    max_step: int,
    max_rows: int | None,
) -> dict:
    """Create a JSON-serializable summary of graph analysis output."""
    fraud_relationships = relationships[
        relationships["fraud_transaction_count"] > 0
    ]

    weak_components = list(nx.weakly_connected_components(graph))
    weak_component_sizes = sorted(
        [len(component) for component in weak_components],
        reverse=True,
    )

    top_fraud_linked_accounts = node_summary.head(10)[
        [
            "account_id",
            "fraud_linked_transaction_count",
            "fraud_linked_ratio",
            "in_degree",
            "out_degree",
            "total_transaction_count",
        ]
    ].to_dict(orient="records")

    top_fraud_relationships = fraud_relationships.sort_values(
        ["fraud_transaction_count", "total_amount"],
        ascending=False,
    ).head(10).to_dict(orient="records")

    return {
        "analysis_window": {
            "max_step": max_step,
            "max_rows": max_rows,
            "actual_transaction_rows": len(dataframe),
            "fraud_transaction_rows": int(dataframe["isFraud"].sum()),
        },
        "graph_metrics": {
            "node_count": int(graph.number_of_nodes()),
            "edge_count": int(graph.number_of_edges()),
            "weakly_connected_component_count": len(weak_components),
            "largest_weakly_connected_component_size": (
                int(weak_component_sizes[0]) if weak_component_sizes else 0
            ),
            "fraud_linked_relationship_count": len(fraud_relationships),
        },
        "top_fraud_linked_accounts": top_fraud_linked_accounts,
        "top_fraud_relationships": top_fraud_relationships,
        "interpretation": (
            "Fraud-linked accounts and relationships are investigation "
            "candidates based on historical labelled transactions in the "
            "selected PaySim analysis window. Graph connectivity alone does "
            "not establish a confirmed fraud ring."
        ),
    }


def save_outputs(
    summary: dict,
    relationships: pd.DataFrame,
    node_summary: pd.DataFrame,
) -> None:
    """Save graph analysis outputs locally."""
    GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(GRAPH_SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    relationships.to_csv(EDGE_SUMMARY_PATH, index=False)
    node_summary.to_csv(NODE_SUMMARY_PATH, index=False)


def run_graph_analysis(max_step: int, max_rows: int | None) -> None:
    """Build graph, create metrics, persist outputs, and print summary."""
    dataframe = load_graph_window(max_step=max_step, max_rows=max_rows)
    relationships = aggregate_relationships(dataframe)
    graph = build_account_graph(relationships)
    node_summary = build_node_summary(graph, relationships)

    summary = create_graph_summary(
        graph=graph,
        dataframe=dataframe,
        relationships=relationships,
        node_summary=node_summary,
        max_step=max_step,
        max_rows=max_rows,
    )

    save_outputs(
        summary=summary,
        relationships=relationships,
        node_summary=node_summary,
    )

    print(json.dumps(summary, indent=2))


def parse_arguments() -> argparse.Namespace:
    """Parse graph analysis command-line options."""
    parser = argparse.ArgumentParser(
        description="Build an account relationship graph from PaySim."
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=24,
        help="Use transactions through this simulation step.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200000,
        help="Optional cap on loaded rows for local graph analysis.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_graph_analysis(
        max_step=args.max_step,
        max_rows=args.max_rows,
    )