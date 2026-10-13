"""
G.O.A.T. Shield Stage 2A - NetworkX Cross-Bank Transaction Graph Engine
Constructs directed multi-institution transaction topologies and serializes
node-link graph payloads with topological analysis.
"""

from typing import List, Dict, Any, Optional
import networkx as nx


def build_graph(scored_transactions: List[Dict[str, Any]], accounts_dict: Optional[Dict[str, List[str]]] = None) -> nx.MultiDiGraph:
    """
    Constructs a NetworkX MultiDiGraph from scored transactions and accounts.
    """
    G = nx.MultiDiGraph()

    # Pre-index accounts across banks
    from engine.simulator import generate_accounts, get_bank_for_account
    all_accounts_dict = accounts_dict if accounts_dict is not None else generate_accounts()

    # Initialize all 150 accounts as nodes
    for bank_name, accounts in all_accounts_dict.items():
        for acc in accounts:
            G.add_node(
                acc,
                id=acc,
                bank=bank_name,
                tx_count=0,
                total_sent=0.0,
                total_received=0.0,
                is_mule_node=False,
                max_risk_score=0,
                status="Normal",
                mule_rings=set()
            )

    node_dict = G._node
    # Populate transactions as edges and update node metrics
    for t in scored_transactions:
        sender = t["sender"]
        receiver = t["receiver"]
        amt = t["amount"]
        score = t.get("risk_score", 0)
        status = t.get("status", "Normal")
        is_mule = t.get("is_mule", False)
        ring_id = t.get("mule_ring_id")

        # Ensure nodes exist
        if sender not in node_dict:
            G.add_node(sender, id=sender, bank=t["bank"], tx_count=0, total_sent=0.0, total_received=0.0, is_mule_node=False, max_risk_score=0, status="Normal", mule_rings=set())
        if receiver not in node_dict:
            r_bank = get_bank_for_account(receiver)
            G.add_node(receiver, id=receiver, bank=r_bank, tx_count=0, total_sent=0.0, total_received=0.0, is_mule_node=False, max_risk_score=0, status="Normal", mule_rings=set())

        # Update sender stats
        s_node = node_dict[sender]
        s_node["tx_count"] += 1
        s_node["total_sent"] += amt
        if score > s_node["max_risk_score"]:
            s_node["max_risk_score"] = score
            s_node["status"] = status
        if is_mule:
            s_node["is_mule_node"] = True
            if ring_id:
                s_node["mule_rings"].add(ring_id)

        # Update receiver stats
        r_node = node_dict[receiver]
        r_node["tx_count"] += 1
        r_node["total_received"] += amt
        if score > r_node["max_risk_score"]:
            r_node["max_risk_score"] = score
            r_node["status"] = status
        if is_mule:
            r_node["is_mule_node"] = True
            if ring_id:
                r_node["mule_rings"].add(ring_id)

        # Add directed edge
        G.add_edge(
            sender,
            receiver,
            key=t["id"],
            id=t["id"],
            source=sender,
            target=receiver,
            amount=amt,
            currency=t.get("currency", "INR"),
            timestamp=t.get("timestamp", ""),
            type=t.get("type", "IMPS"),
            is_cross_bank=t.get("is_cross_bank", False),
            is_mule=is_mule,
            mule_ring_id=ring_id,
            risk_score=score,
            status=status,
            triggered_rules=t.get("triggered_rules", [])
        )

    return G


def get_graph_data(
    scored_transactions: List[Dict[str, Any]],
    cross_bank_only: bool = False,
    ring_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Serializes NetworkX graph to a JSON-ready payload for dashboard visualization.
    """
    G = build_graph(scored_transactions)

    # Compute network-wide metrics on full graph
    # Convert MultiDiGraph to simple DiGraph for density calculation
    simple_digraph = nx.DiGraph(G)
    density = round(nx.density(simple_digraph), 6)
    
    # Calculate connected components
    undirected = G.to_undirected()
    components_count = nx.number_connected_components(undirected)

    # Filter edges if requested
    filtered_edges = []
    active_node_ids = set()

    for u, v, key, data in G.edges(keys=True, data=True):
        if cross_bank_only and not data.get("is_cross_bank", False):
            continue
        if ring_id and data.get("mule_ring_id") != ring_id:
            continue

        edge_dict = {
            "id": data["id"],
            "source": u,
            "target": v,
            "amount": data["amount"],
            "currency": data["currency"],
            "timestamp": data["timestamp"],
            "type": data["type"],
            "is_cross_bank": data["is_cross_bank"],
            "is_mule": data["is_mule"],
            "mule_ring_id": data["mule_ring_id"],
            "risk_score": data["risk_score"],
            "status": data["status"],
            "triggered_rules": data.get("triggered_rules", [])
        }
        filtered_edges.append(edge_dict)
        active_node_ids.add(u)
        active_node_ids.add(v)

    # Serialize nodes
    nodes_list = []
    mule_nodes_count = 0

    for n, data in G.nodes(data=True):
        is_active = (n in active_node_ids) if (cross_bank_only or ring_id) else True
        if not is_active:
            continue

        if data.get("is_mule_node", False):
            mule_nodes_count += 1

        nodes_list.append({
            "id": n,
            "bank": data.get("bank", "Unknown"),
            "tx_count": data.get("tx_count", 0),
            "total_sent": round(data.get("total_sent", 0.0), 2),
            "total_received": round(data.get("total_received", 0.0), 2),
            "is_mule_node": data.get("is_mule_node", False),
            "max_risk_score": data.get("max_risk_score", 0),
            "status": data.get("status", "Normal"),
            "mule_rings": list(data.get("mule_rings", []))
        })

    # Top degree accounts
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
    out_degrees = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:5]

    top_hubs = [
        {"account": acc, "in_degree": deg, "bank": G.nodes[acc].get("bank", ""), "is_mule": G.nodes[acc].get("is_mule_node", False)}
        for acc, deg in in_degrees
    ]

    top_sources = [
        {"account": acc, "out_degree": deg, "bank": G.nodes[acc].get("bank", ""), "is_mule": G.nodes[acc].get("is_mule_node", False)}
        for acc, deg in out_degrees
    ]

    cross_bank_total = sum(1 for _, _, _, d in G.edges(keys=True, data=True) if d.get("is_cross_bank", False))
    mule_edges_total = sum(1 for _, _, _, d in G.edges(keys=True, data=True) if d.get("is_mule", False))

    stats = {
        "total_nodes": len(G.nodes),
        "total_edges": len(G.edges),
        "displayed_nodes": len(nodes_list),
        "displayed_edges": len(filtered_edges),
        "cross_bank_edges": cross_bank_total,
        "mule_nodes": mule_nodes_count,
        "mule_edges": mule_edges_total,
        "graph_density": density,
        "connected_subgraphs": components_count,
        "top_inbound_hubs": top_hubs,
        "top_outbound_sources": top_sources
    }

    return {
        "nodes": nodes_list,
        "edges": filtered_edges,
        "stats": stats
    }
