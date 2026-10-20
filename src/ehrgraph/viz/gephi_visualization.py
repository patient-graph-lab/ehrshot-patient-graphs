from pathlib import Path
import pandas as pd
import networkx as nx


def export_gephi_only_positive(tg, results, out_dir, conditions=("safe", "leaky", "null"),
                 include_raw_ids=False, largest_component_only=True, outcome=1,
                 include_neighbors=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_id = {n: i for i, n in enumerate(tg.nodes)}
    # anonymous patient index: lets you spot same-patient links in Gephi
    # without writing real patient IDs to disk
    patient_idx = {p: i for i, p in enumerate(dict.fromkeys(n[0] for n in tg.nodes))}

    # edge category relative to the other graph (shared / safe_only / leaky_only)
    s = {frozenset(e) for e in tg.graphs["safe"].edges()}
    lk = {frozenset(e) for e in tg.graphs["leaky"].edges()}

    def edge_set(u, v):
        e = frozenset((u, v))
        if e in s and e in lk:
            return "shared"
        return "safe_only" if e in s else "leaky_only" if e in lk else "null"

    # file-name tag, so a filtered export never overwrites the full one
    tag = "_".join(filter(None, [
        "lcc" if largest_component_only else "",
        f"outcome{outcome}" if outcome is not None else "",
        "plus_neighbors" if outcome is not None and include_neighbors else "",
    ]))

    stems = {}
    for cond in conditions:
        full = tg.graphs[cond]
        cent = {m: results.centrality[cond][m].to_dict()
                for m in results.centrality[cond].columns}
        leiden = results.communities[cond]["leiden"].to_dict()

        keep = set(full.nodes)
        if largest_component_only:
            keep = max(nx.connected_components(full), key=len, default=set())
        if outcome is not None:
            keep = {n for n in keep if tg.labels[n] == outcome}
            if include_neighbors:
                keep = keep | {m for n in keep for m in full[n]}
        g = full.subgraph(keep)     # kept nodes and the edges between them

        nodes = []
        for n in g.nodes:
            row = {
                "Id": node_id[n],
                "Label": node_id[n],
                "outcome": tg.labels[n],
                "patient_idx": patient_idx[n[0]],
                "leiden": leiden.get(n),
            }
            for m, d in cent.items():
                row[m] = d.get(n)
            if include_raw_ids:
                row["patient_id"] = n[0]
                row["prediction_time"] = n[1]
            nodes.append(row)

        edges = [
            {
                "Source": node_id[u],
                "Target": node_id[v],
                "Type": "Undirected",
                "Weight": d.get("weight", 1.0),
                "same_patient": int(u[0] == v[0]),
                "same_outcome": int(tg.labels[u] == tg.labels[v]),
                "edge_set": edge_set(u, v) if cond != "null" else "null",
            }
            for u, v, d in g.edges(data=True)
        ]

        stem = f"{tg.task}_{cond}" + (f"_{tag}" if tag else "")
        stems[cond] = stem
        pd.DataFrame(nodes).to_csv(out_dir / f"{stem}_nodes.csv", index=False)
        pd.DataFrame(edges).to_csv(out_dir / f"{stem}_edges.csv", index=False)
        print(f"{cond}: {len(nodes):,} of {full.number_of_nodes():,} nodes, {len(edges):,} edges")

    return stems


def csv_to_gexf_only_positive(nodes_csv, edges_csv, out_path):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    g = nx.Graph()
    for r in nodes.to_dict("records"):
        nid = int(r.pop("Id"))
        r.pop("Label", None)
        g.add_node(nid, label=str(nid), **{k: v for k, v in r.items() if pd.notna(v)})
    for r in edges.to_dict("records"):
        s, t = int(r.pop("Source")), int(r.pop("Target"))
        r.pop("Type", None)
        g.add_edge(s, t, **r)   # "Weight" -> keep as lowercase "weight" if you want Gephi to treat it as the edge weight
    nx.write_gexf(g, out_path)


def export_gephi_full_graph(tg, results, out_dir, conditions=("safe", "leaky", "null"),
                 include_raw_ids=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_id = {n: i for i, n in enumerate(tg.nodes)}
    # anonymous patient index: lets you spot same-patient links in Gephi
    # without writing real patient IDs to disk
    patient_idx = {p: i for i, p in enumerate(dict.fromkeys(n[0] for n in tg.nodes))}

    # edge category relative to the other graph (shared / safe_only / leaky_only)
    s = {frozenset(e) for e in tg.graphs["safe"].edges()}
    lk = {frozenset(e) for e in tg.graphs["leaky"].edges()}

    def edge_set(u, v):
        e = frozenset((u, v))
        if e in s and e in lk:
            return "shared"
        return "safe_only" if e in s else "leaky_only" if e in lk else "null"

    for cond in conditions:
        g = tg.graphs[cond]
        cent = {m: results.centrality[cond][m].to_dict()
                for m in results.centrality[cond].columns}
        leiden = results.communities[cond]["leiden"].to_dict()

        nodes = []
        for n in g.nodes:
            row = {
                "Id": node_id[n],
                "Label": node_id[n],
                "outcome": tg.labels[n],
                "patient_idx": patient_idx[n[0]],
                "leiden": leiden.get(n),
            }
            for m, d in cent.items():
                row[m] = d.get(n)
            if include_raw_ids:
                row["patient_id"] = n[0]
                row["prediction_time"] = n[1]
            nodes.append(row)

        edges = [
            {
                "Source": node_id[u],
                "Target": node_id[v],
                "Type": "Undirected",
                "Weight": d.get("weight", 1.0),
                "same_patient": int(u[0] == v[0]),
                "same_outcome": int(tg.labels[u] == tg.labels[v]),
                "edge_set": edge_set(u, v) if cond != "null" else "null",
            }
            for u, v, d in g.edges(data=True)
        ]

        pd.DataFrame(nodes).to_csv(out_dir / f"{tg.task}_{cond}_nodes.csv", index=False)
        pd.DataFrame(edges).to_csv(out_dir / f"{tg.task}_{cond}_edges.csv", index=False)
        print(f"{cond}: {len(nodes):,} nodes, {len(edges):,} edges")


def csv_to_gexf_full_graph(nodes_csv, edges_csv, out_path):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    g = nx.Graph()
    for r in nodes.to_dict("records"):
        nid = int(r.pop("Id"))
        r.pop("Label", None)
        g.add_node(nid, label=str(nid), **{k: v for k, v in r.items() if pd.notna(v)})
    for r in edges.to_dict("records"):
        s, t = int(r.pop("Source")), int(r.pop("Target"))
        r.pop("Type", None)
        g.add_edge(s, t, **r)   # "Weight" -> keep as lowercase "weight" if you want Gephi to treat it as the edge weight
    nx.write_gexf(g, out_path)