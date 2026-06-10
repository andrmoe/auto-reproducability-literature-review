from __future__ import annotations

from datetime import datetime
import calendar
import re
import textwrap

import bibtexparser
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
import networkx as nx


def parse_bib_month(month_value: str | None) -> int:
    if not month_value:
        return 1

    value = month_value.strip().lower()
    value = value.strip("{}")

    month_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    if value in month_map:
        return month_map[value]

    if value.isdigit():
        month_num = int(value)
        if 1 <= month_num <= 12:
            return month_num

    for i, name in enumerate(calendar.month_name):
        if name and name.lower() == value:
            return i

    for i, name in enumerate(calendar.month_abbr):
        if name and name.lower() == value:
            return i

    return 1


def parse_pub_date(entry: dict) -> datetime:
    date_field = entry.get("date")
    if date_field:
        date_field = date_field.strip().strip("{}")
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                parsed = datetime.strptime(date_field, fmt)
                if fmt == "%Y":
                    return datetime(parsed.year, 1, 1)
                if fmt == "%Y-%m":
                    return datetime(parsed.year, parsed.month, 1)
                return parsed
            except ValueError:
                pass

    year = entry.get("year")
    if not year:
        raise ValueError(
            f'Missing year/date in BibTeX entry {entry.get("ID", "<unknown>")}'
        )

    year_match = re.search(r"\d{4}", str(year))
    if not year_match:
        raise ValueError(
            f'Could not parse year in BibTeX entry {entry.get("ID", "<unknown>")}'
        )

    year_num = int(year_match.group(0))
    month_num = parse_bib_month(entry.get("month"))
    return datetime(year_num, month_num, 1)


def load_bib_entries(bib_path: str) -> dict[str, dict]:
    with open(bib_path, "r", encoding="utf-8") as f:
        bib_database = bibtexparser.load(f)

    return {entry["ID"]: entry for entry in bib_database.entries}


def clean_title(title: str) -> str:
    title = title.replace("{", "").replace("}", "")
    return " ".join(title.split())


def build_graph(bib_path: str) -> nx.DiGraph:
    entries = load_bib_entries(bib_path)
    g = nx.DiGraph()

    papers = {
        "artisan": (100.0, 2, "auto-reproducibility", ["super", "heye2025", "corebench", "liang2024", "paperbench", "paper2code", "liang2024"]),
        "lin2025": (190.0, 1, "auto-reproducibility", ["autop2c", "paperbench", "paper2code", "autoreproduce"]),
        "paper2code": (300.0, 1, "auto-reproducibility", ["paperbench", "scireplicatebench"]),
        "paperbench": (180.0, 0, "validation", ["corebench"]),
        "autop2c": (30.0, 2, "auto-reproducibility", []),
        "heye2025": (140.0, 1, "auto-reproducibility", ["reproscreener"]),
        "super": (110.0, 1, "validation", []),
        "corebench": (380.0, 1, "validation", []),
        "replicationbench": (320.0, 1, "validation", ["corebench", "scireplicatebench"]),
        "replicatorbench": (380.0, 2, "validation", ["corebench"]),
        "socscireprobench": (410.0, 2, "validation", ["corebench"]),
        "scireplicatebench": (200.0, 0, "validation", ["paper2code", "paperbench", "paper2code", "corebench"]),
        "autoreproduce": (300.0, 0, "auto-reproducibility", ["lin2025", "super", "paperbench", "paper2code", "corebench"]),
        "shah2026": (180.0, 0, "auto-reproducibility", ["corebench", "paperbench"]),
        "kubota2026": (60.0, 1, "journal policy", ["shah2026"]),
        "liang2024": (80.0, 3, "auto-reproducibility", []),
        "jailbreakfoundry": (350.0, 1, "auto-reproducibility", ["scireplicatebench"]),
        "envbench": (400, 2, "validation", ["corebench"]),
        "reproscreener": (210, 0, "validation", ["heye2025"]),
        "autoexperiment": (140, 1, "validation", ["paperbench", "super"])
    }

    for key, (y, generation, category, citations) in papers.items():
        if key not in entries:
            raise KeyError(f"BibTeX key {key!r} not found in {bib_path}")

        entry = entries[key]
        title = clean_title(entry.get("title", key))
        pub_date = parse_pub_date(entry)

        g.add_node(
            key,
            title=title.split(':')[0],
            pub_date=pub_date,
            generation=generation,
            category=category,
            y=300-140*generation+ 40*(pub_date.month % 4),
        )
        for citation in citations:
            g.add_edge(key, citation, citers_generation=min(generation, papers[citation][1]))

    return g


def compute_positions(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    return {
        node: (mdates.date2num(data["pub_date"]), data["y"])
        for node, data in graph.nodes(data=True) if data
    }


def make_label(data: dict) -> str:
    title = textwrap.fill(data["title"], width=28)
    year_month = data["pub_date"].strftime("%Y-%m")
    return f"{title}\n({year_month})"

generation_colors = ["#6666ff", "#00aa00", "#ff6666", "#888888"]
category_colors = {
    "auto-reproducibility": "#ffffaa",
    "validation": "#aaffff",
    "journal policy": "#ffaaff"
}

def draw_graph(graph: nx.DiGraph, output: str = "temp/citation_graph.pdf") -> None:
    pos = compute_positions(graph)
    fig, ax = plt.subplots(figsize=(16, 10))

    for source, target in graph.edges():
        x1, y1 = pos[source]
        x2, y2 = pos[target]

        rad = 0.10
        if y2 < y1:
            rad = -0.10

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2+20, y2),
            arrowstyle="-|>",
            mutation_scale=20,
            linewidth=1.5,
            color=generation_colors[graph.edges[source, target]["citers_generation"]],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=18,
            shrinkB=18,
            alpha=0.9,
            zorder=1,
        )
        ax.add_patch(arrow)
    legend_elements = []
    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        label = make_label(data)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=12,
            family="serif",
            zorder=3,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.15",
                "facecolor": category_colors[data["category"]],
                "edgecolor": generation_colors[data["generation"]],
                "linewidth": 4,
            },
        )
    legend_elements = [Patch(facecolor=color, edgecolor="black", label=category) for category, color in category_colors.items()] + \
                      [Patch(facecolor=generation_colors[generation], edgecolor="black", 
                          label="Seed papers" if generation == 0 else f"{generation}. generation") 
                          for generation in range(4)]
    ax.legend(handles=legend_elements, title="Legend")
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    ax.set_xlabel("Publication date")
    ax.set_title("Citation graph")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax.margins(x=0.05, y=0.08)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    graph = build_graph("refs.bib")
    draw_graph(graph)