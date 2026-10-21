"""Explicit bar comparisons of saved steady CFD temperature samples."""
from pathlib import Path

import numpy as np

from .reporting import _save, numeric, write_csv


def steady_temperature_bars(cases, directory, plt):
    """Keep steady CFD wall and fluid-side maxima distinct from solid history."""
    if not cases:
        return {}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("wall_temperature_K", "maximum_wall_temperature_K", "raw_max_wall_temperature_comparison",
         "Maximum Steady CFD Wall Temperature", "all saved body-wall face temperature samples"),
        ("near_wall_fluid_temperature_K", "maximum_near_wall_fluid_temperature_K",
         "raw_max_near_wall_fluid_temperature_comparison", "Maximum Steady CFD Fluid-Side Surface Temperature",
         "all saved fluid-side values on body-wall faces; not the entire gas domain"),
    )
    created = {}
    for field, output_field, stem, title, sampling in specifications:
        rows = []
        for case in cases:
            cfg = case["cfg"]
            name = cfg.get("project", {}).get("name", case.get("name"))
            values = [numeric(row.get(field)) for row in case["values"]]
            if not name or not values or any(value is None or value <= 0 for value in values):
                raise ValueError(f"Steady temperature bars require finite positive {field} samples and a case name")
            rows.append(dict(design=name, **{output_field: max(values)},
                             thermal_mode=cfg["thermal"]["mode"], sampling=sampling,
                             interpretation="steady CFD solution; no physical exposure time"))
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(rows)), 5))
        bars = ax.bar(np.arange(len(rows)), [row[output_field] for row in rows],
                      color=[plt.get_cmap("tab10")(i % 10) for i in range(len(rows))])
        ax.set_xticks(np.arange(len(rows)), [row["design"] for row in rows], rotation=25, ha="right")
        ax.bar_label(bars, labels=[f"{row[output_field]:,.1f} K" for row in rows], padding=4)
        modes = {row["thermal_mode"] for row in rows}
        wall_model = "radiative-equilibrium" if modes == {"radiative_equilibrium"} else " / ".join(sorted(modes))
        ax.set(ylabel="Maximum temperature [K]", xlabel="Nose geometry",
               title=f"{title}\n({wall_model}; steady solution)")
        ax.set_ylim(0, max(row[output_field] for row in rows) * 1.15)
        ax.grid(axis="y", alpha=.3)
        ax.set_axisbelow(True)
        created[stem + "_csv"] = str(write_csv(directory / (stem + ".csv"), rows))
        created[stem + "_png"] = _save(fig, directory / (stem + ".png"), plt)
    return created
