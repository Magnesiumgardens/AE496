"""Solved surface heat rates and time-integrated energy for calibrated loading."""
from pathlib import Path

import numpy as np

from .aerothermal_loading import integrated_powers


def cumulative_trapezoid(times, rates):
    times, rates = np.asarray(times, dtype=float), np.asarray(rates, dtype=float)
    result = np.zeros_like(rates)
    scale = np.diff(times).reshape((-1,) + (1,) * (rates.ndim - 1))
    result[1:] = np.cumsum(.5 * (rates[1:] + rates[:-1]) * scale, axis=0)
    return result


def save_surface_energy(output, load, wall_history, all_temperatures, times, settings, write_csv):
    """Keep local J/m² separate from integrated-boundary, unit-span J/m.

    Fluxes at original CFD vertices use the solved matching FE temperature.
    Global powers instead use every selected quadratic FE boundary edge and
    native spatial interpolation of both H and driving temperature. Time
    integration is trapezoidal over exported physical substeps, including the
    nonzero incoming flux at the uniform initial state.
    """
    count = int(np.count_nonzero(load.surface_active))
    times = np.r_[0., np.asarray(times, dtype=float)]
    flux_names = ("convective_heat_flux_into_solid_W_m2", "radiative_heat_flux_out_W_m2",
                  "net_heat_flux_into_solid_W_m2")
    energy_names = ("cumulative_convective_energy_J_m2", "cumulative_radiative_energy_J_m2",
                    "cumulative_net_energy_J_m2")
    rates = np.asarray([[row[name] for name in flux_names] for row in wall_history]).reshape(len(times), count, 3)
    energies = cumulative_trapezoid(times, rates)
    flux_rows, energy_rows = [], []
    for number, row in enumerate(wall_history):
        step, node = divmod(number, count)
        identity = {name: row[name] for name in ("time_s", "surface_order_index", "x_m", "y_m", "wall_temperature_K")}
        if "thermal_segment_active" in row:identity["thermal_segment_active"] = row["thermal_segment_active"]
        flux_rows.append({**identity, **dict(zip(flux_names, rates[step, node]))})
        energy_rows.append({**identity, **dict(zip(energy_names, energies[step, node]))})
        row.update(dict(zip(energy_names, map(float, energies[step, node]))))
    powers = np.asarray([integrated_powers(load, values, settings) for values in all_temperatures])
    global_energy = cumulative_trapezoid(times, powers)
    power_names = ("convective_power_per_unit_span_W_m", "radiative_power_per_unit_span_W_m", "net_power_per_unit_span_W_m")
    integrated_names = ("cumulative_convective_energy_per_unit_span_J_m", "cumulative_radiative_energy_per_unit_span_J_m",
                        "cumulative_net_energy_per_unit_span_J_m")
    global_rows = [{"time_s": float(time), **dict(zip(power_names, map(float, power))),
                    **dict(zip(integrated_names, map(float, energy)))}
                   for time, power, energy in zip(times, powers, global_energy)]
    output = Path(output)
    write_csv(output / "surface_heat_flux_history.csv", flux_rows)
    write_csv(output / "surface_energy_history.csv", energy_rows)
    write_csv(output / "global_energy_history.csv", global_rows)
    quadrature = getattr(load, "solid_energy_quadrature", None)
    if quadrature is not None:
        from .thermal_energy import evaluate_energy_balance
        from .utils import atomic_json
        solid_rows, balance = evaluate_energy_balance(quadrature, times, all_temperatures, powers, settings)
        write_csv(output / "solid_energy_history.csv", solid_rows)
        atomic_json(output / "energy_balance.json", balance)
        load.energy_balance = balance
    return float(global_energy[-1, 2])
