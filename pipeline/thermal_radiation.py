"""Radiation-to-ambient surfaces attached to the existing quadratic solid mesh."""
from __future__ import annotations

import numpy as np

from .thermal_loading import ThermalAnalysisError

STEFAN_BOLTZMANN = 5.670374419e-8
SOLID_NODES = "THERM_SOLID_NODES"
SOLID_ELEMENTS = "THERM_SOLID_ELEMENTS"


def _checked(response, operation):
    text = str(response).upper()
    if "*** ERROR ***" in text or "*** WARNING ***" in text:
        raise ThermalAnalysisError(f"MAPDL radiation {operation} was not accepted:\n{response}")


def add_radiation_surfaces(mapdl, lines, load, nnum, xyz, settings):
    """Keep CONV on PLANE293 and put radiation on dedicated SURF151 faces.

    ESURF retains native corner/midside ordering. The ambient node is explicitly
    excluded from the face selection and never enters solid result statistics.
    The radiation material has no density or heat capacity.
    """
    if not load.boundary_quadrature:
        raise ThermalAnalysisError("Radiation requires verified quadratic boundary edges")
    mapdl.allsel()
    mapdl.cm(SOLID_NODES, "NODE")
    mapdl.cm(SOLID_ELEMENTS, "ELEM")
    ambient = int(np.max(nnum)) + 1
    span = max(float(np.ptp(xyz[:, :2], axis=0).max()), 1.)
    mapdl.n(ambient, float(xyz[:, 0].max() + span), float(xyz[:, 1].max() + span), 0.)
    _checked(mapdl.et(2, "SURF151"), "element definition")
    _checked(mapdl.keyopt(2, 3, 0), "planar option")
    _checked(mapdl.keyopt(2, 4, 0), "midside option")
    _checked(mapdl.keyopt(2, 5, 1), "ambient node option")
    _checked(mapdl.keyopt(2, 9, 1), "radiation option")
    _checked(mapdl.r(2, 1., STEFAN_BOLTZMANN), "form factor / Stefan constant")
    _checked(mapdl.mp("EMIS", 2, settings["emissivity"]), "emissivity")
    mapdl.type(2)
    mapdl.real(2)
    mapdl.mat(2)
    mapdl.esel("S", "TYPE", "", 1)
    mapdl.lsel("NONE")
    mask = getattr(load, "segment_active", np.ones(len(lines), dtype=bool))
    for index, line in enumerate(lines):
        if mask[index]:
            mapdl.lsel("A", "LINE", "", int(line))
    mapdl.nsll("S", 1)
    _checked(mapdl.esurf(ambient), "surface generation")
    mapdl.esel("S", "TYPE", "", 2)
    generated = list(mapdl.mesh.elem)
    expected = {tuple(sorted(map(int, np.asarray(nnum)[edge[0]])))
                for edge in load.boundary_quadrature}
    actual = []
    for record in generated:
        record = np.asarray(record, dtype=int)
        nodes = set(map(int, record[10:])) - {0}
        if len(record) < 14 or int(record[1]) != 2 or ambient not in nodes or len(nodes) != 4:
            raise ThermalAnalysisError("SURF151 is missing an ambient or quadratic face node")
        actual.append(tuple(sorted(nodes - {ambient})))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ThermalAnalysisError("Radiation surfaces do not match every exposed solid edge exactly; rear cuts must remain unloaded")
    mapdl.allsel()
    # This is a surroundings temperature, never a prescribed SOLID temperature.
    mapdl.d(ambient, "TEMP", settings["trad"])
    mapdl.type(1)
    mapdl.mat(1)
    mapdl.real(1)
    return dict(status="configured", formulation="SURF151", surface_elements=len(actual),
                ambient_node_id=ambient, ambient_temperature_K=settings["trad"],
                emissivity=settings["emissivity"], form_factor=1.,
                stefan_boltzmann_W_m2K4=STEFAN_BOLTZMANN,
                ambient_excluded_from_solid_results=True,
                solid_temperature_constraints=0)


def select_solid_results(mapdl, nnum):
    """Read only solid nodes/elements, including their quadratic midside nodes."""
    mapdl.cmsel("S", SOLID_ELEMENTS)
    mapdl.cmsel("S", SOLID_NODES)
    if not np.array_equal(np.asarray(mapdl.mesh.nnum, dtype=int), nnum):
        raise ThermalAnalysisError("Solid result selection changed after adding the radiation ambient node")
