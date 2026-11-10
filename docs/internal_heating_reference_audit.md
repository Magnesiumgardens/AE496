# Internal-heating reference: recovered inputs and unresolved measurement

The requested study compares geometries using the same material and a 1800 K
temperature-screening limit at a consistently defined 10 mm depth, using the
existing Mach 8 spatial heating calibration. A reported 143 s threshold time
is a user-supplied reference, not a target to reproduce by changing loads or
properties. Reaching a user-defined thermal limit is not a structural-failure
prediction; not reaching it during 200 s does not establish indefinite survival
or identify the limiting component of a complete vehicle.

The subsequently supplied MK3 slide specifies **1873.15 K** (rather than the
earlier approximate 1800 K), with Mach 8 from 0–85 s and Mach 7 from 85–170 s.
Its 143 s crossing is therefore not a constant-Mach-8 benchmark. The separate
1347.15 K line is labeled a provisional temperature-rise margin. It is not a
validated structural factor of safety. The original requested 1800 K screen
remains configurable; select 1873.15 K when comparing the same temperature
criterion. The new runner does not add the reference's Mach transition.

## What the supplied reference actually contains

Reviewed `MK3_ThermalTest.wbpj`, `journal7.wbjn`, and `COPY_files.zip`, also
uploaded to the repository's main branch. Workbench session journal 3 maps
`CFD_Mach8_30km_Combined_Taw_h.csv` columns into spatial temperature (K) and heat
transfer coefficient (W/m2/K). Heat flux is ultimately marked Not Used. That
supports a convection boundary condition, not a prescribed solid temperature
history.

The archive does not include the referenced Mechanical databases (`SYS.mechdb`
or `SYS-1.mechdb`), their meshes, probe definitions, or thermal results. The
Mechanical folders are empty. The precise depth measurement, material axes,
surface/back-face conditions, and the claimed 143 s result cannot be recovered
from the uploaded project metadata alone.

## Recovered material data

`materials/reference_carbon_carbon_2d.yaml` contains the numeric Engineering
Data XML tables, rather than estimates from screenshots. Density is 1600 kg/m3;
conductivity and specific heat vary with temperature. Material X/Y are fabric
in-plane directions; Z is through-thickness. The data span 255.556 to 2200 K.
The reference labels confidence Mixed and cites a source workbook that was
not supplied. No emissivity curve was recovered. The separate tungsten entry
contains preliminary constant properties, not a verified high-temperature
tungsten table.

Changing material properties requires a new solid thermal solution. Old
tungsten temperature fields cannot be relabeled as carbon-carbon results.
Existing CFD calibration can remain the boundary-load source, with its flight
conditions and radiation assumptions recorded explicitly.

## The remaining physical distinction

- 10 mm behind the nose tip along its axis is a geometric probe offset.
- 10 mm inward normal to a particular heated surface is a different point;
  another heated surface can be closer to that point.
- A 10 mm wall measured at its inner face additionally requires the actual
  wall geometry and inner-face boundary condition.

Some thin nose sections may have no region 10 mm from every heated surface.
An invalid probe must be reported, not moved to the nearest available node.
The material's through-thickness Z direction must also be mapped to the actual
solid geometry; global XY axes do not automatically follow a curved surface.

`study.py internal-compare` implements the separate full-solid comparison,
with material tables, FE quadrature of sensible enthalpy, per-probe threshold
crossings and finite-element interpolation. The physical probe convention
and material-axis mapping must be selected explicitly before producing a
design ranking. See [settings, assumptions and run commands](internal_heating_comparison.md).
