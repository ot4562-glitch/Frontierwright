# Frontierwright Third-Party Notices

Frontierwright itself is licensed under the Apache License 2.0.

The Frontierwright source distribution and wheel do **not** vendor the source code
or binary payloads of the Python dependencies listed below. They are resolved and
installed as separate packages by the user's Python package installer. Each package
remains subject to its own license terms and ships its own authoritative license
materials.

## Direct runtime dependencies

| Package | Frontierwright requirement | License metadata |
| --- | --- | --- |
| Typer | `typer>=0.16,<1` | MIT |
| Textual | `textual>=1,<9` | MIT |
| Rich | `rich>=13,<16` | MIT |

## Optional training dependencies

| Package | Frontierwright requirement | License metadata |
| --- | --- | --- |
| PyTorch | `torch>=2.4,<3` | Package metadata declares a compound expression including Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0, and MIT components |
| psutil | `psutil>=6,<8` | BSD-3-Clause |

The exact licenses for transitive dependencies depend on the versions selected by
the installer. Frontierwright does not relicense those packages.

For redistribution scenarios that bundle third-party wheels, shared libraries,
model assets, datasets, or other external artifacts together with Frontierwright,
the redistributor must include the corresponding upstream license and notice files
required by those artifacts. This repository does not currently bundle such
third-party payloads into the Frontierwright wheel or sdist.

This notice is informational and does not replace the authoritative license text
distributed by each dependency.
