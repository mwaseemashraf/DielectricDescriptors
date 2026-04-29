# Plot choices

- `descriptor_property_scatter.png`: descriptor-property scatter plots, following the local-structure/order-parameter descriptor literature style for assessing whether a descriptor separates or trends with a target property.
- `physics_descriptor_property_scatter.png`: the same descriptor-property scatter format for soft-mode, lattice, neighbor-variation, and relaxation descriptors motivated by dielectric-response literature.
- `descriptor_family_parity.png`: calculated-vs-predicted parity plots, following cluster-expansion validation practice for testing descriptor families against DFT targets.
- The target used for performance is `trace(eps_ionic)/3`, the orientational average of the parsed ionic dielectric tensor. The full tensor is preserved in `vasp_dielectric_summary.csv`.
- The added soft-mode descriptors follow the DFPT relation in which ionic susceptibility is controlled by mode effective charges divided by phonon frequency squared. The added distortion descriptors follow oxide-permittivity ML studies that identify local geometric asymmetry and neighbor-distance variation as important features.
