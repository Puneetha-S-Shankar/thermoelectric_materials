## Identity

| Column | What it means |
|--------|---------------|
| jid | JARVIS material ID — unique identifier |
| formula | Chemical formula (e.g. Bi2Te3) |
| spg_symbol | Space group symbol (e.g. Fm-3m) — describes crystal symmetry |
| spg_number | Space group number (1–230) |
| crys | Crystal system — cubic, hexagonal, tetragonal, etc. |
| dimensionality | 2D or 3D |
| bulk_dim | Same as dimensionality, added by our pipeline |
| source_db | Which JARVIS database it came from |

---

## Thermoelectric Transport Properties (computed via BoltzTrap at 600K)

| Column | What it means |
|--------|---------------|
| n-Seebeck | n-type Seebeck coefficient S (μV/K) — voltage generated per unit temperature difference for electron carriers. High negative value = good n-type thermoelectric |
| p-Seebeck | p-type Seebeck coefficient (μV/K) — same but for hole carriers. High positive value = good p-type thermoelectric |
| n-powerfact | n-type power factor S²σ (W/mK²) — combines Seebeck and conductivity. Higher = better electrical energy conversion |
| p-powerfact | p-type power factor (W/mK²) |
| nkappa | n-type thermal conductivity κ (W/mK) — how much heat the material conducts. Low κ = better ZT |
| pkappa | p-type thermal conductivity κ (W/mK) |
| ncond | n-type electrical conductivity σ (S/m) — how well electrons carry current |
| pcond | p-type electrical conductivity σ (S/m) — same for holes |

---

## Electronic Structure

| Column | What it means |
|--------|---------------|
| optb88vdw_bandgap | Electronic bandgap (eV) computed with OPT functional. Gap between valence and conduction bands. Best thermoelectrics are semiconductors with ~0.5–2 eV bandgap |
| mbj_bandgap | Bandgap (eV) with mBJ functional — more accurate than OPT, especially for small-gap materials |
| hse_gap | Bandgap (eV) with HSE06 hybrid functional — most accurate but computationally expensive, available for only 110 materials |
| avg_elec_mass | Average electron effective mass m* — heavier electrons have higher Seebeck but lower conductivity |
| avg_hole_mass | Average hole effective mass m* |
| effective_masses_300K | Full effective mass tensor at 300K — directional mass of carriers |

---

## Thermodynamic Stability

| Column | What it means |
|--------|---------------|
| formation_energy_peratom | Energy released when forming the compound from elements (eV/atom). Negative = thermodynamically favorable to form |
| ehull | Energy above the convex hull (eV/atom). 0 = perfectly stable. <0.1 eV = synthesizable. >0.1 eV = likely unstable |
| optb88vdw_total_energy | Total DFT ground state energy of the unit cell (eV) |
| exfoliation_energy | Energy needed to peel a 2D layer from bulk (eV/atom). Low value = easily exfoliable |

---

## Mechanical Properties

| Column | What it means |
|--------|---------------|
| bulk_modulus_kv | Bulk modulus K (GPa) — resistance to uniform compression. Related to lattice stiffness and thermal conductivity |
| shear_modulus_gv | Shear modulus G (GPa) — resistance to shear deformation. High G = brittle material |
| elastic_tensor | Full 6×6 elastic constant tensor Cij (GPa) — complete mechanical response in all directions |
| poisson | Poisson ratio ν — ratio of transverse to axial strain. ~0.25 for most materials. Related to ductility |

---

## Phonon Properties (related to lattice thermal conductivity)

| Column | What it means |
|--------|---------------|
| modes | Phonon frequencies at the Γ-point (cm⁻¹) — vibrational modes of the crystal. Low-frequency soft modes = low thermal conductivity κ |
| max_ir_mode | Maximum IR-active phonon frequency (cm⁻¹) — highest energy infrared-active vibration |
| min_ir_mode | Minimum IR-active phonon frequency (cm⁻¹) |

---

## Dielectric & Piezoelectric Properties

| Column | What it means |
|--------|---------------|
| epsx/y/z | Static dielectric constant ε in x, y, z directions — how much the material polarizes in an electric field. High ε = screens charges, affects carrier scattering |
| mepsx/y/z | Magnetic dielectric constant — dielectric response in magnetic materials |
| dfpt_piezo_max_dielectric | Maximum total dielectric constant (ionic + electronic) from DFPT |
| dfpt_piezo_max_dielectric_electronic | Electronic contribution to dielectric constant |
| dfpt_piezo_max_dielectric_ionic | Ionic contribution to dielectric constant |
| dfpt_piezo_max_eij | Maximum piezoelectric stress coefficient eij (C/m²) — charge generated per unit strain |
| dfpt_piezo_max_dij | Maximum piezoelectric strain coefficient dij (pC/N) — strain per unit electric field |

---

## Magnetic Properties

| Column | What it means |
|--------|---------------|
| magmom_oszicar | Magnetic moment (μB) from OSZICAR output — total magnetic moment of the unit cell |
| magmom_outcar | Magnetic moment (μB) from OUTCAR — more accurate version. Non-zero = magnetic material |

---

## Other Properties

| Column | What it means |
|--------|---------------|
| slme | Spectroscopy Limited Maximum Efficiency (%) — theoretical max solar cell efficiency. Proxy for optical absorption quality |
| spillage | Spin-orbit spillage — topological indicator. >0.5 suggests topologically non-trivial (could have protected surface states) |
| efg | Electric field gradient tensor Vzz (10²¹ V/m²) — second derivative of electrostatic potential at nucleus |
| max_efg | Maximum component of the electric field gradient |
| density | Mass density (kg/m³) — heavier materials tend to have lower thermal conductivity |
| nat | Number of atoms in the unit cell — more atoms = more phonon scattering = lower κ |
| Tc_supercon | Superconducting critical temperature Tc (K) — temperature below which material becomes superconducting |
| atoms | Full crystal structure object — lattice parameters, atomic positions, elements |
| icsd | Inorganic Crystal Structure Database reference ID — links to experimental crystal structure |

---

## DOS Features (from edos_pdos join)

| Column | What it means |
|--------|---------------|
| edos_up | Electron density of states spin-up channel — normalized, interpolated to fixed bins. Encodes electronic structure as a vector for ML |
| pdos_elast | Phonon density of states — distribution of phonon frequencies. Low-energy phonons = low lattice κ |