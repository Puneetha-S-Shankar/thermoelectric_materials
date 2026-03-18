# Thermoelectric Materials Dataset

ML-ready dataset for predicting thermoelectric efficiency (ZT = S²σT/κ),
built by merging 3 JARVIS materials science databases.

---

## Datasets Used

| Dataset | Size | Description |
|---|---|---|
| `dft_3d` | 75,993 | 3D bulk materials — core thermoelectric properties |
| `dft_2d` | 1,103 | 2D monolayer materials — same schema as dft_3d |
| `edos_pdos` | 48,469 | Electron and phonon density of states |

---

## Columns in Each Dataset

### dft_3d and dft_2d (66 columns)
```
jid, formula, spg_number, spg_symbol, crys, dimensionality, nat, icsd,
formation_energy_peratom, optb88vdw_total_energy, ehull, exfoliation_energy,
optb88vdw_bandgap, mbj_bandgap, hse_gap, func,
n-Seebeck, p-Seebeck, n-powerfact, p-powerfact, nkappa, pkappa, ncond, pcond,
avg_elec_mass, avg_hole_mass, effective_masses_300K,
bulk_modulus_kv, shear_modulus_gv, elastic_tensor, poisson,
modes, max_ir_mode, min_ir_mode,
epsx, epsy, epsz, mepsx, mepsy, mepsz,
dfpt_piezo_max_dielectric, dfpt_piezo_max_dielectric_electronic,
dfpt_piezo_max_dielectric_ionic, dfpt_piezo_max_eij, dfpt_piezo_max_dij,
magmom_oszicar, magmom_outcar, slme, spillage,
efg, max_efg, density, Tc_supercon, atoms,
encut, kpoint_length_unit, maxdiff_mesh, maxdiff_bz,
typ, spg, xml_data_link, raw_files, reference, search
```

### edos_pdos (5 columns)
```
jid, atoms, edos_up, pdos_elast, _source_db
```

---

## Columns We Selected

We dropped metadata, convergence parameters, and raw file links.
Kept only physically meaningful properties for ZT prediction:
```
jid, formula, spg_symbol, spg_number, crys, dimensionality, nat,
formation_energy_peratom, ehull, optb88vdw_total_energy, exfoliation_energy,
optb88vdw_bandgap, mbj_bandgap, hse_gap,
n-Seebeck, p-Seebeck, n-powerfact, p-powerfact, nkappa, pkappa, ncond, pcond,
avg_elec_mass, avg_hole_mass, effective_masses_300K,
bulk_modulus_kv, shear_modulus_gv, elastic_tensor, poisson,
modes, max_ir_mode, min_ir_mode,
epsx, epsy, epsz, mepsx, mepsy, mepsz,
dfpt_piezo_max_dielectric, dfpt_piezo_max_dielectric_electronic,
dfpt_piezo_max_dielectric_ionic, dfpt_piezo_max_eij, dfpt_piezo_max_dij,
magmom_oszicar, magmom_outcar,
slme, spillage, efg, max_efg, density, Tc_supercon, icsd,
edos_up, pdos_elast
```

**Dropped:** `encut`, `kpoint_length_unit`, `maxdiff_mesh`, `maxdiff_bz`,
`xml_data_link`, `raw_files`, `reference`, `search`, `typ`, `spg`, `func`, `atoms`

---

## Final Output

| File | Rows | Columns |
|---|---|---|
| `thermoelectric_full.csv` | 77,108 | 57 |
| `thermoelectric_ml_ready.csv` | 24,024 | 57 |

ML-ready subset = materials that have Seebeck coefficient data.
Within this subset, all 8 thermoelectric properties have 100% coverage.

---

## How to Run
```bash
pip install jarvis-tools pandas numpy matplotlib pyarrow
jupyter notebook thermoelectric_v2.ipynb
```

Run cells top to bottom. Already-cached datasets are skipped automatically on re-run.