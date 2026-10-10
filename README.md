# kCN 100 K wavelength-dependent pump–probe analysis

Analysis code for the **100 K mid-infrared pump-wavelength/frequency-dependent transient-reflectivity measurements** of insulating κ-(BEDT-TTF)₂Cu₂(CN)₃ (kCN), measured on **26 October 2023**. This repository accompanies:

> M. Tepie *et al.*, “Photoinduced frustration modulation in κ-type quantum spin liquid candidates,” *Physical Review B* **112**, L140401 (2025). https://doi.org/10.1103/clxp-z2g5

**Scope:** This is the analysis of one 100 K wavelength-dependent kCN dataset, **not** the complete data and code for every figure in the paper.

## Contents

| File | Purpose |
| --- | --- |
| `kCN_WL_dep_100K_20231026.ipynb` | Original analysis sequence: loading scans, averaging/error estimates, relaxation fits, FFTs, and coherent-phonon fits. |
| `Idler_package.py` | Pump–probe loading, analysis, fitting, and plotting utilities. |
| `osc_fit_package.py` | Multimode damped-oscillation models and fits. |
| `requirements.txt` | Python dependencies. |
| `CITATION.cff` | Repository citation metadata and associated paper. |

The submitted Python functions and analysis notebook are copied **without modifying their computational source or calling interfaces**. Notebook saved outputs and execution counters are cleared for source control; executing the notebook will regenerate outputs when experimental data are provided.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -r requirements.txt
jupyter notebook kCN_WL_dep_100K_20231026.ipynb
```

On Windows, activate with `.venv\Scripts\activate` instead.

## Data directory layout

The notebook sets its input roots relative to the current working directory using `os.getcwd()`. Run Jupyter **from this repository folder**, then put the data in these folders:

```text
kCN-100K-wavelength-dependent-analysis/
├── Idler_package.py
├── osc_fit_package.py
├── kCN_WL_dep_100K_20231026.ipynb
├── equilibrium_data/
│   └── kCN_100K_b_eq.dat
└── 20231026/
    ├── kCN_MIR-Idler_PP20231026_132006_angle0_scan0.csv
    ├── ...
    └── kCN_MIR-Idler_PP20231026_<time>_angle<angle>_scan<cycle>.csv
```

The script expects CSV filenames matching the recorded date, time, angle and scan indices; other input metadata are specified directly in the notebook. The `equilibrium_data/` and dated data directories are deliberately excluded from Git. **No experimental measurement files are included in this repository.**

## Analysis overview

1. Load 100 K equilibrium optical-conductivity reference data.
2. Load and average repeated pump–probe scans across MIR pump frequencies.
3. Estimate scan-to-scan standard errors and visualize the transient response.
4. Fit nonoscillatory backgrounds to obtain relaxation components and oscillatory residuals.
5. Fourier-transform selected residual windows; compare phonon amplitude/phase versus pump frequency.
6. Fit residuals to two damped coherent-phonon modes and inspect parameter uncertainties.

### Interpretation and reproducibility notes

- The notebook uses experimental acquisition naming conventions and local raw data that are **not** provided here. A clean installation alone will not reproduce figures until those data are supplied.
- Error bars on averaged traces represent estimates based on repeated scans; fit parameter standard errors are those reported by `lmfit`. FFT-derived quantities without a defensible uncertainty are not assigned artificial error bars.
- Frequency ordering, time-zero/sign conventions, FFT windows, and model/fit interfaces are preserved as in the uploaded analysis.
- The code was syntax-checked; a full numerical reproduction was not run because raw measurement files were not provided.

## Citation

Please cite the associated paper when using or discussing this workflow:

```bibtex
@article{Tepie2025Frustration,
  author  = {Tepie, M. and Glerean, F. and Ovčar, J. and Priya, S. and Miyagawa, K. and Taniguchi, H. and Kanoda, K. and Lončarić, I. and Dressel, M. and Mitrano, M.},
  title   = {Photoinduced frustration modulation in {\kappa}-type quantum spin liquid candidates},
  journal = {Physical Review B},
  volume  = {112},
  pages   = {L140401},
  year    = {2025},
  doi     = {10.1103/clxp-z2g5}
}
```

## License

No license has been assigned. Add a `LICENSE` file after deciding the terms under which you want to permit reuse.
