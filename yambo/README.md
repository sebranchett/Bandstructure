Yambo band-structure plotting in Python

This directory contains a Python tool to read and plot band-structure data produced by [Yambo)](https://www.yambo-code.eu/).

The script:

- Reads interpolated DFT, GW and BSE bands from YAMBO post processing calculation files
- Reads the high symmetry labels from the YAMBO post processing interpolated DFT file
- Produces a publication-style band-structure plot with:
  - Vertical lines at high-symmetry points
  - LaTeX-style labels (Γ, K, M, …) on the x‑axis

Requirements:
- Python 3.x
- NumPy
- Matplotlib
- Pandas
