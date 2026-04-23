Quantum Espresso band-structure plotting in Python

This directory contains a Python tool to read and plot band-structure data produced by [Quantum Espresso](https://www.quantum-espresso.org/).

The script:


- Reads scf.out file to extract the Fermi energy
- Reads bands.in to extract the k-path and labels
- Reads bands.out to extract band energies
- Produces a publication-style band-structure plot with:
  - Vertical lines at high-symmetry points
  - LaTeX-style labels (Γ, K, M, …) on the x‑axis

Requirements:
- Python 3.x
- NumPy
- Matplotlib
