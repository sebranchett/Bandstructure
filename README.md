AMS BAND band-structure plotting in Python

This repository contains a small Python tool to read and plot band-structure data produced by AMS BAND (Amsterdam Modeling Suite, BAND module).

The script:

Reads .gnuplot band-structure files exported from AMS BAND
Converts energies from Hartree to eV and applies an optional energy shift (e.g. to set the Fermi level or VBM to 0 eV)
“Stitches” together individual k‑path segments (e.g. Γ–K, K–M, …) into a continuous 1D k‑axis
Automatically detects the band closest to 0 eV and brings it to the top of the list 
Produces a publication-style band-structure plot with:
Vertical lines at high-symmetry points
LaTeX-style labels (Γ, K, M, …) on the x‑axis
Customizable y‑axis limits, colors, and line styles

The code is deliberately written with new Python users in mind:
functions are short, variable names are descriptive, and there are many inline comments explaining what each step does.

Requirements:
Python 3.x
NumPy
Matplotlib
