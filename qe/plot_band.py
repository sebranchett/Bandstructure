import numpy as np
import matplotlib.pyplot as plt
import re

# -----------------------------
# Extract Fermi Level from SCF
# -----------------------------
def get_fermi_from_scf(scf_filename):
    """Parses the highest occupied level from scf.out."""
    highest_occupied = 0.0
    try:
        with open(scf_filename, "r") as f:
            for line in f:
                if "highest occupied, lowest unoccupied level (ev):" in line:
                    parts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                    if parts:
                        highest_occupied = float(parts[0])
                        print(f"Found Fermi Level (Highest Occupied): {highest_occupied} eV")
                        return highest_occupied
    except FileNotFoundError:
        print(f"Warning: {scf_filename} not found. Using 0.0 as Fermi Level.")
    return highest_occupied

# -----------------------------
# Read k-path from bands.in
# -----------------------------
def read_kpath_from_bands_in(filename):
    labels = []
    num_points = []
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
        
        start_idx = -1
        for i, line in enumerate(lines):
            if "K_POINTS" in line:
                start_idx = i + 2
                break
        
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 4:
                num_points.append(int(parts[3]))
                label = parts[4].strip().replace('!', '')
                if label.lower() == 'gamma': label = r'$\Gamma$'
                labels.append(label)
        return labels, num_points
    except:
        return [], []

# -----------------------------
# Read bands.out (Cleaned Logic)
# -----------------------------
def read_bands_out(filename):
    kpoints = []
    bands = []
    current_energies = []
    float_regex = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+\.\d+|[-+]?\d+")

    with open(filename, "r") as f:
        lines = f.readlines()

    start_reading = False
    for line in lines:
        if "End of band structure calculation" in line:
            start_reading = True
            kpoints, bands, current_energies = [], [], [] 
            continue
        
        if not start_reading:
            continue

        if "k =" in line:
            if current_energies:
                bands.append(current_energies)
                current_energies = []
            
            k_part = line.split('(')[0]
            coords = [float(x) for x in float_regex.findall(k_part)]
            if len(coords) >= 3:
                kpoints.append(coords[-3:])

        elif "bands (ev):" in line or not line.strip():
            continue
        else:
            if "(" not in line:
                energies = [float(x) for x in float_regex.findall(line)]
                current_energies.extend(energies)

    if current_energies:
        bands.append(current_energies)

    min_k = min(len(kpoints), len(bands))
    kpoints = np.array(kpoints[:min_k])
    min_b = min(len(b) for b in bands[:min_k])
    bands = np.array([b[:min_b] for b in bands[:min_k]])

    print(f"Final Count: {len(kpoints)} k-points, {bands.shape[1]} bands.")
    return kpoints, bands

# -----------------------------
# Plotting
# -----------------------------
def plot_band_structure(bands_in, bands_out, scf_out):
    fermi_energy = get_fermi_from_scf(scf_out)
    labels, num_points = read_kpath_from_bands_in(bands_in)
    kpoints, energies = read_bands_out(bands_out)
    
    # Compute Distance
    dk = np.linalg.norm(kpoints[1:] - kpoints[:-1], axis=1)
    kdist = np.concatenate([[0.0], np.cumsum(dk)])

    plt.figure(figsize=(7, 6))
    
    # Shift energies relative to Fermi
    for i in range(energies.shape[1]):
        plt.plot(kdist, energies[:, i] - fermi_energy, color="black", linewidth=1.0)

    # Vertical Symmetry Lines logic
    tick_pos = [0]
    curr = 0
    for n in num_points:
        curr += n
        if curr < len(kdist):
            tick_pos.append(curr)
    
    if (len(kdist)-1) not in tick_pos:
        tick_pos.append(len(kdist)-1)

    # Vertical lines and x-ticks
    for p in tick_pos:
        plt.axvline(kdist[p], color="gray", linestyle="--", alpha=0.5)

    plt.xticks([kdist[p] for p in tick_pos], labels[:len(tick_pos)])
    
    # Fermi Level line
    plt.axhline(0, color="red", linestyle=":", linewidth=1.5, label="Fermi Level")
    
    # Axis formatting
    plt.ylabel("Energy - $E_{VBM}$ (eV)")
    plt.xlabel("k-path")
    #plt.title(f"Band Structure (Shifted by {fermi_energy} eV)")
    
    # SET Y-AXIS LIMITS HERE
    plt.ylim(-4, 4)
    
    plt.xlim(0, kdist[-1])
    plt.grid(False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_band_structure(
        bands_in="bands.in",
        bands_out="bands.out",
        scf_out="scf.out"
    )
