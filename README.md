# Granular Material Simulation Analysis

This repository contains Python scripts and data for analyzing simulations of granular materials composed of rods, crosses, stars, and spheres. The analysis focuses on force distributions, maximum packing heights, overlap calculations, and contact statistics, supporting the findings presented in the associated scientific article.

## Repository Structure

### `contact/`
- `contact.py`: Script for analyzing contact statistics between spheres and rods/crosses/stars.
- `contact_mean_std_sph_p_rod.txt`: Mean and standard deviation data for contacts per sphere per rod.

### `forces_distribution/`
- `forces_distribution.py`: Main script for analyzing force distributions (normal and tangential forces).
- `forces_cache/`: Directory containing cached force data (large files stored via Git LFS).
  - `forces_cross.parquet`, `forces_star.parquet`: Force data for cross and star shapes.
  - `forces_rod.parquet.part_*`: Split parts of the large rod force data file (reconstruct with `cat forces_rod.parquet.part_* > forces_rod.parquet`).
- `plots/`: Output directory for generated plots.
  - `fit_parameters/`: Fitted parameters for force distributions (rod, cross, star).

### `maximum_height_packing/`
- `max_height_packing.py`: Script for analyzing maximum packing heights and packing fractions.
- `maximum_height_mean_std_n_spheres_per_rod.txt`: Mean and std data for maximum heights.
- `packing_fraction_mean_std_n_spheres_per_rod.txt`: Mean and std data for packing fractions.

### `overlap/`
- `deltan.py`: Script for calculating overlap parameters and material properties.
- `plotting_utils_overlap.py`: Utility functions for overlap plotting.
- `cleaned_overlap_density.csv`: Cleaned overlap density data.
- `overlap_density_meaned_distrib_new_data.txt`: Processed overlap density distributions.

## Data Files

Large data files are stored using Git LFS. To access them:
1. Install Git LFS: `git lfs install`
2. Clone the repository: `git clone https://github.com/GRASP-LAB/Mesocopic-Roughness.git`
3. The large files will be downloaded automatically.

For the split rod data file, reconstruct it locally:
```bash
cd forces_distribution/forces_cache
cat forces_rod.parquet.part_* > forces_rod.parquet
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy

Install dependencies using pip:
```bash
pip install numpy pandas matplotlib scipy
```

## Usage

Each analysis script can be run independently. Navigate to the respective folder and execute the Python script:

```bash
cd contact
python contact.py

cd ../forces_distribution
python forces_distribution.py

cd ../maximum_height_packing
python max_height_packing.py

cd ../overlap
python deltan.py
```

Plots and fitted parameters will be saved in the respective directories.

## Data Format

Simulation data is stored in text files with the following formats:
- Mean/std data: `# type\nn_spheres mean std`
- Force data: Cached numpy arrays in `forces_cache/`
- Overlap data: CSV format with density distributions

## Citation

If you use this code or data in your research, please cite the associated scientific article:

[Add citation details here]
=======
# Mesocopic-Roughness
Analysis scripts and data for granular material simulations involving rods, crosses, stars, and spheres. Includes force distributions, packing heights, overlap calculations, and contact analysis supporting scientific article findings.
>>>>>>> b74bf789c3dc0de478a160f1a00de209093bf249
