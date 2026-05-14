# Impact of FIFA World Cup on Urban Green Space in Host Cities

This repository contains the official code for the paper:

**"[Your Paper Title]"**

📄 **Paper**: [Link to paper / arXiv when available]  
👤 **Authors**: [Author Names]  
🏛️ **Affiliation**: [Institution]

## Overview

This project investigates the **causal impact of hosting the FIFA World Cup 
on urban green space** in host cities, using quasi-experimental methods.

The analysis combines:
- **Difference-in-Differences (DID)** modeling to identify the causal 
  effect of World Cup hosting on urban green space changes
- **Parallel Trends Test** to validate the DID identifying assumption
- **IFCI ([full name to be specified])** estimation as a complementary 
  measure for [briefly describe what IFCI captures in this context]

By comparing host cities with comparable non-host cities before and after 
World Cup events, this study provides empirical evidence on how mega-sports 
events shape urban ecological landscapes.

## Repository Structure

## Requirements

- **Python**: 3.8 or higher
- **OS**: Windows / macOS / Linux

### Dependencies

See `requirements.txt`. Main packages:
- `pandas` — data manipulation
- `numpy` — numerical computing
- `statsmodels` — econometric models (DID regression)
- `matplotlib` — visualization
- `scipy` — statistical tests

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/WangXGeo/code_World_Cup.git
cd code_World_Cup

# 2. (Recommended) Create a virtual environment
conda create -n didenv python=3.9
conda activate didenv

# 3. Install dependencies
pip install -r requirements.txt
```

## Data

### Data Sources
The dataset is fully open and included in this repository under `./data/`.

- **Green space data**: [e.g., Landsat NDVI / MODIS / urban green space 
  statistics from XXX database]
- **City-level covariates**: [e.g., population, GDP, urbanization rate 
  from World Bank / UN-Habitat / national statistics]
- **World Cup hosting records**: FIFA official records
- **Time period**: [e.g., 1990–2022]
- **Geographic coverage**: [e.g., XX host cities and XX control cities 
  worldwide]

### Data Structure
Required columns in `panel_data.csv`:
- `city_id`: City identifier
- `year`: Year of observation
- `treated`: Treatment indicator (1 = World Cup host city, 0 = control)
- `post`: Post-treatment period indicator (1 = after hosting, 0 = before)
- `green_space`: Urban green space measure (dependent variable)
- [other control variables...]

## Usage

Run the analysis in the following order:

### Step 1: Parallel Trends Test
Verify the DID identifying assumption — host and non-host cities should 
follow similar green space trends before the World Cup:

```bash
python code_DID_alone_Parallel_trends_test.py
```

**Output**: Pre-treatment trend plots and statistical test results.

### Step 2: Main DID Estimation
Estimate the causal effect of World Cup hosting on urban green space:

```bash
python code_DID_all.py
```

**Output**:
- Regression coefficient table
- Estimated treatment effect (DID coefficient)
- Standard errors and significance levels

### Step 3: IFCI Estimation
Compute the IFCI index for complementary analysis:

```bash
python code_IFCI.py
```

**Output**: IFCI values for each city/year.

## Results

[Optional — fill in once finalized, e.g.:]

| Outcome Variable          | DID Coefficient | Std. Error | p-value |
|---------------------------|----------------|------------|---------|
| Urban green space (NDVI)  | 0.XXX          | 0.XXX      | <0.01   |
| Green space per capita    | 0.XXX          | 0.XXX      | <0.05   |

Parallel trends test: passed (p > 0.10 for all pre-treatment periods).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026worldcup,
  title   = {[Your Paper Title]},
  author  = {Wang, X. and [Co-authors]},
  journal = {[Journal Name]},
  year    = {2026},
  doi     = {[DOI]}
}
```

## License

This project is licensed under the Apache License 2.0 — see the 
[LICENSE](LICENSE) file for details.

## Contact

For questions or issues regarding this code:
- 🐛 **Bug reports**: Open an [issue](https://github.com/WangXGeo/code_World_Cup/issues)
- 📧 **Email**: [your-email@example.com]

## Acknowledgments

[Optional — funding sources, collaborators, etc.]
