# 🧬 Ligand Metric Comparison

This repository explores and compares different metrics for evaluating ligands.
Ligand validation is essential in structural biology and drug discovery — and while the **Q-score** is widely used, it has limitations, especially with small molecules.

To improve evaluation robustness, we test and analyze several metrics:

- **Q-score** – the current standard, but sensitive to resolution and background noise.
- **Wasserstein Distance (WD)** – spatially-aware and effective in detecting structural changes.
- **Total Variation Distance (TVD)** – efficient but sometimes too insensitive to small changes.

🧪 We evaluate how each metric performs in various conditions:
- Symmetry and consistency
- Sensitivity to added/missing density
- Positional shifts of ligands
- Background noise robustness

Both **synthetic** and **real ligand data** are used in the experiments to assess how well each metric captures structural similarity.

---

**Authors:** Martyna Stasiak, Maria Musiał, Patryk Janiak, Mateusz Bernart  
**Supervisor:** Dariusz Brzeziński  
**Affiliation:** Poznań University of Technology (2024/25, Winter Semester)
