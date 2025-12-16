# Soil ARG Prediction Tool

This repository contains a Streamlit-based web application for predicting the relative abundance of antibiotic resistance genes (ARGs) in agricultural soils.

The model estimates the fraction of ARGs in soil metagenomes based on concentrations of heavy metals (Mn, Zn, Pb, Cu, Cr, Ni) and polycyclic aromatic hydrocarbons (PAH).

## Features
- Upload Excel datasets with soil contamination and metagenomic data
- Logarithmic regression model for ARG fraction prediction
- Interactive thresholds for soil quality classification
- Automatic classification of soil samples as:
  - Clean
  - Moderately contaminated
  - Contaminated
- Visualization of observed and predicted ARG fractions
- Export of results as Excel files

## Input Data Format
The input Excel file must contain the following columns:

| Column name | Description |
|------------|-------------|
| Sample | Sample identifier |
| Mn, Zn, Pb, Cu, Cr, Ni | Heavy metal concentrations (mg/kg) |
| PAH | Polycyclic aromatic hydrocarbons |
| ARG | Number of contigs containing antibiotic resistance genes |
| total_contigs | Total number of contigs in the metagenome |

## Scientific Background
Antibiotic resistance genes are increasingly recognized as indicators of anthropogenic pressure in agricultural soils. Heavy metals and PAHs may co-select for antimicrobial resistance in soil microbial communities.

This tool provides a statistical and machine learning-based approach for assessing soil quality and ARG burden under varying contamination levels.

## Installation (Local)
```bash
pip install -r requirements.txt
streamlit run soil_app_interactive.py

