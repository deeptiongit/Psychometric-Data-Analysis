#Problematic Internet Use Detection using Unsupervised Learning

A Reproducible Research Project on Child Mind Institute's Psychometric Dataset
📘 Overview

This repository contains a reproducible research project conducted on the Child Mind Institute's Problematic Internet Usage (PIU) dataset, focusing on identifying behavioral patterns using unsupervised learning techniques. The project applies clustering methods such as K-Means, BIRCH, and DBSCAN, evaluated using Silhouette Scores to assess cluster quality.
🧪 Objectives

    Explore psychometric traits associated with problematic internet use.

    Apply unsupervised learning to uncover latent groupings in behavior.

    Evaluate clustering quality using silhouette analysis.

    Ensure the research is lightweight and reproducible on a simple computer.

🗂️ Dataset Description

    Original Data Format:

        .parquet: Raw psychometric dataset.

        train.csv / test.csv: Processed subsets for initial experimentation.

    Source: Child Mind Institute - Problematic Internet Use Competition 
  [Datasets](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

🛠️ Preprocessing

    Numerical and categorical columns were identified and cleaned.

    Missing values handled using appropriate imputers.

    Feature scaling performed using StandardScaler.

    PCA used optionally to reduce dimensionality before clustering.

    All operations kept lightweight to run on standard CPUs.

📈 Modeling Approach
Algorithm	Description	Library Used
K-Means	Partitioning-based clustering using Euclidean distance.	scikit-learn
BIRCH	Hierarchical clustering suitable for large datasets.	scikit-learn
DBSCAN	Density-based clustering robust to noise.	scikit-learn

All models were trained first on the CSV version of the dataset for rapid experimentation and then retrained on the full Parquet dataset for deeper insights.
🧮 Evaluation

Clustering effectiveness was measured using Silhouette Score:

from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)

This score quantifies how well samples are clustered with others that are similar.
🖥️ Requirements

    Python 3.8+

    Libraries:

        pandas

        numpy

        scikit-learn

        matplotlib / seaborn (for visualizations)

        pyarrow or fastparquet (for reading .parquet files)

Install all dependencies with:

pip install -r requirements.txt

🔁 Reproducibility

    The code is modular and can be run in sequence:

        Data Loading

        Preprocessing

        Clustering

        Evaluation

    Compatible with Jupyter Notebooks and standard local machines.

    Results are deterministic across runs with fixed seeds.


📢 Citation / Acknowledgment

This project uses data from the Child Mind Institute, made available through Kaggle.
Please cite the original data source if you use this code or findings in your work.
🚀 Future Work

    Try more advanced clustering techniques (e.g., HDBSCAN, Gaussian Mixture Models).

    Visualize clusters with UMAP or t-SNE.

    Explore correlations with demographic metadata.
