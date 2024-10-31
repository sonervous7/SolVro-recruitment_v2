# Cocktail Clustering Project 🍸

### Overview
This project is designed to analyze, clean, and cluster cocktail data based on various characteristics, such as ingredient types, alcohol content, and glass type. Through data preprocessing, feature engineering, clustering, and visualization, this project aims to uncover patterns in cocktail recipes and group them into clusters for easier exploration and analysis.

### Project Structure
The repository is organized into several modules, each responsible for a distinct part of the data processing and analysis pipeline:

- **data**: Contains the dataset (`cocktail_dataset.json`) and output files generated during execution (`cluster_analysis.xlsx`, `clustered_cocktails.xlsx`, etc.).
- **src**: Contains all the source code, organized into modules:
  - `data_loader.py`: Loads the cocktail dataset.
  - `data_cleaner.py`: Cleans and preprocesses data.
  - `data_augmenter.py`: Fills missing ingredient data and standardizes the format.
  - `feature_extraction.py`: Extracts and engineers relevant features from raw data.
  - `data_scaler.py`: Scales data for clustering.
  - `dimensionality_reducer.py`: Reduces data dimensionality for visualization.
  - `model_evaluator.py`: Evaluates clustering performance using different metrics.
  - `visualizer.py`: Visualizes clusters and various cocktail attributes.
  - `pipeline.py`: Manages the entire data processing and clustering pipeline.
  - `main.py`: Entry point to run the entire project pipeline.

---

### Getting Started

#### Prerequisites
- Python 3.8 or later
- [Git](https://git-scm.com/) for cloning the repository
- Recommended: A virtual environment to isolate project dependencies

#### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cocktail-clustering.git
   cd cocktail-clustering
  ```
2. **Create a virtual environment (optional but recommended)**:
  ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```
3. **Usage**
   ```bash
   python src/main.py
  ```
### Project Workflow

1. **Data Loading**: Loads the cocktail dataset from a JSON file.
2. **Data Cleaning**: Cleans and preprocesses data by removing duplicates, filling missing values, and standardizing formats.
3. **Data Augmentation**: Adds additional information about ingredients to fill in missing values and enrich data.
4. **Feature Engineering**: Extracts relevant features, such as ingredient types and alcohol content, to create structured data for analysis.
5. **Data Scaling**: Scales the data to prepare it for clustering.
6. **Dimensionality Reduction**: Reduces data dimensionality for easier visualization.
7. **Clustering**: Groups cocktails into clusters based on engineered features.
8. **Evaluation and Visualization**: Evaluates clustering quality and visualizes the clusters based on various attributes.

Upon completion, you can find the output files in data folder (e.g., cluster_analysis.xlsx, clustered_cocktails.xlsx, full_cluster_analysis.xlsx, summary_statistics) in the data folder.

---

### Example Outputs

- **`cluster_analysis.xlsx`**: Summary of clusters and their characteristics.
- **`clustered_cocktails.xlsx`**: Lists of cocktails in each cluster.
- **`full_cluster_analysis.xlsx`**: Detailed analysis including statistical summaries for each cluster.
- **`summary_statistics.xlsx`**: Basic descriptive statistics for the dataset.

---

### Contributing

Feel free to fork this repository and make contributions. You can submit a pull request or open an issue if you have any questions or suggestions.

---

### License

This project is licensed under the GNU 3.0 License - see the [LICENSE](LICENSE) file for details.

---

### Contact

For questions, reach out via GitHub or email at [jakub.buszynski@gmail.com].

Happy clustering! 🎉





