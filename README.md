# Disease Detection System

A comprehensive machine learning system for detecting various diseases including Diabetes, Osteoporosis, Parkinson's Disease, and Sleep Disorders. This project implements multiple machine learning algorithms to provide accurate disease predictions based on various health parameters.

## Features

- **Diabetes Detection**: Uses Random Forest Classifier with SMOTE for handling imbalanced data
- **Osteoporosis Detection**: Implements multiple models (Decision Tree, Random Forest, Logistic Regression) with hyperparameter tuning
- **Parkinson's Disease Detection**: Utilizes XGBoost classifier with feature importance analysis
- **Sleep Disorder Detection**: Employs Decision Tree and Random Forest with extensive EDA

## Project Structure

```
.
├── diseasedetect/
│   ├── parkinsons/
│   │   ├── parkdata.csv
│   │   └── parkinson.py
│   ├── diabetes/
│   │   ├── diabetes.csv
│   │   └── diabeticode.py
│   ├── sleepdisorder/
│   │   ├── Sleep_health_and_lifestyle_dataset.csv
│   │   └── sleepdis.py
│   └── osteoporosis/
│       ├── osteoporosis.csv
│       └── osteo.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sumit-Verma19/disease-detection-prediction.git
cd disease-detection-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Diabetes Detection
```bash
python diabeticode.py
```
- Input various health parameters like glucose level, blood pressure, etc.
- Get prediction results with risk assessment

### Osteoporosis Detection
```bash
python osteo.py
```
- Provides comprehensive analysis using multiple models
- Includes visualization of model performance metrics

### Parkinson's Disease Detection
```bash
python parkinson.py
```
- Uses voice pattern analysis for detection
- Shows feature importance and model performance metrics

### Sleep Disorder Detection
```bash
python sleepdis.py
```
- Analyzes sleep patterns and lifestyle factors
- Includes extensive exploratory data analysis

## Model Performance

### Diabetes Detection
- Uses Random Forest Classifier with SMOTE
- Handles imbalanced data
- Includes margin percentage calculations
- Includes a prediction system with risk assessment

### Osteoporosis Detection
- Implements multiple models:
  - Decision Tree
  - Random Forest
  - Logistic Regression
- Includes ROC curves and confusion matrices

### Parkinson's Disease Detection
- XGBoost Classifier
- Feature importance analysis
- ROC curve visualization

### Sleep Disorder Detection
- Multiple model comparison:
  - Decision Tree
  - Random Forest
- Extensive EDA visualizations

## Requirements

See `requirements.txt` for detailed package requirements.

## Contributing

Contributions are welcome! If you would like to improve this repository or add new projects, please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to the contributors, open-source libraries, and datasets that made this project possible.
