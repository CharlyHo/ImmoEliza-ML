# ImmoEliza – Real Estate Price Prediction (Belgium)

This project aims to predict the selling price of properties in Belgium using machine learning techniques.

## Project Description

This project explores various machine learning models to predict real estate prices in Belgium. The dataset includes features such as property type, location, size, and other relevant attributes.

### 📊 Models Tried

| Model                       | Notes                                                             |
| --------------------------- | ----------------------------------------------------------------- |
| **Linear Regression**       | Simple baseline, interpretable but underperforms                  |
| **Ridge Regression**        | Performs well with regularization, reduced overfitting            |
| **Lasso Regression**        | Too slow to train on full dataset, not used in final selection    |
| **Random Forest Regressor** | Strong performance but prone to **overfitting** on this dataset   |
| **XGBoost Regressor**       | Best generalization, handled non-linear relationships effectively |

### 📈 Evaluation Metrics

Each model was evaluated using the following metrics:

**R² Score**: Measures how well the model explains the variation in the `price`. A value closer to 1 indicates better performance.

**MAE (Mean Absolute Error)**: Represents the average size of the errors in the predictions, regardless of direction.

**RMSE (Root Mean Squared Error)**: Calculates the square root of the average squared differences between predicted and actual values, giving more weight to larger errors.

## Installation

Clone this repository:

```bash

git clone https://github.com/CharlyHo/ImmoEliza-ML.git

```

(Optional) Create a virtual environment:

```bash

python -m venv venv
source venv/bin/activate
```

## Usage

To run the experiments, you can use the following command:

```bash
python main.py
```

---

## Contributors

- [Charly](https://github.com/CharlyHo)
- [Choti](https://github.com/jgchoti)
- [Younes](https://github.com/Reigen-cs)
- [Klebs](https://github.com/lkseier)

<!--

- (Visuals)
- (Contributors)
- (Timeline)
- (Personal situation) -->
