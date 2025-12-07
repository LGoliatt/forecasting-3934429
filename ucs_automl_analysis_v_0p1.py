import os
import json
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Define the RegressionMetric class
class RegressionMetric:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_metrics_by_list_names(self, metrics_list):
        results = {}
        if "R2" in metrics_list:
            results["R2"] = r2_score(self.y_true, self.y_pred)
        if "R" in metrics_list:
            results["R"] = r2_score(self.y_true, self.y_pred) ** 0.5
        if "RMSE" in metrics_list:
            results["RMSE"] = mean_squared_error(
                self.y_true, self.y_pred, squared=False
            )
        if "MAE" in metrics_list:
            results["MAE"] = mean_absolute_error(self.y_true, self.y_pred)
        if "MAPE" in metrics_list:
            y_true_nonzero = [y for y in self.y_true if y != 0]
            y_pred_nonzero = [
                self.y_pred[i] for i, y in enumerate(self.y_true) if y != 0
            ]
            if len(y_true_nonzero) > 0:
                results["MAPE"] = (
                    100
                    * (
                        abs(
                            (pd.Series(y_true_nonzero) - pd.Series(y_pred_nonzero))
                            / pd.Series(y_true_nonzero)
                        )
                    )
                ).mean()
            else:
                results["MAPE"] = None
        if "SMAPE" in metrics_list:  ## Added by ROMULO MURUCCI
            # Filtra pares onde a soma dos valores não é zero
            valid_pairs = [
                (y_true, y_pred)
                for y_true, y_pred in zip(self.y_true, self.y_pred)
                if (y_true + y_pred) != 0
            ]

            if len(valid_pairs) > 0:
                y_true_valid = [pair[0] for pair in valid_pairs]
                y_pred_valid = [pair[1] for pair in valid_pairs]

                results["SMAPE"] = (
                    200
                    * (
                        abs(pd.Series(y_true_valid) - pd.Series(y_pred_valid))
                        / (abs(pd.Series(y_true_valid)) + abs(pd.Series(y_pred_valid)))
                    )
                ).mean()
            else:
                results["SMAPE"] = None
        if "A10" in metrics_list:
            results["A10"] = sum(
                abs(
                    (pd.Series(self.y_true) - pd.Series(self.y_pred))
                    / pd.Series(self.y_true)
                )
                <= 0.1
            ) / len(self.y_true)
        return results


# Processing logic
results = []

# Find all folders starting with 'json_automl_d'
base_path = "."  # or your working directory
folders = [
    f
    for f in os.listdir(base_path)
    if os.path.isdir(f) and f.startswith("json_automl_d")
]

for folder_path in folders:
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                y_test = data[0].get("y_test")
                y_pred = data[0].get("y_pred", [])
                model_name = data[0].get("estimator", "unknown")
                dataset_name = data[0].get("dataset", "unknown")

                if y_test and y_pred:
                    metrics = RegressionMetric(y_test, y_pred)
                    m = metrics.get_metrics_by_list_names(
                        ["R2", "R", "RMSE", "MAE", "MAPE", "SMAPE", "A10"]
                    )
                    m["Model"] = model_name
                    m["Dataset"] = dataset_name
                    results.append(m)

# Convert and format
df = pd.DataFrame(results)
summary = df.groupby(["Dataset", "Model"]).agg(["mean", "std"])

formatted_summary = pd.DataFrame()
for metric in summary.columns.levels[0]:
    formatted_summary[metric] = summary[metric].apply(
        lambda x: f"{x['mean']:.3f} ({x['std']:.3f})", axis=1
    )

# Show summary
print(formatted_summary)

df.to_csv("results.csv")
formatted_summary.to_csv("summary_results.csv")
