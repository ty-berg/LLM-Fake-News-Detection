import openai
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Set your OpenAI API key
openai.api_key = "YOURKEYHERE"

# Path to your dataset
input_file = 'refined_cleaned_data.jsonl'

# Simplified ground truth categories
true_categories = {"TRUE", "True", "True.", "TRUE:", "MOSTLY TRUE", "Mostly True", "TRUE BUT OUTDATED"}
false_categories = {"FALSE", "False", "False.", "FALSE:", "MOSTLY FALSE", "Mostly False"}

# Function to query GPT-4 with or without explanation
def query_gpt4(statement, explanation=False):
    if explanation:
        prompt = f'Explain why the following statement is "False": "{statement}"'
    else:
        prompt = f'Answer only with "True" or "False". "{statement}"'

    try:
        # Call the API
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000 if explanation else 1,  # Adjust token limit for explanations
        )

        # Access the response content
        response_text = response.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        print(f"Error querying GPT-4: {e}")
        return None

# Process the dataset and compare results
def process_claims(limit=10):
    results = {"correct": 0, "incorrect": 0, "errors": 0, "false_positives": []}
    y_true = []  # Ground truth labels (binary)
    y_pred = []  # Model predictions (binary)
    count = 0  # Counter to limit the number of lines processed

    with open(input_file, 'r') as file:  # Open the dataset file
        for line in file:
            """
            if count >= limit:  # Stop after processing the specified number of lines
                break
            """
            try:
                record = json.loads(line)
                claim = record["claim_cont"]
                ground_truth = record["rating_str"]

                # Simplify ground truth
                expected = 1 if ground_truth in true_categories else 0  # 1 for True, 0 for False

                # Query GPT-4
                model_output = query_gpt4(claim)

                if model_output is None:
                    results["errors"] += 1
                    continue

                # Convert model output to binary
                predicted = 1 if model_output.lower() == "true" else 0

                # Append to metrics lists
                y_true.append(expected)
                y_pred.append(predicted)

                # Compare results
                if predicted == expected:
                    results["correct"] += 1
                else:
                    results["incorrect"] += 1

                # Collect false negatives for explanation
                if predicted == 0 and expected == 1 and len(results["false_positives"]) < 10:
                    explanation = query_gpt4(claim, explanation=True)
                    results["false_positives"].append({"claim": claim, "explanation": explanation})

                count += 1  # Increment processed lines count
                if count % 100 == 0:
                    print(f"Processed {count} lines...")

            except json.JSONDecodeError:
                results["errors"] += 1

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else None  # Avoid AUC for single-class cases

    # Plot AUC-ROC curve if applicable
    if auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # Dashed diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
        "total_lines": count,
        "errors": results["errors"],
        "false_negatives_with_explanations": results["false_positives"],
    }

    return metrics

# Run the evaluation for a limited number of lines
if __name__ == "__main__":
    metrics = process_claims(limit=100)  # Process only 100 lines for testing
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        if key == "false_negatives_with_explanations":
            print(f"{key}:")
            for item in value:
                print(f"Claim: {item['claim']}")
                print(f"Explanation: {item['explanation']}")
        else:
            print(f"{key}: {value}")

