#################################
# Classification Model Evaluation
#################################

import pandas as pd

#################################
# Task 1
#################################
# A model has been created that predicts whether a customer is a churn or not. The actual values of 10 test data observations and the probability values predicted by the model are given.
# 1. Create a confusion matrix with a threshold value of 0.5.
# 2. Calculate Accuracy, Recall, Precision, F1 Score

data = [
    (1, 0.7),
    (1, 0.8),
    (1, 0.65),
    (1, 0.9),
    (1, 0.45),
    (1, 0.5),
    (0, 0.55),
    (0, 0.35),
    (0, 0.4),
    (0, 0.25)
]

df = pd.DataFrame(data, columns=["real_value", "predicted_value"])


# 1. Create a confusion matrix with a threshold value of 0.5.

# Set threshold
threshold = 0.5

# Convert probabilities to binary predictions using the threshold
df["predicted_label"] = (df["predicted_value"] >= threshold).astype(int)

print(df.head(10))

# Initialize counters
TP = TN = FP = FN = 0

# Loop through each row to classify and count
for i in range(len(df)):
    actual = df.loc[i, "real_value"]
    predicted_prob = df.loc[i, "predicted_value"]
    predicted_label = 1 if predicted_prob >= threshold else 0

    if actual == 1 and predicted_label == 1:
        TP += 1
    elif actual == 0 and predicted_label == 0:
        TN += 1
    elif actual == 0 and predicted_label == 1:
        FP += 1
    elif actual == 1 and predicted_label == 0:
        FN += 1

# Create a styled confusion matrix
confusion_matrix = pd.DataFrame({
    "Churn (1)": [TP, FP],
    "Non-Churn (0)": [FN, TN]
}, index=["Churn (1)", "Non-Churn (0)"])

# Add multi-index for better labeling
confusion_matrix.columns.name = "Model Prediction"
confusion_matrix.index.name = "Real Value"

# Add row totals
confusion_matrix[""] = confusion_matrix.sum(axis=1)

# Add column totals (but leave bottom-right cell empty)
totals = confusion_matrix.sum(axis=0)
totals[""] = ""  # remove bottom-right total

# Append total row
confusion_matrix.loc[""] = totals

# Result
print(confusion_matrix)

# 2. Calculate Accuracy, Recall, Precision, F1 Score

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Recall
recall = TP / (TP + FN)

# Precision
precision = TP / (TP + FP)

# F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)

# Results
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Accuracy: 0.80
# Recall: 0.83
# Precision: 0.83
# F1 Score: 0.83

#################################
# Task 2
#################################
# A classification model was created to detect fraudulent transactions among transactions made through the bank.
# The model, which achieved a 90.5% accuracy rate, was deemed sufficient and the model was put into operation.
# However, after it was put into operation, the model's outputs were not as expected, and the business unit reported that the model was unsuccessful.
# The Confusion matrix of the model's prediction results is given below. Accordingly,

# 1. Calculate Accuracy, Recall, Precision, F1 Score
# 2. Comment on what the data science team may have overlooked.

# Confusion matrix data
conf_matrix = pd.DataFrame({
    "Fraud (1)": [5, 90],
    "Non-Fraud (0)": [5, 900],
    "": [10, 990]
}, index=["Fraud (1)", "Non-Fraud (0)"])

# Add titles
conf_matrix.columns.name = "Model Prediction"
conf_matrix.index.name = "Real Value"

# Add subtotal row
totals = conf_matrix.sum(axis=0)
totals[""] = ""  # remove bottom-right total
conf_matrix.loc[""] = totals

print(conf_matrix)


# 1. Calculation Accuracy, Recall, Precision, F1 Score

# Confusion matrix values
TP = 5 # True Positive: Fraud correctly predicted
FN = 5 # False Negative: Fraud incorrectly predicted Non-Fraud
FP = 90 # False Positive: Non-Fraud incorrectly predicted Fraud
TN = 900 # True Negative: Non-Fraud correctly predicted

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Recall
recall = TP / (TP + FN)

# Precision
precision = TP / (TP + FP)

# F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)

# Results
print(f"Accuracy: {accuracy:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1_score:.4f}")

#Accuracy: 0.905
#Recall: 0.500
#Precision: 0.0526
#F1 Score: 0.0952


# 2. Comment on what the data science team may have overlooked.

# Veri bilimi ekibi, modelin genel doğruluğu (accuracy) yüksek olduğu için modelin iyi performans gösterdiğini varsaymış olabilir.
# Ancak bu, ciddi bir yanılgıdır çünkü veri seti oldukça dengesizdir:
    # Fraud (1) sınıfı çok az sayıda örnek içerirken, Non-Fraud (0) sınıfı baskındır.
# Bu durumda model, çoğunluk sınıfı tahmin ederek yüksek doğruluk elde edebilir ama asıl önemli olan azınlık sınıfı olan fraud vakalarını doğru tahmin etmektir.

# Modelin precision değeri çok düşüktür (yaklaşık %5), bu da modelin fraud dediği örneklerin çoğunun aslında fraud olmadığını gösterir.
# Bu durum, sahte alarmların çok fazla olması anlamına gelir ve operasyonel maliyetleri artırabilir.
# Ayrıca recall değeri de sadece %50, yani model gerçek fraud vakalarının yarısını kaçırıyor.

# Bu nedenle, model değerlendirmesinde sadece accuracy değil, precision, recall ve F1 skor gibi metriklerin de dikkate alınması gerekirdi.

###########################

# The data science team may have assumed the model performs well due to its high accuracy.
# However, this is misleading because the dataset is highly imbalanced:
    # the Fraud (1) class has very few instances compared to the dominant Non-Fraud (0) class.
# In such cases, a model can achieve high accuracy simply by predicting the majority class, but the real challenge is correctly identifying the minority class — fraud cases.

# The model’s precision is very low (around 5%), meaning that most of the transactions flagged as fraud are actually not fraud.
# This leads to a high number of false alarms, which can increase operational costs.
# Additionally, the recall is only 50%, so the model is missing half of the actual fraud cases.

# Therefore, evaluation should not rely only on accuracy but also include metrics like precision, recall, and F1 score.