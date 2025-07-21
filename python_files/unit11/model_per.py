import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import (
    make_classification,
    load_breast_cancer,
    load_iris,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

# === Confusion Matrix Example ===
print("Confusion Matrix Example:")
cm_example = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0])
tn, fp, fn, tp = cm_example.ravel()
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# === Confusion Matrix Plot ===
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Confusion Matrix - SVC")
plt.show()

# === F1, Accuracy, Precision, Recall ===
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

print("\nF1 Scores:")
print("Macro:", f1_score(y_true, y_pred, average='macro'))
print("Micro:", f1_score(y_true, y_pred, average='micro'))
print("Weighted:", f1_score(y_true, y_pred, average='weighted'))
print("Per Class:", f1_score(y_true, y_pred, average=None))

print("\nAccuracy:", accuracy_score([0, 1, 2, 3], [0, 2, 1, 3]))
print("Precision:", precision_score(y_true, y_pred, average='macro'))
print("Recall:", recall_score(y_true, y_pred, average='macro'))

# === Classification Report ===
y_true_cls = [0, 1, 2, 2, 2]
y_pred_cls = [0, 0, 2, 2, 1]
print("\nClassification Report:\n", classification_report(
    y_true_cls, y_pred_cls, target_names=['class 0', 'class 1', 'class 2']))

# === ROC AUC Binary ===
X_bin, y_bin = load_breast_cancer(return_X_y=True)
clf_bin = LogisticRegression(solver="liblinear", random_state=0).fit(X_bin, y_bin)
auc_binary = roc_auc_score(y_bin, clf_bin.predict_proba(X_bin)[:, 1])
print("\nBinary ROC AUC:", auc_binary)

# === ROC AUC Multiclass ===
X_multi, y_multi = load_iris(return_X_y=True)
clf_multi = LogisticRegression(solver="liblinear").fit(X_multi, y_multi)
auc_multi = roc_auc_score(y_multi, clf_multi.predict_proba(X_multi), multi_class='ovr')
print("Multiclass ROC AUC:", auc_multi)

# === ROC Curve for Multiclass ===
iris = load_iris()
X, y = iris.data, iris.target
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

X = np.c_[X, np.random.RandomState(0).randn(X.shape[0], 200 * X.shape[1])]
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.5, random_state=0)

classifier = OneVsRestClassifier(SVC(kernel="linear", probability=True, random_state=0))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
plt.plot(fpr[2], tpr[2], color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc[2])
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.title("ROC Curve - Class 2")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# === Log Loss ===
print("\nLog Loss:", log_loss(["spam", "ham", "ham", "spam"],
                              [[.1, .9], [.9, .1], [.8, .2], [.35, .65]]))

# === Regression Metrics ===
y_true_reg = [3, -0.5, 2, 7]
y_pred_reg = [2.5, 0.0, 2, 8]
print("\nRegression Metrics:")
print("MSE:", mean_squared_error(y_true_reg, y_pred_reg))
print("MAE:", mean_absolute_error(y_true_reg, y_pred_reg))
print("R² Score:", r2_score(y_true_reg, y_pred_reg))
