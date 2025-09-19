import numpy as np
from generate_dataset import generate_binary_problem
from logistic_regression import LogisticRegressionEP34
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Ορισμός των κέντρων
C = np.array([[0, 12],
              [0, 12]])

# Δημιουργία του dataset με N=1000 σημεία
X, y = generate_binary_problem(centers=C, N=1000)

# Διαχωρισμός σε train/test sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Εκπαίδευση του μοντέλου
model = LogisticRegressionEP34(lr=0.01)
model.fit(X_train, y_train, 
          iterations=10000, 
          batch_size=None, 
          show_step=1000, 
          show_line=True)

# Υπολογισμός προβλέψεων
y_pred_train = (model.predict(X_train) >= 0.5).astype(int)
y_pred_test = (model.predict(X_test) >= 0.5).astype(int)

# Υπολογισμός και εκτύπωση ευστοχίας
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("\nΑποτελέσματα Μοντέλου:")
print(f"Ευστοχία στο σύνολο εκπαίδευσης: {train_accuracy:.4f}")
print(f"Ευστοχία στο σύνολο ελέγχου: {test_accuracy:.4f}")

# Υπολογισμός και εκτύπωση τελικού loss
final_train_loss = model.loss(X_train, y_train)
final_test_loss = model.loss(X_test, y_test)
print(f"\nΤελικό loss εκπαίδευσης: {final_train_loss:.4f}")
print(f"Τελικό loss ελέγχου: {final_test_loss:.4f}")
