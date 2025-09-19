import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time
import platform

# Φόρτωση του συνόλου δεδομένων
data = load_breast_cancer()
X, y = data.data, data.target

# Λίστα για αποθήκευση των ευστοχιών
accuracies = []

# Καταγραφή χρόνου έναρξης
start_time = time.time()

# Εκτέλεση 20 επαναλήψεων
for i in range(20):
    # Διαχωρισμός σε train/test sets (70/30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    # Κανονικοποίηση των δεδομένων στο [0,1]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Εκπαίδευση του μοντέλου scikit-learn
    model = LogisticRegression(random_state=i)
    model.fit(X_train_scaled, y_train)
    
    # Υπολογισμός ευστοχίας στο test set
    y_pred_test = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    accuracies.append(test_accuracy)
    
    print(f"\nΕπανάληψη {i+1}/20:")
    print(f"Ευστοχία στο σύνολο ελέγχου: {test_accuracy:.4f}")

# Υπολογισμός χρόνου εκτέλεσης
execution_time = time.time() - start_time

# Υπολογισμός στατιστικών
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print("\nΣυνολικά Αποτελέσματα:")
print(f"Μέσος όρος ευστοχίας: {mean_accuracy:.4f}")
print(f"Τυπική απόκλιση ευστοχίας: {std_accuracy:.4f}")
print(f"\nΧρόνος εκτέλεσης: {execution_time:.2f} δευτερόλεπτα")

# Πληροφορίες συστήματος
print("\nΠληροφορίες Συστήματος:")
print(f"Επεξεργαστής: {platform.processor()}")
print(f"Σύστημα: {platform.system()} {platform.release()}")
