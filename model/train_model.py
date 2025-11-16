import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("DIABETES PREDICTION MODEL TRAINING")
print("="*60)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

print("\n1. Dataset Loaded Successfully!")
print("   Shape:", df.shape)

print("\nClass Distribution:")
print(df['Outcome'].value_counts())

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

for col in zero_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("\nData preprocessing completed!")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
df['Outcome'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution')

plt.subplot(2, 3, 2)
plt.hist([df[df['Outcome']==0]['Glucose'], df[df['Outcome']==1]['Glucose']], 
         bins=20, label=['Non-Diabetic', 'Diabetic'], alpha=0.7)
plt.title('Glucose Distribution')
plt.legend()

plt.subplot(2, 3, 3)
correlation = df.corr()
sns.heatmap(correlation[['Outcome']].sort_values(by='Outcome', ascending=False), 
            annot=True, cmap='RdYlGn', center=0)
plt.title('Feature Correlation')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300)
print("\nEDA visualizations saved!")
plt.close()

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model.fit(X_train_scaled, y_train)

print("\nModel trained successfully!")

y_test_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("\nPERFORMANCE METRICS")
print("Accuracy:  {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall:    {:.2f}%".format(recall*100))
print("F1-Score:  {:.2f}%".format(f1*100))

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Non-Diabetic', 'Diabetic'],
            yticklabels=['Non-Diabetic', 'Diabetic'])
plt.title('Confusion Matrix')

plt.subplot(1, 2, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy*100, precision*100, recall*100, f1*100]
plt.bar(metrics, values, alpha=0.7)
plt.title('Performance Metrics')
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300)
print("\nPerformance visualizations saved!")
plt.close()

cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Mean Accuracy: {:.2f}%".format(cv_scores.mean()*100))

with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved as 'diabetes_model.pkl'")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'scaler.pkl'")

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
