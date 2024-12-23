import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
data = pd.read_csv('coorg_tourism_dataset.csv')

# Preprocess the data
label_encoder = LabelEncoder()
all_possible_weather = ['Cool', 'Warm', 'Rainy', 'Pleasant']
label_encoder.fit(all_possible_weather)
data['Weather'] = label_encoder.transform(data['Weather'])

# Define features and target variable
X = data[['Weather', 'Festival_or_Occasion', 'Weekend_or_Long_Holiday']]
y = data['Number_of_Tourists_Visited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train different models
# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)

# Support Vector Machine (SVM) Regression
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
r2_svm = r2_score(y_test, y_pred_svm)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)

# Take user input
date_input = input("Enter the date (DD/MM/YYYY): ")
date_object = datetime.strptime(date_input, "%d/%m/%Y").date()

event_input = input("Is there any event or festival on this day? (yes/no): ")
festival_or_occasion = 1 if event_input.lower() == 'yes' else 0

weekend_input = input("Does this date fall on a weekend or long holiday? (yes/no): ")
weekend_or_long_holiday = 1 if weekend_input.lower() == 'yes' else 0

# Function to determine weather based on the month
def get_weather_karnataka(date):
    month = date.month
    if month in [11, 12, 1, 2]:  # November to February: Winter Season
        return "Cool"
    elif month in [3, 4, 5]:  # March to May: Summer Season
        return "Warm"
    elif month in [6, 7, 8, 9]:  # June to September: Monsoon Season
        return "Rainy"
    else:  # October: Post-Monsoon Season
        return "Pleasant"

# Get weather details for Karnataka based on the date
weather = get_weather_karnataka(date_object)
weather_encoded = label_encoder.transform([weather])[0]

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Weather': [weather_encoded],
    'Festival_or_Occasion': [festival_or_occasion],
    'Weekend_or_Long_Holiday': [weekend_or_long_holiday]
})

# Make predictions using all models
predicted_rf = rf_model.predict(input_data)[0]
predicted_lr = lr_model.predict(input_data)[0]
predicted_svm = svm_model.predict(input_data)[0]
predicted_dt = dt_model.predict(input_data)[0]

# Display predictions
print(f"Predicted Number of Tourists (Random Forest): {int(predicted_rf)}")
print(f"Predicted Number of Tourists (Linear Regression): {int(predicted_lr)}")
print(f"Predicted Number of Tourists (SVM): {int(predicted_svm)}")
print(f"Predicted Number of Tourists (Decision Tree): {int(predicted_dt)}")

# Print R² scores for all models
print(f"\nR-squared (R²) Score - Random Forest: {r2_rf}")
print(f"R-squared (R²) Score - Linear Regression: {r2_lr}")
print(f"R-squared (R²) Score - SVM: {r2_svm}")
print(f"R-squared (R²) Score - Decision Tree: {r2_dt}")

# Visualization: Compare the R² scores of the models
models = ['Random Forest', 'Linear Regression', 'SVM', 'Decision Tree']
r2_scores = [r2_rf, r2_lr, r2_svm, r2_dt]

plt.figure(figsize=(10, 6))
plt.bar(models, r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title("R-squared (R²) Scores of Different Models", fontsize=14)
plt.ylabel("R² Score")
plt.xlabel("Models")
plt.ylim(0, 1)  # R² scores range from 0 to 1
plt.show()
factors = ['Weather', 'Festival/Occasion', 'Weekend/Holiday']
values = [weather_encoded, festival_or_occasion, weekend_or_long_holiday]

