# Advanced-payment-prediction-model-python

**Generating Data**

The dataset includes financial and customer-related features such as:

✅ Customer Age – The age of the customer.

✅ Income – The annual income of the customer.

✅ Property Value – The value of the property being purchased.

✅ Loan Amount & Loan Term – The loan amount and its repayment duration (e.g., 10, 15, 20, 25, 30 years).

✅ Interest Rate – The interest rate applied to the loan.

✅ Previous Advance Payments – The number of past advance payments made.

✅ Credit Score – The customer’s creditworthiness score.

✅ Employment Status – Whether the customer is employed, self-employed, or unemployed.

✅ Advance Payment (Target Variable) – Whether the customer makes an advance payment (Yes/No).

The target variable (Advance Payment) is binary – 1 (Yes) or 0 (No), where 30% of customers in our dataset make advance payments.

**Data Preprocessing**

Splitting Features & Target Variable:

The independent variables (features) (like income, loan amount, etc.) are separated from the target variable (Advance Payment).

Handling Categorical Data:

"Employment Status" (Employed, Self-Employed, Unemployed) is converted into numerical values using one-hot encoding.

Train-Test Split:

The dataset is split into training (80%) and testing (20%) sets to train and evaluate the model.

Feature Scaling:

The numerical features are standardized to improve model performance. This ensures that all features are on the same scale, preventing bias in the predictions.

**Model Training & Prediction**

A Random Forest Classifier is used to train the model.

The model learns patterns from the customer data and predicts whether a new customer will make an advance payment.

Predictions are made on the test set.

**Model Evaluation**

The model’s performance is assessed using key metrics:

✅ Accuracy Score – Measures the percentage of correct predictions.

✅ Classification Report – Shows precision, recall, and F1-score for each class.

✅ Confusion Matrix – Displays the number of correct vs. incorrect predictions.

✅ ROC-AUC Score – Measures how well the model differentiates between payers and non-payers.

**Business Insights & Recommendations**

📌 Key Findings from the Model:

Customers with higher incomes, better credit scores, and past advance payments are more likely to make advance payments.

Self-employed customers are slightly less likely to make advance payments compared to employed ones.

Higher interest rates discourage advance payments.

📌 Business Strategies to Increase Advance Payments:

✅ Offer Discounts on Interest Rates – Encourage early payments by reducing interest rates for likely payers.

✅ Custom Loan Plans – Provide flexible early repayment options for high-credit-score customers.

✅ Loyalty Rewards for Repeat Payers – Incentivize customers with a history of advance payments.

✅ Targeted Marketing Campaigns – Focus on high-income and high-credit-score customers.









