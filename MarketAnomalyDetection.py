# === Train-Test Split ===
# Define features (X) and target (y)
X = df.drop(columns=['Y', 'Data'])  # Features
y = df['Y']  # Target variable

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Training and Evaluation ===
# Define models with hyperparameters
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
    'Gradient Boost': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# Initialize an empty list to store results
model_results = []

# Loop through each model, train it, and evaluate performance
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train) 
    y_pred = model.predict(X_test)  
    
    # Evaluate the model using various metrics
    cv_score = np.mean(cross_val_score(model, X, y, cv=5))  # Cross-validation score
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    model_results.append({
        'model': name,
        'cv_score': cv_score,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Convert results into a DataFrame
performance_df = pd.DataFrame(model_results)

# Display the summary of model performance
print("\n=== Model Performance Summary ===")
print(performance_df)

# === Visualization of Model Performance ===
# Define the metrics to visualize
performance_metrics = ['cv_score', 'test_accuracy', 'precision', 'recall', 'f1_score']

# Plot the performance metrics for each model
performance_df.set_index('model', inplace=True)
performance_df[performance_metrics].plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
