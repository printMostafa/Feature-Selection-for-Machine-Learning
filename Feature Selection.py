import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel, f_classif, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set plot style
plt.rcParams['font.family'] = 'Arial'

# 1. Load data
def load_data():
    """Load breast cancer dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    return X, y, data.feature_names

# 2. Explore data
def explore_data(X, y):
    """Simple data exploration"""
    print("Data shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, square=True)
    plt.title('Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()  # Display the plot
    return corr_matrix
   
# 3. Prepare data
def prepare_data(X, y):
    """Split data and standardize values"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# 4. Feature Selection Methods

# a. Filter Methods
def apply_filter_methods(X_train, y_train, feature_names, k=10):
    """Apply filter methods to select best features"""
    results = {}
    
    # ANOVA F-value
    selector_f = SelectKBest(f_classif, k=k)
    selector_f.fit(X_train, y_train)
    f_support = selector_f.get_support()
    f_features = [feature_names[i] for i in range(len(feature_names)) if f_support[i]]
    
    # Visualize ANOVA F-value scores
    plt.figure(figsize=(12, 6))
    f_scores = selector_f.scores_
    indices = np.argsort(f_scores)[::-1]
    plt.bar(range(k), f_scores[indices][:k], color='skyblue')
    plt.title('Top {} Features by ANOVA F-value'.format(k))
    plt.xlabel('Feature Index')
    plt.ylabel('F-value Score')
    plt.xticks(range(k), [feature_names[i] for i in indices[:k]], rotation=90)
    plt.tight_layout()
    plt.savefig('anova_f_scores.png')
    plt.show()
    
    results['ANOVA F-value'] = f_features
    
    # Chi-squared
    # Convert to positive values as chi2 requires non-negative values
    X_train_pos = X_train - X_train.min(axis=0)
    selector_chi = SelectKBest(chi2, k=k)
    selector_chi.fit(X_train_pos, y_train)
    chi_support = selector_chi.get_support()
    chi_features = [feature_names[i] for i in range(len(feature_names)) if chi_support[i]]
    results['Chi-squared'] = chi_features
    
    # Mutual Information
    selector_mi = SelectKBest(mutual_info_classif, k=k)
    selector_mi.fit(X_train, y_train)
    mi_support = selector_mi.get_support()
    mi_features = [feature_names[i] for i in range(len(feature_names)) if mi_support[i]]
    results['Mutual Information'] = mi_features
    
    return results

# b. Wrapper Methods
def apply_wrapper_methods(X_train, y_train, feature_names, k=10):
    """Apply wrapper methods to select best features"""
    results = {}
    
    # Recursive Feature Elimination (RFE)
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    rfe = RFE(estimator=model, n_features_to_select=k)
    rfe.fit(X_train, y_train)
    rfe_support = rfe.get_support()
    rfe_features = [feature_names[i] for i in range(len(feature_names)) if rfe_support[i]]
    results['RFE'] = rfe_features
    
    return results

# c. Embedded Methods
def apply_embedded_methods(X_train, y_train, feature_names, k=10):
    """Apply embedded methods to select best features"""
    results = {}
    
    # Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    indices = np.argsort(rf_importances)[::-1]
    rf_features = [feature_names[i] for i in indices[:k]]
    
    # Visualize Random Forest feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(k), rf_importances[indices][:k], color='lightgreen')
    plt.title('Top {} Features by Random Forest Importance'.format(k))
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.xticks(range(k), [feature_names[i] for i in indices[:k]], rotation=90)
    plt.tight_layout()
    plt.savefig('rf_importance_scores.png')
    plt.show()
    
    results['Random Forest'] = rf_features
    
    # L1-based feature selection (Lasso)
    lasso_model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    lasso = SelectFromModel(lasso_model)
    lasso.fit(X_train, y_train)
    lasso_support = lasso.get_support()
    lasso_features = [feature_names[i] for i in range(len(feature_names)) if lasso_support[i]]
    results['Lasso'] = lasso_features
    
    return results

# 5. Evaluate Feature Selection Methods
def evaluate_feature_selection(X_train, X_test, y_train, y_test, feature_names, all_selected_features):
    """Evaluate model performance using different feature subsets"""
    results = {}
    
    for method_name, selected_features in all_selected_features.items():
        # Get indices of selected features
        selected_indices = [list(feature_names).index(feature) for feature in selected_features]
        
        # Select subset of data with only selected features
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        # Train classification model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_selected, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save results
        results[method_name] = {
            'accuracy': accuracy,
            'selected_features': selected_features
        }
    
    return results

# 6. Visualize Results
def visualize_results(evaluation_results):
    """Visualize results of different feature selection methods"""
    methods = list(evaluation_results.keys())
    accuracies = [evaluation_results[method]['accuracy'] for method in methods]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color='skyblue')
    plt.xlabel('Feature Selection Method')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Feature Selection Methods Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png')
    plt.show()  # Display the plot
    
    # Plot most frequent features
    feature_counts = {}
    for method in methods:
        for feature in evaluation_results[method]['selected_features']:
            if feature in feature_counts:
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1
    
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    top_features = [x[0] for x in sorted_features[:10]]
    top_counts = [x[1] for x in sorted_features[:10]]
    
    plt.figure(figsize=(12, 6))
    plt.barh(top_features, top_counts, color='lightgreen')
    plt.xlabel('Number of Appearances in Selection Methods')
    plt.ylabel('Features')
    plt.title('Most Important Features Across Different Selection Methods')
    plt.tight_layout()
    plt.savefig('most_important_features.png')
    plt.show()  # Display the plot

# 7. Main Function
def main():
    """Main project function"""
    print("Starting Feature Selection Project...")
    
    # Load data
    X, y, feature_names = load_data()
    print("Data loaded successfully.")
    print("------------------------------------------------------------------------------------------------------")
    # Explore data
    print("\nExploring data...")
    corr_matrix = explore_data(X, y)
    print("Data exploration completed successfully.")
    print("------------------------------------------------------------------------------------------------------")
    # Prepare data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    print("Data preparation completed successfully.")
    print("------------------------------------------------------------------------------------------------------")
    # Apply feature selection methods
    print("\nApplying feature selection methods...")
    k = 10  # Number of features to select
    
    filter_results = apply_filter_methods(X_train, y_train, feature_names, k)
    print("Filter methods applied successfully.")
    print("------------------------------------------------------------------------------------------------------")
    wrapper_results = apply_wrapper_methods(X_train, y_train, feature_names, k)
    print("Wrapper methods applied successfully.")
    print("------------------------------------------------------------------------------------------------------")
    embedded_results = apply_embedded_methods(X_train, y_train, feature_names, k)
    print("Embedded methods applied successfully.")
    print("------------------------------------------------------------------------------------------------------")
    # Collect results from feature selection methods
    all_selected_features = {}
    all_selected_features.update(filter_results)
    all_selected_features.update(wrapper_results)
    all_selected_features.update(embedded_results)
    
    # Evaluate feature selection methods
    print("\nEvaluating feature selection methods...")
    evaluation_results = evaluate_feature_selection(X_train, X_test, y_train, y_test, feature_names, all_selected_features)
    
    # Display evaluation results
    print("\nEvaluation Results:")
    for method, result in evaluation_results.items():
        print(f"{method}: Accuracy = {result['accuracy']:.4f}")
        print(f"Selected features: {', '.join(result['selected_features'])}")
        print()
        print("------------------------------------------------------------------------------------------------------")
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(evaluation_results)
    print("Results visualization completed successfully.")
    print("------------------------------------------------------------------------------------------------------")
    print("\nProject execution completed!")

if __name__ == "__main__":
    main()