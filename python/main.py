import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

# ============================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================
print("="*70)
print("КЛАСТЕРИЗАЦИЯ (K-MEANS + ИЕРАРХИЧЕСКАЯ) И КЛАССИФИКАЦИЯ СТУДЕНТОВ")
print("="*70)

df = pd.read_csv('Survey_AI.csv')
print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
print(f"\nКолонки в датасете:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. '{col}'")

# Выбор признаков для кластеризации
features_for_clustering = ['Q1.AI_knowledge', 'Q7.Utility_grade', 'Q16.GPA']

# АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ НАЗВАНИЙ КОЛОНОК
print("\n" + "="*70)
print("АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ПРИЗНАКОВ")
print("="*70)

# Словарь с возможными вариантами названий
possible_features = {
    'Q5.Feelings': ['Q5.Feelings', 'Q5.Feelings '],
    'Q8.Advantage_teaching': ['Q8.Advantage_teaching', 'Q8.Advantage_teaching '],
    'Q9.Advantage_learning': ['Q9.Advantage_learning', 'Q9.Advantage_learning '],
    'Q10.Advantage_evaluation': ['Q10.Advantage_evaluation', 'Q10.Advantage_evaluation '],
    'Q11.Disadvantage_educational_process': ['Q11.Disadvantage_educational_process', 'Q11.Disadvantage_educational_process '],
    'Q3#1.AI_dehumanization': ['Q3#1.AI_dehumanization', 'Q3#1.AI_dehumanization '],
    'Q3#2.Job_replacement': ['Q3#2.Job_replacement', 'Q3#2.Job_replacement '],
    'Q3#3.Problem_solving': ['Q3#3.Problem_solving', 'Q3#3.Problem_solving '],
    'Q4#1.AI_costly': ['Q4#1.AI_costly', 'Q4#1.AI_costly '],
    'Q4#2.Economic_crisis': ['Q4#2.Economic_crisis', 'Q4#2.Economic_crisis '],
    'Q4#3.Economic_growth': ['Q4#3.Economic_growth', 'Q4#3.Economic_growth ']
}

# Находим реальные названия колонок
additional_features = []

for key, variants in possible_features.items():
    found = False
    for variant in variants:
        if variant in df.columns:
            additional_features.append(variant)
            print(f"Найден признак: '{variant}'")
            found = True
            break
    if not found:
        print(f"Признак {key} не найден в датасете")
        # Пробуем найти частичное совпадение
        for col in df.columns:
            if key in col:
                additional_features.append(col)
                print(f"  → Найден похожий: '{col}'")
                found = True
                break

print(f"\nВсего признаков для анализа: {len(additional_features)}")

# ============================================================
# 2. ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ
# ============================================================
print("\n" + "="*70)
print("ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАСТЕРИЗАЦИИ")
print("="*70)

# Проверяем наличие признаков для кластеризации
available_cluster_features = []
for feat in features_for_clustering:
    if feat in df.columns:
        available_cluster_features.append(feat)
        print(f"Найден признак для кластеризации: '{feat}'")
    else:
        print(f"Признак {feat} не найден!")

if len(available_cluster_features) < 2:
    raise ValueError("Недостаточно признаков для кластеризации!")

X_cluster = df[available_cluster_features].copy().dropna()
cluster_indices = X_cluster.index
print(f"Данные для кластеризации: {X_cluster.shape}")

# Масштабирование
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

print("\nСтатистика после масштабирования:")
print(f"Среднее: {X_cluster_scaled.mean(axis=0).round(4)}")
print(f"Стд: {X_cluster_scaled.std(axis=0).round(4)}")

# ============================================================
# 3. МЕТОД 1: K-MEANS КЛАСТЕРИЗАЦИЯ
# ============================================================
print("\n" + "="*70)
print("МЕТОД 1: K-MEANS КЛАСТЕРИЗАЦИЯ")
print("="*70)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_cluster_scaled)

df_clustered_kmeans = X_cluster.copy()
df_clustered_kmeans['Cluster'] = kmeans_labels
df_clustered_kmeans['Cluster_Name'] = df_clustered_kmeans['Cluster'].map(
    {i: f'K{i+1}' for i in range(n_clusters)}
)

print("\nРаспределение по кластерам (K-Means):")
print(df_clustered_kmeans['Cluster_Name'].value_counts().sort_index())

# Визуализация K-Means кластеров
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Цвета для кластеров
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# График 1: AI_knowledge vs GPA
ax = axes[0]
for cluster in range(n_clusters):
    cluster_data = df_clustered_kmeans[df_clustered_kmeans['Cluster'] == cluster]
    ax.scatter(cluster_data['Q7.Utility_grade'], 
               cluster_data['Q16.GPA'],
               c=colors[cluster],
               label=f'K{cluster+1}',
               alpha=0.6,
               s=50,
               edgecolors='white',
               linewidth=0.5)

# Добавляем центроиды
centroids = scaler_cluster.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids[:, 0], centroids[:, 2],  # AI_knowledge и GPA
           c='red', marker='X', s=200, edgecolors='black', linewidth=2,
           label='Центроиды')

ax.set_xlabel('Utility_grade', fontsize=12)
ax.set_ylabel('GPA', fontsize=12)
ax.set_title('K-Means: Utility_grade vs GPA', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# График 2: Utility_grade vs GPA
ax = axes[1]
for cluster in range(n_clusters):
    cluster_data = df_clustered_kmeans[df_clustered_kmeans['Cluster'] == cluster]
    ax.scatter(cluster_data['Q1.AI_knowledge'], 
               cluster_data['Q7.Utility_grade'],
               c=colors[cluster],
               label=f'K{cluster+1}',
               alpha=0.6,
               s=50,
               edgecolors='white',
               linewidth=0.5)

# Добавляем центроиды
ax.scatter(centroids[:, 1], centroids[:, 2],  # Utility_grade и GPA
           c='red', marker='X', s=200, edgecolors='black', linewidth=2,
           label='Центроиды')

ax.set_xlabel('AI_knowledge', fontsize=12)
ax.set_ylabel('Utility_grade', fontsize=12)
ax.set_title('K-Means: AI_knowledge vs Utility_grade', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 4. МЕТОД 2: ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ
# ============================================================
print("\n" + "="*70)
print("МЕТОД 2: ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ")
print("="*70)

Z = linkage(X_cluster_scaled, method='ward')
hier_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1

df_clustered_hier = X_cluster.copy()
df_clustered_hier['Cluster'] = hier_labels
df_clustered_hier['Cluster_Name'] = df_clustered_hier['Cluster'].map(
    {i: f'H{i+1}' for i in range(n_clusters)}
)

print("\nРаспределение по кластерам (Иерархическая):")
print(df_clustered_hier['Cluster_Name'].value_counts().sort_index())

# Дендрограмма
plt.figure(figsize=(12, 5))
plt.title('Дендрограмма иерархической кластеризации', fontsize=14)
plt.xlabel('Индекс студента', fontsize=12)
plt.ylabel('Расстояние', fontsize=12)

dendrogram(
    Z,
    truncate_mode='level',
    p=5,
    leaf_rotation=90.,
    leaf_font_size=8.,
    show_contracted=True,
    color_threshold=7
)
plt.axhline(y=7, color='r', linestyle='--', label='Порог отсечения (k=5)')
plt.legend()
plt.tight_layout()
plt.show()

# Визуализация иерархических кластеров
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: AI_knowledge vs GPA
ax = axes[0]
for cluster in range(n_clusters):
    cluster_data = df_clustered_hier[df_clustered_hier['Cluster'] == cluster]
    ax.scatter(cluster_data['Q7.Utility_grade'], 
               cluster_data['Q16.GPA'],
               c=colors[cluster],
               label=f'H{cluster+1}',
               alpha=0.6,
               s=50,
               edgecolors='white',
               linewidth=0.5)

ax.set_xlabel('Utility_grade', fontsize=12)
ax.set_ylabel('GPA', fontsize=12)
ax.set_title('Иерархическая: Utility_grade vs GPA', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# График 2: Utility_grade vs GPA
ax = axes[1]
for cluster in range(n_clusters):
    cluster_data = df_clustered_hier[df_clustered_hier['Cluster'] == cluster]
    ax.scatter(cluster_data['Q1.AI_knowledge'], 
               cluster_data['Q7.Utility_grade'],
               c=colors[cluster],
               label=f'H{cluster+1}',
               alpha=0.6,
               s=50,
               edgecolors='white',
               linewidth=0.5)

ax.set_xlabel('AI_knowledge', fontsize=12)
ax.set_ylabel('Utility_grade', fontsize=12)
ax.set_title('Иерархическая: AI_knowledge vs Utility_grade', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 5. СРАВНЕНИЕ МЕТОДОВ КЛАСТЕРИЗАЦИИ
# ============================================================
print("\n" + "="*70)
print("СРАВНЕНИЕ МЕТОДОВ КЛАСТЕРИЗАЦИИ")
print("="*70)

comparison_df = pd.DataFrame({
    'Кластер': [f'Кластер {i+1}' for i in range(n_clusters)],
    'K-Means': df_clustered_kmeans['Cluster'].value_counts().sort_index().values,
    'Иерархическая': df_clustered_hier['Cluster'].value_counts().sort_index().values
})

print("\n" + comparison_df.to_string(index=False))

# Визуализация сравнения
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means распределение
ax = axes[0]
kmeans_counts = df_clustered_kmeans['Cluster'].value_counts().sort_index()
ax.bar([f'K{i+1}' for i in range(n_clusters)], kmeans_counts.values, color=colors)
ax.set_xlabel('Кластер', fontsize=12)
ax.set_ylabel('Количество студентов', fontsize=12)
ax.set_title('Распределение K-Means кластеров', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Иерархическое распределение
ax = axes[1]
hier_counts = df_clustered_hier['Cluster'].value_counts().sort_index()
ax.bar([f'H{i+1}' for i in range(n_clusters)], hier_counts.values, color=colors)
ax.set_xlabel('Кластер', fontsize=12)
ax.set_ylabel('Количество студентов', fontsize=12)
ax.set_title('Распределение иерархических кластеров', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================
# 6. ВЫБОР МЕТОДА ДЛЯ КЛАССИФИКАЦИИ
# ============================================================
print("\n" + "="*70)
print("ВЫБОР МЕТОДА ДЛЯ КЛАССИФИКАЦИИ")
print("="*70)

chosen_labels = kmeans_labels
print("Для классификации выбрана KMeans кластеризация")

# ============================================================
# 7. ПОДГОТОВКА ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ
# ============================================================
print("\n" + "="*70)
print("ПОДГОТОВКА К КЛАССИФИКАЦИИ")
print("="*70)

df_full = df.loc[cluster_indices].copy()
df_full['Cluster'] = chosen_labels
df_full['Cluster_Name'] = df_clustered_hier['Cluster_Name']

X = df_full[additional_features].copy()
y = df_full['Cluster'].copy()

print(f"Размер матрицы признаков X: {X.shape}")
print(f"Размер целевой переменной y: {y.shape}")
print(f"\nКлассы (кластеры): {sorted(y.unique())}")

if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())
    print("Пропуски заполнены средними значениями")

# Масштабирование
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# ============================================================
# 8. ОБУЧЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ
# ============================================================
print("\n" + "="*70)
print("ОБУЧЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ")
print("="*70)

# ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
decision_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
random_forest = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
logistic_regression = LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42)

models = {
    'Decision Tree': decision_tree,
    'Random Forest': random_forest,
    'Logistic Regression': logistic_regression
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\n{'-'*50}")
    print(f"МОДЕЛЬ: {name}")
    
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    predictions[name] = y_pred_test
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"Точность на обучении: {train_acc:.3f}")
    print(f"Точность на тесте: {test_acc:.3f}")
    print(f"Кросс-валидация: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ============================================================
# 9. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================
print("\n" + "="*70)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*70)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
model_names = list(results.keys())
train_scores = [results[m]['train_accuracy'] for m in model_names]
test_scores = [results[m]['test_accuracy'] for m in model_names]
cv_scores = [results[m]['cv_mean'] for m in model_names]
cv_errors = [results[m]['cv_std'] for m in model_names]

x = np.arange(len(model_names))
width = 0.25

plt.bar(x - width, train_scores, width, label='Train', color='#1f77b4', alpha=0.8)
plt.bar(x, test_scores, width, label='Test', color='#ff7f0e', alpha=0.8)
plt.bar(x + width, cv_scores, width, label='CV Mean', color='#2ca02c', alpha=0.8)
plt.errorbar(x + width, cv_scores, yerr=cv_errors, fmt='none', 
             ecolor='black', capsize=5)

plt.xlabel('Модель', fontsize=12)
plt.ylabel('Точность', fontsize=12)
plt.title('Сравнение точности моделей', fontsize=14)
plt.xticks(x, model_names, rotation=15)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
for i, (name, model) in enumerate(models.items()):
    y_pred = predictions[name]
    cm = confusion_matrix(y_test, y_pred)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.subplot(1, 3, i+1)
    
    class_names = [f'C{i+1}' for i in range(n_clusters)]
    
    sns.heatmap(cm_norm, 
                annot=True,           
                fmt='.2f',             
                cmap='Blues',          
                xticklabels=class_names,  
                yticklabels=class_names,  
                square=True,            
                cbar=False,            
                annot_kws={'size': 10}) 
    
    plt.title(f'{name}', fontsize=12)
    plt.xlabel('Предсказанный', fontsize=9)
    plt.ylabel('Истинный', fontsize=9)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
plt.show()

# ============================================================
# 10. ДЕРЕВО РЕШЕНИЙ И ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================
print("\n" + "="*70)
print("ДЕРЕВО РЕШЕНИЙ И ВАЖНОСТЬ ПРИЗНАКОВ")
print("="*70)

plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=additional_features,
          class_names=[f'C{i+1}' for i in range(n_clusters)],
          filled=True, rounded=True, fontsize=10, max_depth=4)
plt.title('Дерево решений для классификации кластеров', fontsize=16)
plt.tight_layout()
plt.show()

feature_importance = pd.DataFrame({
    'feature': additional_features,
    'importance': random_forest.feature_importances_
}).sort_values('importance', ascending=False)

print("\nТоп-10 наиболее важных признаков:")
print(feature_importance.head(10).to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], 
         feature_importance.head(10)['importance'], color='#1f77b4')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.title('Топ-10 важных признаков (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================
# 11. ROC-КРИВЫЕ
# ============================================================
print("\n" + "="*70)
print("ROC-КРИВЫЕ")
print("="*70)

y_test_bin = label_binarize(y_test, classes=range(n_clusters))
y_score = logistic_regression.predict_proba(X_test)

plt.figure(figsize=(10, 8))
for i in range(n_clusters):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'Кластер C{i+1} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 12. ИТОГОВАЯ ТАБЛИЦА
# ============================================================
print("\n" + "="*70)
print("ИТОГОВАЯ ТАБЛИЦА")
print("="*70)

results_summary = pd.DataFrame({
    'Модель': model_names,
    'Train Acc': [f"{results[m]['train_accuracy']:.3f}" for m in model_names],
    'Test Acc': [f"{results[m]['test_accuracy']:.3f}" for m in model_names],
    'CV (mean±std)': [f"{results[m]['cv_mean']:.3f}±{results[m]['cv_std']:.3f}" for m in model_names]
})

print("\n" + results_summary.to_string(index=False))

best_model = model_names[np.argmax([results[m]['test_accuracy'] for m in model_names])]
print(f"\nЛучшая модель: {best_model}")

# ============================================================
# 13. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================
df_results = df_full.copy()
for name, model in models.items():
    df_results[f'Predicted_{name.replace(" ", "_")}'] = model.predict(X_scaled)

df_clustered_kmeans.to_csv('kmeans_clusters.csv', index=False)
df_clustered_hier.to_csv('hierarchical_clusters.csv', index=False)
df_results.to_csv('student_classification_results.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)

print("\nРезультаты сохранены:")
print("  - kmeans_clusters.csv")
print("  - hierarchical_clusters.csv")
print("  - student_classification_results.csv")
print("  - feature_importance.csv")
print("\n" + "="*70)
print("ГОТОВО!")
print("="*70)