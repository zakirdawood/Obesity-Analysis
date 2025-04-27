import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score, silhouette_score
from scipy.cluster.hierarchy import fcluster
from sklearn.neighbors import KNeighborsClassifier

# load data
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# extract target variable
y = data['NObeyesdad']
data = data.drop(columns=['NObeyesdad'])

# create new columns out of binary and categorical columns
data_encoded = pd.get_dummies(data, drop_first=False, dtype=float)

# standardize continous features
scaler = StandardScaler()
X = scaler.fit_transform(data_encoded)

# tranform categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# split data using stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=74)

# create lists to plot accuracies of decision trees at different depths
training_accuracies = []
testing_accuracies = []

for depth in range(1, 26):
    clf = tree.DecisionTreeClassifier(max_depth=depth, random_state=74)
    clf.fit(X_train, y_train)

    training_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))
    testing_accuracies.append(accuracy_score(y_test, clf.predict(X_test)))

# plot accuracy at different depths
plt.plot(list(range(1, 26)),training_accuracies,'rv-',list(range(1, 26)),testing_accuracies,'bo--')
plt.legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Tree Depth')
plt.ylabel('Classifier Accuracy')
plt.title('Decision Tree Accuracies at Different Depths')
plt.show()

# plot the decision tree at depth 6
final_clf = tree.DecisionTreeClassifier(max_depth=6, random_state=74)
final_clf.fit(X_train, y_train)
tree.plot_tree(final_clf, feature_names=data_encoded.columns, class_names=label_encoder.classes_, filled=True, fontsize=5)
plt.title("Decision Tree At Depth of 6")
plt.show()

# print accuracy at depth of 6
print("Training Accuracy at Decision Tree Depth of 6: " + str(round(training_accuracies[5], 4)))
print("Test Accuracy at Decision Tree Depth of 6: " + str(round(testing_accuracies[5], 4)))

# create gaussian naives baye model and fit it
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# find training vs test accuracies using the model
nb_train_accuracy = accuracy_score(y_train, naive_bayes.predict(X_train))
nb_test_accuracy = accuracy_score(y_test, naive_bayes.predict(X_test))
print("Naive Bayes Training Accuracy: " + str(round(nb_train_accuracy, 4)))
print("Naive Bayes Test Accuracy: " + str(round(nb_test_accuracy, 4)))

# display where naive bayes is predicting correctly vs incorrectly using a confusion matrix
cm = confusion_matrix(y_test, naive_bayes.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation='vertical')
plt.title("Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.show()

# 2d pca to help complete and visualize clustering
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)
p1 = pca_data[:, 0]  # first column
p2 = pca_data[:, 1]  # second column

# check what pca components are made of
feature_affects = pd.DataFrame(pca.components_, columns=data_encoded.columns, index=['PCA1', 'PCA2'])
print(feature_affects.T)

# plot pca
plt.scatter(pca_data[:, 0], pca_data[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA')
plt.grid(True)
plt.show()

# complete link agglomerative clustering
sample_labels = [f"Sample {i+1}" for i in range(len(pca_data))]
complete_link = hierarchy.linkage(pca_data, method='complete')

# plot entire dendrogram
hierarchy.dendrogram(complete_link, labels=sample_labels, orientation='right', leaf_font_size=6)
plt.title("Complete Link Hierarchical Clustering")
plt.tight_layout()
plt.show()

scores = []
cluster_range = range(2, 16)

for k in cluster_range:
    clusters = fcluster(complete_link, t=k, criterion='maxclust')
    score = silhouette_score(pca_data, clusters)
    scores.append(score)

plt.plot(cluster_range, scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.grid(True)
plt.show()

# show last 3 clusters on original pca graph
clusters = fcluster(complete_link, t=3, criterion='maxclust')
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Complete Linkage Clusters on PCA Data')
plt.grid(True)
plt.show()

# silhoutte score
silhoutte = silhouette_score(pca_data, clusters)
print("Silhouette Score (Complete-Link with 3 clusters): " + str(silhoutte))
 
# find ideal number of kmeans clusters with elbow method
inertia_arr = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, random_state=74)
    kmeans.fit(pca_data)
    inertia_arr.append(int(kmeans.inertia_))

# plot inertia vs. number of clusters
plt.figure()
plt.plot(range(1, 16), inertia_arr)
plt.title('K-means Inertia vs. Number of Clusters')
plt.xlabel('Clusters')
plt.ylabel('Inertia')
plt.show()

# kmeans with elbow value of k=5
kmeans = KMeans(n_clusters=5, random_state=74)
kmeans_clusters = kmeans.fit_predict(pca_data)

# plot clusters
plt.figure()
scatterplot = plt.scatter(p1, p2, c=kmeans_clusters, s=7)
plt.title('K-Means Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# silhoutte score
silhouette_avg = silhouette_score(pca_data, kmeans_clusters)
print(f"K-Means Silhouette Score: {silhouette_avg:.4f}")

# range of k values to test for knn
k_values = range(1, 11)

# plot different metrics
lst_accuracy_score = []
lst_precision_score = []
lst_recall_score = []
lst_f1_score = []

# cross-validation for multiple metrics
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    accuracy = cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()
    precision = cross_val_score(knn, X, y, cv=10, scoring='precision_macro').mean()
    recall = cross_val_score(knn, X, y, cv=10, scoring='recall_macro').mean()
    f1score = cross_val_score(knn, X, y, cv=10, scoring='f1_macro').mean()
    lst_accuracy_score.append(accuracy)
    lst_precision_score.append(precision)
    lst_recall_score.append(recall)
    lst_f1_score.append(f1score)
 
# plot metrics vs k
plt.figure(figsize=(10,7))
plt.plot(k_values, lst_accuracy_score, label='accuracy')
plt.plot(k_values, lst_precision_score, label='precision')
plt.plot(k_values, lst_recall_score, label='recall')
plt.plot(k_values, lst_f1_score, label='f1')
plt.xlabel('Number of Neighbors')
plt.ylabel('Score')
plt.title('KNN Metrics Score vs. Number of Neighbors')
plt.legend()
plt.grid(True)
plt.show()

# train and predict knn on k value with best metrics (k=1)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# display Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(xticks_rotation='vertical')
plt.title(f'KNN Confusion Matrix')
plt.show()

# metrics for the best k classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1score = f1_score(y_test, y_pred, average='macro')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1score:.4f}')