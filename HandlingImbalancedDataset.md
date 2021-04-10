# Handling imbalanced dataset in Customer Churn Prediction

Customer churn prediction is to measure why customers are leaving a business. In this notebook we will be looking at customer churn in telecom business. We will build a deep learning model to predict the churn and use precision, recall, f1-score to measure performance of our model. We will then handle imbalance in data using various techniques and imporve f1-score.

### Challenge faced with imbalanced data

One of the main challenges faced by the utility industry today is electricity theft. Electricity theft is the third largest form of theft worldwide. Utility companies are increasing turning towards advanced analystics and machine learning algorithms to identify consumption patterns that indicate theft.

However, one of the biggest stumbling blocks is the humongous data and its distribution. Fraudulent transactions are significantly lower than normal healthy transactions i.e accounting it to around 1-2% of the total number of observations. The ask it to imporve identification of the rare minority class as opposed to achieving higher overall accuracy.

Machine Learning algorithms tend to produce unsatisfactory classifiers when faced with imbalanced datasets. For any imbalanced data set, if the event to be predicted belongs to the minority class and the event rate is less than 5%, it is usually referred to as a rare event.

### Example of imbalanced data

Let's understand this with the help of an example.

Ex: In an utilities fraud detection data set you have the following data:

Total Observations : 1000
Fraudulent Observations: 20
Non Fraudulent Observations: 980
Event Rate : 2%

### Challenges with standard Machine Learning technqiues

The conventional model evaluation methods do not accurately measure model performance when faced with imbalanced datasets.

Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have a number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of missclassification of the minority class as compared to the majortiy class.

Evaluation of a classification algorithm performance is measured by the Confusion Matrix which contains information about the actual and the predicted class.

![image](https://user-images.githubusercontent.com/23405520/114258355-a0d6f500-99e3-11eb-8284-fd028f6f5243.png)



<b> Accuracy of a model = (TP+TN) / (TP+FN+FP+TN) </b>

However, while working in an imbalanced domain accuracy is not an appropriate measure to evaluate model performance. For eg: A classifier which achieves an accuracy of 98% with an event rate of 2% is not accurate, it is classifies all instances as the majority class. And eliminates the 2% minority class observations as noise.

### Examples of imbalanced data

Thus, to sum it up, while trying to resolve specific business challenges with imbalanced data sets, the classifiers produced by standard machine learning algorithms might not give accurate results. Apart from fraudulent transactions, other examples of a common business problem with imbalanced dataset are:

Datasets to identify customer churn where a vast majority of customers will continue using the service. Specifically, Telecommunication companies where Churn Rate is lower than 2 %.
Data sets to identify rare diseases in medical diagnostics etc.
Natural Disaster like Earthquakes

## Approach to handling Imbalanced Data
.

### 1. Data Level approach: Resampling Techniques
Dealing with imbalanced dataset entails strategies such as improving classification algorithms or balancing classes in the training data (data preprocessing) before providing the data as input to the machine learning algorithms. The later technqiue is preferred as it has wider application.

The main objective of balancing classes is to either increasing the frequency of the minority class or decreasing the frequency of the majority class. This is done in order to obtain approximately the same number of instances for both the classes. Let us look at a few resampling technqiues.

### 2. Random Under-Sampling
Random Undersampling aims to balance class distribution by randomly eliminating majority class examples. This is done until the majority and minority class instances are balanced out.

- Total Observations = 100
- Fraudulent Observations = 20
- Non Fraudulent Observations = 980
- Event Rate : 2%

In this case we are taking 10% samples without replacement from Non Fraud instances. And combining them with Fraud instances.

- Non Fraudulent Observations after random under sampling = 10 % of 980 = 98
- Total Observations after combining them with Fraudulent observations = 20 + 98 = 118
- Event Rate for the new dataset after under samplign = 20/118 = 17%

<b> Advantages: </b>

It can help improve run time and storage problems by reducing the number of training data samples when the training data set is huge.

<b> Disadvantage: </b>

- It can discard potentially useful information which could be important for building rule classifiers.
- The sample chosen by random under sampling may be a biased sample. And it will not be an accurate representative of the population. Thereby, resulting in accurate results with the actual test data set.


### 3. Random Over-Sampling

Over-Sampling increases the number of instances in the minority class by randomly replicating them in order to present a higher representation of the minority class in the sample.

- Total Observations = 1000
- Fraudulent Observations = 20
- Non Fraudulent Observations = 980
- Event Rate : 2%

In this case we are replicating 20 fraud observations 20 times.

- Non Fraudulent Observations = 980
- Fraudulent Observations after replicating the minority class observationis = 400
- Total observations in the new data set after oversamplign = 1380
- Event Rate for the new data set after under sampling = 400 / 1380 = 29%

<b> Advantages: </b>

- Unlike under sampling this method leads to no information loss.
- Outperforms under sampling

<b> Disadvantages : </b>

- It increases the likelihood of overfitting since it replicates the minority class events.

### 4. Cluster-Based Over Sampling
In this case, the K-means clustering algorithm is independently applied to minority and majority class instances. This is to identify clusters in the dataset. Subsequently, each cluster is oversampled such that all the clusters of the same class have an equal number of instances and all classes have the same size.

- Total Observations : 100
- Fraudulent Observations : 20
- Non Fraudulent Observations : 980
- Event Rate : 2%


<b> Majority Class Clusters </b>

- Cluster 1: 150 Observations
- Cluster 2: 120 Observations
- Cluster 3: 230 Observations
- Cluster 4: 200 Observations
- Cluster 5: 150 Observations
- Cluster 6: 130 Observations

<b> Minority Class Clusters </b>

- Cluster 1: 8 Observations
- Cluster 2: 12 Observations

After oversampling of each cluster, all clusters of the same class contains the same number of observations

<b> Majority Class Cluster </b>

- Cluster 1: 170 Observations
- CLuster 2: 170 Observations
- Cluster 3: 170 Observations
- Cluster 4: 170 Observations
- Cluster 5: 170 Observations
- Cluster 6: 170 Observations

<b> Minority Class Clusters </b>

- Cluster 1: 250 Observations
- Cluster 2: 250 Observations


<b> Advantages </b>

- This clustering technqiue helps overcome the challenges between class imbalance. Where the number of examples representing positive class differs from the number of examples representing a negative class.

- Also, overcome challenges within class imbalance, where a class is composed of different sub clusters. And each sub cluster does not contain the same number of examples.

<b> Disadvantages </b>

- The main drawback of this algorithm, like most oversampling techniques is the possibility of over-fitting the trainng data.


### 5. (Informed Over Sampling) (SMOTE): Synthetic Minority Over-Sampling Technique for imbalanced data.
This technique is followed to avoid overfitting which occurs when exact replicas of minority instances are added to the main dataset. A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These snthetic instances are then added to the original dataset. The new dataset is used as a sample to train the classification models.

- Total Observations : 1000

- Fraud Observations: 20

- Non-Fraud Observations : 980

- Event Rate = 2%

A sample of instances is taken from the minority class and similar synthetic instances are generated 20 times.

Post generations of synthetic instances, the following dataset is created:

- Minority Class (Fraudulent Observations) = 300
- Majority Class (Non-Fraudulent Observations) = 980
- Event Rate = 300/ 1280 = 23.4 %

<b> Advantages </b>

Mitigates the problem of overfitting caused by random oversampling as synthetic examples are generated rather than replication of instances
No loss of useful information


<b> Disadvantages </b>

- While generating synthetic examples SMOTE does not take into consideration neighboring examples from other classes. This can result in increase in overlapping of classes and can introduce additional noise
- SMOTE is not very effective for high dimensional data

![ICP9 (1)](https://user-images.githubusercontent.com/23405520/114258514-8e10f000-99e4-11eb-8a80-1318cc97c684.png)

![ICP3](https://user-images.githubusercontent.com/23405520/114258519-936e3a80-99e4-11eb-974f-1eb88ecf53b1.png)

### 6. Modified synthetic minority oversampling technique (MSMOTE) for imbalanced data


It is a modified version of SMOTE. SMOTE does not consider the underlying distribution of the minority class and latent noises in the dataset. To improve the performance of SMOTE a modified method MSMOTE is used.

This algorithm classifies the samples of minority classes into 3 distinct groups – Security/Safe samples, Border samples, and latent nose samples. This is done by calculating the distances among samples of the minority class and samples of the training data.

Security samples are those data points which can improve the performance of a classifier. While on the other hand, noise are the data points which can reduce the performance of the classifier. The ones which are difficult to categorize into any of the two are classified as border samples.

While the basic flow of MSOMTE is the same as that of SMOTE (discussed in the previous section). In MSMOTE the strategy of selecting nearest neighbors is different from SMOTE. The algorithm randomly selects a data point from the k nearest neighbors for the security sample, selects the nearest neighbor from the border samples and does nothing for latent noise.

https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/
