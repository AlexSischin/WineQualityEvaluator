# Goal

Train models of white and red wine quality using _scikit learn_ and _Tensorflow_. Compare the following algorithms:

- Linear Regression
- Softmax Regression
- Dense Neural Network

Try transfer learning to improve red wine model based on white wine model.

# Data analysis

## Dataset overview

White whine:

```text
RangeIndex: 4898 entries, 0 to 4897
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         4898 non-null   float64
 1   volatile acidity      4898 non-null   float64
 2   citric acid           4898 non-null   float64
 3   residual sugar        4898 non-null   float64
 4   chlorides             4898 non-null   float64
 5   free sulfur dioxide   4898 non-null   float64
 6   total sulfur dioxide  4898 non-null   float64
 7   density               4898 non-null   float64
 8   pH                    4898 non-null   float64
 9   sulphates             4898 non-null   float64
 10  alcohol               4898 non-null   float64
 11  quality               4898 non-null   int64  
dtypes: float64(11), int64(1)
```

![img_2.png](img_2.png)

Red whine:

```text
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         1599 non-null   float64
 1   volatile acidity      1599 non-null   float64
 2   citric acid           1599 non-null   float64
 3   residual sugar        1599 non-null   float64
 4   chlorides             1599 non-null   float64
 5   free sulfur dioxide   1599 non-null   float64
 6   total sulfur dioxide  1599 non-null   float64
 7   density               1599 non-null   float64
 8   pH                    1599 non-null   float64
 9   sulphates             1599 non-null   float64
 10  alcohol               1599 non-null   float64
 11  quality               1599 non-null   int64  
dtypes: float64(11), int64(1)
```

![img_3.png](img_3.png)

Notes:

- Both tables have the same columns, therefore models for white and red wine are very likely to perform well on the same
  set of input features. Therefore, we can improve red whine model using **transfer learning** from white wine model,
  since it has much more examples. We could also create a more complex model that evaluates quality for both types of
  wine, but it is pointless.
- All the data is numerical which is very convenient because we **don't have to encode categories**, and we are less
  likely to have extra bias because of outlier categories.
- Dataset is not balanced, and some classes are even missing. This will hurt performance of our models very much and
  will make them harder to compare. Unfortunately, we can't collect or engineer more data. Therefore, in order to
  compare models we will consider two metrics: **F1 score** and **precision on classes missing in the training set**. We
  might want to add a more bias to models, in order to make them generalize better on unknown data.

## Correlation matrix

White wine:

![img.png](img.png)

Red wine:

![img_1.png](img_1.png)

Notes:

- Some input variables correlate with each other. They can cause troubles as well as improve performance. We should
  **inspect them closer**. Namely: residual sugar (white wine); free sulfur dioxide (white wine); fixed acidity (red
  wine); free sulfur dioxide (red wine); pH (red wine).
- Some input variables have very weak correlation with quality. They don't contribute to better results and are more
  likely to cause high bias. But we should **investigate them** before throwing away. Namely: citric acid (white wine);
  residual sugar (white wine); free sulfur dioxide (white wine); pH (white wine); sulphates (white wine); residual
  sugar (red wine); free sulfur dioxide (red wine); pH (red wine).





