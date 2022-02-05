# machine_learning
## Assignment 31
- Implement the Machine Learning and Matplotlib training section of the <a href='https://www.w3schools.com'>w3schools</a> website.
  - <a href='https://www.w3schools.com/python/matplotlib_intro.asp'>Matplotlib</a>
  - <a href='https://www.w3schools.com/python/python_ml_getting_started.asp'>Machine Learning</a>
---
## Assignment 32
#### 3d Scatter
- A set of random numbers with three properties of length, width and weight for balloons and melons.
- Display the generated data in three dimensions.
>  ![3dscatter](https://user-images.githubusercontent.com/77120507/149618006-bde5b46e-6de0-4fa4-8f67-dea8e771b0ff.png)

#### Fruit Visualization
- python visualization (case) fruit classification (banana and Apple war).
> ![1](https://user-images.githubusercontent.com/77120507/149618040-beb648f6-3dda-402f-8af1-d47c0ffa2e54.png)
![2](https://user-images.githubusercontent.com/77120507/149618043-d4a76456-8c3a-4747-a3d9-304bc4bd28a9.png)
![3](https://user-images.githubusercontent.com/77120507/149618046-60d315f6-d1a3-4b23-8917-b9201a424a51.png)
![4](https://user-images.githubusercontent.com/77120507/149618047-3400b287-051c-44f1-a4b8-d64e0f95b407.png)
![5](https://user-images.githubusercontent.com/77120507/149618048-1e4d83ca-1a2e-4cc7-a21b-9422d83f9eba.png)
![6](https://user-images.githubusercontent.com/77120507/149618049-f3dee545-7b4b-4d69-a8ed-3ab0d13dacc8.png)
![7](https://user-images.githubusercontent.com/77120507/149618051-1f6199c7-6b8e-4705-8720-1adb7230a5fa.png)
![8](https://user-images.githubusercontent.com/77120507/149618052-75e0f4b8-7ed0-43c7-9523-2e8def7fdd09.png)
---
## Assignment 33
#### Know Your Metrics
- Analysis an <a href='https://www.kaggle.com/vijayuv/onlineretail/discussion/130783'>Online Retail data set </a>to find out the problem in April.
> ![1](https://user-images.githubusercontent.com/77120507/150130034-d7e275f6-c1b1-4482-99ed-365a502e6a4f.png)
![2](https://user-images.githubusercontent.com/77120507/150130040-b6253d82-9378-4de8-acef-dc7441cde2bf.png)
![3](https://user-images.githubusercontent.com/77120507/150130041-c9d39386-60f6-495a-85fd-45f8be1ffbe7.png)
---
## Assignment 34
#### Covid
- The eight countries with the highest number of covid cases in the fourth month of 2020.
> ![countries](https://user-images.githubusercontent.com/77120507/150640172-524a5918-50b6-449b-8301-a474365d9911.PNG)

- Draw a graph that shows the mortality rate in relation to the number of cases in Iran on different days.
> ![new cases vs new deaths in iran](https://user-images.githubusercontent.com/77120507/150640175-a67d39b3-4817-4667-ad83-d6ed96fce6a7.png)

#### KNN
- Write KNN(K Nearest Neighbors) algorithm from scratch then compare with sklearn KNeighborsClassifier.
- Working on <a href='https://www.kaggle.com/mustafaali96/weight-height'>weight-height Dataset</a> on kaggle.


#### KNN OCR
- use kNN to build a basic OCR (Optical Character Recognition) application.
- In this case we work on <a href='https://github.com/BenyaminZojaji/Machine_Learning/blob/main/Assignment34/img/mnist.png'>mnist</a>(Modified National Institute of Standards and Technology dataset) numbers.
- result ->  accuracy: 91.76
---
## Assignment 35
#### Nemo
- Train kNN algorithm with Clownfish[^1] image and test it on another Clownfish image.
- written in Python using opencv, matplotlib.

#### Iris EDA[^2]
- Doing kNN algorithm on sckit-learn Iris dataset with different k and plot the accuracy.
> ![accuracy-bar](https://user-images.githubusercontent.com/77120507/152134392-9a379333-3c1c-4396-a2cd-0d52c66e1b20.png)

#### Abalone EDA[^2]
- Doing kNN algorithm on <a href='https://archive.ics.uci.edu/ml/datasets/abalone'>Abalone Dataset</a> and obtain the accuracy of the algorithm.


[^1]: <a href='https://en.wikipedia.org/wiki/Amphiprioninae'>Clownfish in wikipedia</a>
[^2]: EDA stands for Exploratory Data Analysis
---
## Assignment 36
#### Iris
- Drawing confusion matrix for the <a href='https://github.com/BenyaminZojaji/Machine_Learning/tree/main/Assignment35'>iris</a> problem from last assignment.
> ![iris-confmat](https://user-images.githubusercontent.com/77120507/152503716-ec39c3dc-8ce8-4d5d-912f-0272929323ca.png)
![iris-confmat-prettyconfmat](https://user-images.githubusercontent.com/77120507/152504588-b1e684e5-1da9-4b71-8e88-a87ef0ac7b65.png)

#### LLS (Linear Least-Squares)
- Creating continuous random data for students' study hours and their grades.
> ![data](https://user-images.githubusercontent.com/77120507/152536915-f1e7c2fc-8827-44da-8939-bd66c8177483.png)
- Obtaining line slope by LLS methods. (formula and scipy library)
> ![fittedline](https://user-images.githubusercontent.com/77120507/152535641-a1bed5ea-f895-41cb-b0d5-8d4ca587a49f.png)
![scipy-fittedline](https://user-images.githubusercontent.com/77120507/152535658-7702ade7-d41b-4ab0-8333-bd6c7cbcc5ce.png)
- Draw both of them in one figure.
> ![both-fittedlines](https://user-images.githubusercontent.com/77120507/152535669-bd944145-1e2e-4519-b34d-9c4257b78013.png)

#### LLS - 2 independent
- Implementing the LLS method on the <a href='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html'>Boston dataset</a> from the Scikit-learn library.
- This database offers thirteen features per house, I chose 'CRIM' and 'TAX'.
- Lets scatter the data.
> ![data](https://user-images.githubusercontent.com/77120507/152638221-adb5a55e-a7bf-4530-a639-aebe7981bcf1.png)
- Apply the LLS method on data and get the predicted-surface and plot it.
> ![plot-surface](https://user-images.githubusercontent.com/77120507/152638222-133fc3f5-854a-4e69-9e3c-7c5421932df3.png)
- Lets plot data and predicted-surface in one figure. 
> ![both-dataSurface](https://user-images.githubusercontent.com/77120507/152638215-745ba00d-4d07-4f49-a219-cd7ba5910d06.png)
- Lets generate the three different figures from different views to see better.
> ![3-views](https://user-images.githubusercontent.com/77120507/152638023-e3650536-30e8-4acb-9042-b8f1633af79d.png)
