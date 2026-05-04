# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a Random Forest Classifier implemented with scikit-learn. The model is trained to predict whether a person's income exceeds $50K based on census demographic data.

The categorical features are encoded using one-hot encoding, and the target label is binarized. Th emodel is trained using a standard train-test split with a fixed random seed. 

## Intended Use
This model is intended for education purposes to demonstrate building a complete machine learning pipeline, evaluating model performance, and analyzing performance across data slices. 

## Training Data
The model is trained on the UCI Census Income dataset that contains demographic and employment-related attributes. Some of the attributes contained in the dataset include such things like age, education, occupation, marital status, race, and sex. The dataset includes both cateogrical and numerical features. The target  variable is income level. More specifically income levels less than or equal to 50K or greater than 50k. 

## Evaluation Data
The dataset is split into trainingand testing sets using and 80/20 split where the 80% is for training and the 20% is for evaluation. Stratification is used to maintain the distribution of the target variable across both sets.

## Metrics
The model is evaluated using precision, recall, and an F1 score. The model achieved an overall performance score of 0.7353 for precision, 0.6378 for recall, and 0.6831 for an F1 score. The performance was evaluated across slices of categorical features. The results show variability across groups as some slices achieved perfect scores while others show significantly lower recall or precision scores. 

## Ethical Considerations
The dataset does contain some sensitive attributes including race, sex, and marital status. Using these features in a predictive model raises concerns pretaining to potential bias, reinforcement of historical inequalities, and unfair treatment of underrepresented groups. 

## Caveats and Recommendations
There are some caveats that need mentioning. First of which is the fact that since some slices have very small sample sizes, there could be some extreme metric values. Also, the dataset may contain biases that are learned by the model. 

Some recommentations include removing sensitive features, collecting more balanced data, and performing deeper bias analysis before deploying. 