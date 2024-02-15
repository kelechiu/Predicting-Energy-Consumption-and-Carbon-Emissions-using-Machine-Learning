# Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning

# 1. Data Description
The global data on sustainability dataset is a collection of sustainable energy indicators for 176 countries between 2000 and 2020. It forms the foundation of this research paper, and it will be utilised to predict future energy demand and carbon emission levels. The dataset has 3649 rows each representing a unique country’s information. It also has 21 columns/features that are described in detail below: 

<img width="451" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/055418a1-18b8-4426-b274-dd7f393ea0e4">

# 2.	Exploratory Data Analysis

## 2.1.	Top 5 Countries with Highest Co2 Emissions

<img width="342" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/05808d68-d3c6-4fae-a3c4-2df7d5d74c1b"> <img width="324" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/9c562409-4bf1-4883-bc55-4f6edb457a4f">

China has the highest average CO2 emissions while the United States produces the second highest CO2 emissions out of all 176 countries in this dataset.

## 2.2.	Countries with Lowest CO2 Emissions

<img width="360" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/cda7669d-3903-4a25-a5da-4d37f6b52e93"> <img width="348" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/775d1aea-c8d3-466a-929b-de07a5e72e7e">

Tuvalu produces the lowest average CO2 emissions. It is important to note that Tuvalu, Nauru, Kiribati and Vanuatu are all in one continent (Oceania), suggesting that Oceania may produce the lowest average co2 emissions.

## 2.3.	Annual Growth of Average CO2 Emissions

<img width="301" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/d6be6327-9f6a-4391-ba2a-be5f2d80290a">

Average CO2 emissions have gradually risen over the years, and this is the highest it has been since 2000.

## 2.4.	Annual Growth of Average Energy Consumption

<img width="357" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/611a0aa3-0c7f-48f4-80e0-edc27d3fc149">

## 2.5.	World Map of Energy Consumption Levels

<img width="326" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/bfa0a2fb-6ca9-4e39-acee-cd57d3445127"> <img width="344" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/6b454387-ce1d-45d9-ba69-16c7966faeca">

The world map in figure above reveals that North America consumes the most energy while Africa and South America consume the least.

## 2.6.	World Map of CO2 Emissions

<img width="354" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/733e5665-6c89-473b-bab7-7e15868a543c">  <img width="373" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/adc7bcc1-1d87-45ab-9ea9-b23a5be02d9f">

The world map in Figure above reveals that compared to other continents, Asia produces the highest carbon emissions.

## 2.7.	Correlation Matrix

<img width="298" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/3fc569d7-4e5d-467a-81d4-c8ab990aea79"> <img width="354" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/128f7291-0880-44d2-9dc1-22b14588bfff">

There are high correlations between some variables in the dataset, and these will be used as features in the machine learning models. A positive correlation coefficient means that variables move in the same direction, meaning that if one variable increases, the other variable increases as well.  A negative correlation coefficient means that variables move in opposite directions, meaning that if one variable increases, the other variable decreases.

# 3.	Data Pre-processing Methods

Data cleaning is employed in this research paper to ensure the dataset is accurate and complete for further analysis and predictions. Some common data cleaning methods include handling duplicates and missing values, handling inconsistent data and outliers, addressing wrong data types and normalisation.

## 3.1.	Dataset Inspection

All libraries employed in this paper are imported in Jupyter Notebook and the dataset is read as a Pandas DataFrame. Pandas is used in this project because it is a powerful tool for data manipulation and transformation. 

<img width="324" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/551de3be-5dd2-4869-94ec-31827769ff66"> <img width="326" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/bcd0f254-2e80-4c04-b5d4-1a79f589610a"> 
<img width="405" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/9086532e-b488-4ed0-9399-21209cb78826"> <img width="397" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/b207c5e1-93ed-412c-b826-99762ed721b2">

As seen in figure above, the dataset has 3649 rows and 21 columns.The dataset covers data points between 2010 and 2020. The average access to electricity is 78% while the average energy consumption is 25,743 kilowatts/person. The average co2 emissions per country is 159,866 Kilotons.

## 3.2.	Handling Missing Values

Missing values can skew statistical results, therefore handling them ensures the analysis does not produce incorrect conclusions or inaccurate predictions. Figure 14 below reveals that there are missing values in the dataset.

<img width="278" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/769f111a-65ea-491b-a036-9fa80fd51822">

The missing values are imputed using the mean value of their corresponding country (‘Entity’). The dataset is grouped by countries, therefore using this method ensures that country-specific patterns are retained in the analysis. It also enhances the completeness of the dataset, making it suitable for predictive modelling tasks later in this project.

<img width="255" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/fb12b61f-31d9-47ed-b400-6d3be9c81650"> <img width="285" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/6eccce80-0373-49b9-9965-a1063c2a67ae">

As seen in figure above, the missing values have been greatly reduced. Columns with 1 missing values like Latitude and Longitude will not affect the analysis results.

## 3.3.	Handling Duplicate Values

I checked for duplicated rows in the dataset because duplicates can compromise analysis and prediction results. This dataset does not have duplicated rows as seen below.

<img width="308" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/85d72240-c21b-4e53-ba0b-c68b454ca407">

## 3.4.	Clean Dataset

As seen below, the dataset is clean and ready for model fittings.

<img width="451" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/715ce9be-beab-4782-a1c3-cee66cbb834e">

# 4.	Model Selection

The goal of this project is to predict future energy requirements for each country in the dataset and forecast their carbon emission levels. As a result, the choice of predictive models depends on the specific goals of this analysis.

<img width="451" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/9e382073-0f5b-4d1a-ae67-5371b312c16a">

Regression models are used in this analysis because the predicted values (energy consumption and carbon emissions) are numerical continuous variables, and such models are designed to handle continuous variables. Additionally, regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE) and R-Squared are well-suited to evaluate the performance of regression models because they are easy to interpret, and they provide insights into the relationships between a target variable and its features. Therefore, these four regression models are employed in this study to accurately predict energy consumption and carbon emission levels for three countries over the next 5 years: Linear Regression, Random Forest, XGBoost, KNN Regressor.

## 4.1.	Target Variable Selection

<img width="377" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/d51f3559-933a-45e3-b975-25fe793cd3c3">

The target variables used in this analysis are given below:

#### Primary energy consumption per capita (kWh/person)
This is a numerical variable that represents the amount of energy consumed per person in kilo watts. The predictive models employed in this paper will forecast future values of this variable over the next 5 years and across three different countries.

#### Value_co2_emissions_kt_by_country
This is another numerical variable that represents the amount of carbon emissions produced by each country in kilo tons. The goal of this paper is to predict future co2 emission levels over the next 5 years for three different countries.

## 4.2.	Feature Selection

Features are selected separately for each target variable below. Energy consumption level is selected as one of the features for carbon emission levels and vice versa. 

<img width="369" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/2ccebad0-8ebb-4abc-893d-109d745a053f">

## 4.3.	Data Split

The train_test_split function splits the dataset into two, where 80% of the data is used to train the model and 20% is used to test the model. Due to the small size of this dataset, a larger proportion of it is designated for training to enable the model to learn effectively and make accurate predictions.

<img width="385" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/d600deb6-0137-41a7-b7b5-643327ffdac5">

## 4.4.	Feature Scaling

Certain models like Linear Regression are sensitive to the scale of input variables; therefore, features are standardized to ensure that they have the same scale. Scaling also ensures that the coefficients assigned to features are directly comparable across other models.

<img width="276" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/9abd3fcb-d751-49ef-a9e1-2e27458f71e4">

## 4.5.	Enhanced Feature Selection

SelectKBest assigns statistical scores to features based on their relationship to the target variable and selects the top k features for modelling.

<img width="320" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/7bddfdc8-ee41-4aad-bdec-eaf0c87fdb03">

# 5. Model Training

The predictive models are trained using the selected scaled features and their prediction scores are given below.

## 5.1.	Linear Regression

Linear Regression is one of the simplest prediction models and it is used in this analysis to model the relationship between the target variable and its features. Linear Regression coefficients are also easy to interpret as they measure impact of features on target variables.

#### Target Variable 1: Primary energy consumption per capita (kWh/person)

<img width="348" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/16fc44b2-f5e8-4b89-9f54-7f3671fb055d">

#### Target Variable 2: Value_co2_emissions_kt_by_country

<img width="328" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/b7c14748-8ebe-4da4-a913-5cb5ab54d5f3">

## 5.2.	Random Forest

A Random Forest model combines multiple decision trees to predict outcomes, making it more robust and resistant to overfitting issues. It also provides a measure of feature importance that identifies the features that have the most impact on predicting the target variable. Most importantly, it is flexible in modelling complex patterns because it is capable of capturing non-linear relationships between variables.

#### Target Variable 1: Primary energy consumption per capita (kWh/person)

<img width="421" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/f544dca7-525d-4acd-9e10-3073ce59a0bf">

#### Target Variable 2: Value_co2_emissions_kt_by_country

<img width="439" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/d07ffb5f-3bec-4a98-a15b-e6e4af131950">

## 5.3.	XGBoost 

XGBoost model is chosen in this analysis for its’ high performance and efficiency because it can model complex relationships between variables. It also includes regularization terms that limit overfitting issues, ensuring better prediction fitting. 

#### Target Variable 1: Primary energy consumption per capita (kWh/person)

<img width="354" alt="image" src="https://github.com/kelechiu/Predicting-Energy-Consumption-and-Carbon-Emissions-using-Machine-Learning/assets/100145388/455ca931-1eb3-4bb9-a187-b6c71df64f86">

#### Target Variable 2: Value_co2_emissions_kt_by_country
