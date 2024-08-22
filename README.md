# Credit rating forecast for credit card issuance

You have been hired by Pro National Bank as a data scientist, your first task is to create a model capable of estimating the creditworthiness of customers, in order to help the dedicated team understand whether or not to accept the request for the credit card.

To this end, you are given the anonymized data of customers who have already obtained the credit card and regularly pay the installments.

The data is in a CSV file available at this address: https://proai-datasets.s3.eu-west-3.amazonaws.com/credit_scoring.csv

- ID: customer identification number
- CODE_GENDER: customer gender
- FLAG_OWN_CAR: indicator of car ownership
- FLAG_OWN_REALTY: indicator of home ownership
- CNT_CHILDREN: number of children
- AMT_INCOME_TOTAL: annual income
- NAME_INCOME_TYPE: type of income
- NAME_EDUCATION_TYPE: level of education
- NAME_FAMILY_STATUS: marital status
- NAME_HOUSING_TYPE:
- DAYS_BIRTH: Number of days since birth
- DAYS_EMPLOYED: Number of days since hiring date, if positive indicates the number of days since unemployed
- FLAG_MOBIL: indicator of the presence of a mobile number
- FLAG_WORK_PHONE: indicator of the presence of a work phone number
- FLAG_PHONE: indicator of the presence of a telephone number (landline)
- FLAG_EMAIL: indicator of the presence of an email address
- OCCUPATION_TYPE: type of occupation
- CNT_FAM_MEMBERS: number of family members
- TARGET: a variable that is equal to 1 if the customer has a high credit rating given by constant payment of installments and 0 otherwise.

You must create a model that predicts the given target.
