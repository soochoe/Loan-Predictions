# Loan Predictions
### Problem
The objective of this project is to build a machine learning model that can accurately predict if a borrower will pay off their loan on time or not? The financial lending dataset was collected by a peer-to-peer lending company called Lending Club which provides an online  marketplace that matches borrowers with investors in order to make credit more affordable and investing more rewarding.

### Result
- The error metric chosen to test the models is the false positive rate due to the high costs associated with misclassifying a bad loan.

- After running a logistic regression model and a randomforest model, the logistic regression model gave us the best result of a **false positive rate of 9%**, and a true positive rate of 24%. In order for investors to make a profit, the number of potential borrowers and the interest rate must be adequately large to compensate for the losses from loan delinquencies.  

- Interest rate, revolving line utilization rate, annual income were the top three most important features in the classification process. This is similar to the results from the correlation matrix.

### Libraries used:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

#### How to install project?
- Download the CSV file (loans_2007.csv) and Jupyter notebook file (loan-prediction.ipynb) and save them in the same folder

#### How to run project?
- Click run on the Jupyter notebook and you are good to go. Enjoy :)
