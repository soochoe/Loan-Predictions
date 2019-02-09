## 3. Reading in to Pandas ##

import pandas as pd
loans_2007 = pd.read_csv("loans_2007.csv")
loans_2007.drop_duplicates()
print(loans_2007.iloc[0])
print(len(loans_2007.columns))

## 5. First group of columns ##

loans_2007=loans_2007.drop(columns= ["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"])
                

## 7. Second group of features ##

loans_2007 = loans_2007.drop(columns=["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"])

## 9. Third group of features ##

loans_2007 = loans_2007.drop(columns = ["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"])
print(loans_2007.iloc[0])
print(len(loans_2007.columns))

## 10. Target column ##

print(loans_2007["loan_status"].value_counts)
loans_2007["loan_status"].value_counts


## 12. Binary classification ##


a = ["Fully Paid", "Charged Off"]
loans_2007 = loans_2007[loans_2007["loan_status"].isin(a)]

mapping_dict = {
   "Fully Paid" : 1,
   "Charged Off" : 0
}
loans_2007 = loans_2007.replace(mapping_dict)




## 13. Removing single value columns ##

#Create an empty list, drop_columns to keep track of which columns you want to drop
orgi_columns = loans_2007
drop_columns = {"one_unique":[]}

for column in loans_2007:
    non_null = loans_2007[column].dropna()
    unique_non_null = non_null.unique()
    num_true_unique = len(unique_non_null)
    if num_true_unique == 1:
        drop_columns["one_unique"].append(column)
print(drop_columns)
        
loans_2007 = loans_2007.drop(columns = drop_columns["one_unique"])