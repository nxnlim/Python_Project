# Imported following packages to manipulate dataframes, graph relationships 
# between columns and do statistical analysis

%matplotlib inline

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from statistics import mean, median, mode, stdev

with open('BankChurners.csv', newline='') as csvfile:
    bank_data = pd.read_csv('BankChurners.csv')

bank_cleaned = bank_data.replace('Unknown', np.NaN)

# Number of rows with an Attrited Customer / Number of rows with an Existing Customer
(bank_cleaned.loc[bank_cleaned.Attrition_Flag == 'Attrited Customer'].shape[0])/(bank_cleaned.loc[bank_cleaned.Attrition_Flag == 'Existing Customer'].shape[0])


# Fuction to create histograms comparing attrited and existing customers
def hist_v_attrit(column_name):
    # Removes null values from column being used
    bankcol_cleaned = bank_cleaned.dropna(subset=[column_name])
    # Creates histogram with only existing customers
    sns.histplot(bankcol_cleaned.loc[bankcol_cleaned['Attrition_Flag']==
        'Existing Customer', column_name], label = 'Existing', kde = True)
    # Creates histogram with only attrited customers
    sns.histplot(bankcol_cleaned.loc[bankcol_cleaned['Attrition_Flag']==
        'Attrited Customer', column_name], label = 'Attrited',
         kde = True,color='red').legend()

hist_v_attrit('Customer_Age')
plt.title('Histogram of Customer Age')
#plt.savefig('hist_cust_age.png')

hist_v_attrit('Gender')
plt.title('Histogram of Gender')
#plt.savefig('hist_gender.png')

hist_v_attrit('Dependent_count')
plt.title('Histogram of Dependent Count')
#plt.savefig('hist_dep_ct.png')

hist_v_attrit('Education_Level')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'  
    )
plt.title('Histogram of Education Level')
#plt.savefig('hist_ed_lvl.png')

hist_v_attrit('Marital_Status')
plt.title('Histogram of Marital Status')
#plt.savefig('hist_marital.png')

hist_v_attrit('Income_Category')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'  
    )
plt.title('Histogram of Income Category')
#plt.savefig('hist_income.png')

hist_v_attrit('Card_Category')
plt.title('Histogram of Card Category')
#plt.savefig('hist_card_cat.png')

hist_v_attrit('Months_on_book')
plt.title('Histogram of Months on Book')
#plt.savefig('hist_months_on_b.png')

hist_v_attrit('Months_Inactive_12_mon')
plt.title('Histogram of Months Inactive')
#plt.savefig('hist_mon_inactive.png')

hist_v_attrit('Contacts_Count_12_mon')
plt.title('Histogram of Contacts Count')
#plt.savefig('hist_contact_ct.png')

hist_v_attrit('Credit_Limit')
plt.title('Histogram of Credit Limit')
#plt.savefig('hist_c_limit.png')

hist_v_attrit('Total_Revolving_Bal')
plt.title('Histogram of Total Revolving Bal')
#plt.savefig('hist_rev_bal.png')

hist_v_attrit('Avg_Open_To_Buy')
plt.title('Histogram of Avg Open To Buy')
#plt.savefig('hist_open_to_buy.png')

hist_v_attrit('Total_Amt_Chng_Q4_Q1')
plt.title('Histogram of Total Amt Chng Q4 Q1')
#plt.savefig('hist_amt_change_q4q1.png')

hist_v_attrit('Total_Trans_Amt')
plt.title('Histogram of Total Trans Amt')
#plt.savefig('hist_trans_amt.png')

hist_v_attrit('Total_Trans_Ct')
plt.title('Histogram of Total Trans Ct')
#plt.savefig('hist_trans_ct.png')

hist_v_attrit('Total_Ct_Chng_Q4_Q1')
plt.title('Histogram of Total Ct Chng Q4 Q1')
#plt.savefig('hist_ct_change_q4q1.png')

hist_v_attrit('Avg_Utilization_Ratio')
plt.title('Histogram of Avg Utilization Ratio')
#plt.savefig('hist_util_ratio.png')


# Graphs show that the columns that show a different trend when considering are:
# - Total_Revolving_Bal
# - Total_Amt_Chng_Q4_Q1
# - Total_Trans_Amt
# - Total_Trans_Ct
# - Total_Ct_Chng_Q4_Q1


# I can confirm this by taking the count at the value for the attrited customer peak for both groups and getting the ratio.

attrit_data = bank_cleaned.loc[bank_cleaned.Attrition_Flag == 
             'Attrited Customer']
attrit_data[['Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']].aggregate([median, 
              stdev], axis = 0)


sns.lmplot('Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_01.png')

sns.lmplot('Total_Revolving_Bal', 'Total_Trans_Amt', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_02.png')

sns.lmplot('Total_Revolving_Bal', 'Total_Trans_Ct', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_03.png')

sns.lmplot('Total_Revolving_Bal', 'Total_Ct_Chng_Q4_Q1', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_04.png')

sns.lmplot('Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_05.png')

sns.lmplot('Total_Amt_Chng_Q4_Q1', 'Total_Trans_Ct', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_06.png')

sns.lmplot('Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_07.png')

sns.lmplot('Total_Trans_Amt', 'Total_Trans_Ct', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_08.png')

sns.lmplot('Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_09.png')

sns.lmplot('Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', bank_data, fit_reg = False, hue='Attrition_Flag', scatter_kws = {'alpha':0.2})
# plt.savefig('lmplot_10.png')

exist_data = bank_cleaned.loc[bank_cleaned.Attrition_Flag == 'Existing Customer']

def filter_col(df, col_name, coeff):
    '''
    Filters out rows that are with in coeff*standard deviation away from the median in the attrit_data
    df: dataframe to be filtered
    col_name: column in dataframe whose values the dateframe will be filtered by
    coeff: narrows or widens the filter, at 1, it filters by values within 1 standard deviation.
    '''
    std_col = attrit_data[col_name].aggregate(stdev)
    med_col = attrit_data[col_name].aggregate(median)
    return (df.loc[df[col_name] < (med_col+(std_col*coeff))].
            loc[df[col_name] > (med_col-(std_col*coeff))])
    
# Filters through all the columns that attriting depend on.
might_attrit_data = exist_data
filter_cols = ['Total_Revolving_Bal', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']
for col in filter_cols:
    might_attrit_data = filter_col(might_attrit_data, col, .5)
might_attrit_data






