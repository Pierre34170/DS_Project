#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:45:32 2020

@author: pierreperrin
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

#salary parsing


df = df[df['Salary Estimate'] != '-1']
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('k','').replace('$',''))

df['min_salary'] = minus_Kd.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = minus_Kd.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary + df.max_salary)/2

#company name text only

df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)


#state field

df['job_state'] = df['Location'].apply(lambda x: x.split(',')[0])

#age of company

df['age'] = df['Founded'].apply(lambda x: x if x <1 else 2020 - x)

#parsing of job description (python, ...)

#python
df['python_ym'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
print(df.python_ym.value_counts())

#R studio
df['R_ym'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
print(df.R_ym.value_counts())

#spark
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
print(df.spark.value_counts())

#aws
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
print(df.aws.value_counts())

#excel
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
print(df.excel.value_counts())

#df_out = df.drop(['Headquarters'], axis=1)
df_out = df.drop(['Competitors'], axis=1)


df_out.to_csv('salary_data_cleaned.csv', index=False)

pd.read_csv('salary_data_cleaned.csv')

