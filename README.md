# M6 comptition files

Steps:
1. Run download.py to create a data dataframe -> data will have history and future (start and end dates of future should change)
2. Run feature_genearation.py to create history and future dataframes. An initial feature genration is added to simply shift the values over future horizon. 
3. Run baseline.py to train and save models (CV = 3). It also reports bias and wmape on a test set. 
4. Run create_submission_file


https://mofc.unic.ac.cy/the-m6-competition/

