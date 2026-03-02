Here is the codes for ScholarTeamFinder journal paper.

The first step is training retrieval module, which includes query2box with LSTM layer model. And this come from the train_query2box_first.py.

Second, training the ranking module, which includes 3 steps: step1_train_gnn.py, step2_extract_features_offline.py and step3_train_xgboost_ranker.py. 
