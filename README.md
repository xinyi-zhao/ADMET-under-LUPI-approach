# ADMET-under-LUPI-approach

In this project, we use Learning Under Privileged Information method to help improve the perfor- mance of molecules’ toxicity prediction. Firstly we use molecules’ gene expressions(L1000) as privileged information to improve a one-layer model based on ECFP. Then we introduce TOX21 data and imple- ment graph convolution models GCN and GAT. Finally we apply the LUPI method to the two graph convolution models and examine the improvement of their performance and convergence rate. The re- sults show that LUPI improves the performance of all these three models, and the improvement is more obvious when models become more complex. LUPI also decreases the number of epochs needed to con- verge, and reduces the chance of overfitting in the two graph-based models. Our results demonstrate that L1000 data has a close relationship to TOX21 data, and LUPI plays an important role in assisting molecules’ toxicity prediction, and it helps even when only a small portion of privileged data is available.


Keywords: Learning Under Privileged Information; Toxicity Prediction; Graph Convolution; Machine Learning
