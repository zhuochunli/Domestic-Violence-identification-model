# Domestic-Violence-identification-models
Use CNN, RNN, LSTM, Bi-LSTM+self-Attetion models to identify potential domestic violence(DV) crisis.  
Get posts about domestic violence on Facebook, label them with 1(DV exists) and 0(DV doesn't exist) into DV_dataset.xlsx  
Use Word2Vec to train them and generate the 50-dimensional embedding model as dv_50.model  
Use Keras to build all four models, train, and evaluate them: deep learning.py  
Plot the training history  

Domestic Violence Crisis Recognition Method based on Bi-LSTM+Attention: https://ieeexplore.ieee.org/abstract/document/10045367  
```
@inproceedings{wang2022domestic,
  title={Domestic Violence Crisis Recognition Method based on Bi-LSTM+ Attention},
  author={Wang, Zhixiao and Yan, Wenyao and Li, Zhuochun and Huang, Min and Fan, Qinyuan and Wang, Xin},
  booktitle={2022 8th Annual International Conference on Network and Information Systems for Computers (ICNISC)},
  pages={569--575},
  year={2022},
  organization={IEEE}
}
```


References:  
[1]	Sudha Subramani. Domestic Violence Crisis Identification From Facebook Posts Based on Deep Learning[J]. IEEE Access, 2018, 9: 54075 – 54085.  
[2]	Sudha Subramani. Deep Learning for Multi-Class Identification From Domestic Violence Online Posts[J]. IEEE Access, 2019, 4: 46210–46224.
