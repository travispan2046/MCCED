## MCCED

This project is to support the researcg paper: "Anomaly Detection with Memory-Enhanced Composite Neural Networks for Industrial Control Systems"

It is a prototype intrusion detection system including the following functions:data preprocessing, model training, model test (anomaly detection) and data visualization.


### Installation
```
conda install keras=2.2.0
conda install tensorflow-gpu=1.8.0
pip install Bokeh
pip install numpy=1.14.3
pip install scipy=1.1.0
pip install scikit-learn=0.19.1
```
### Data Preprocessing


We recommend that the folders look like :
```
<base_dir/MCCED>
                |</dataset/>
                  |--swat_train
                  |--swat_test
                  |--wadi_train
                  |--wadi_test
                |</seconddata/>
                |</resultdata/>
                |</code/>
                  |--PreprocessingSWAT.py
                  ...

```
We used `SWaT` dataset and `WADI` dataset. `normtogether.txt` is the setting file. Begin preprocessing data :
<br>`SWaT/WADI:`
```
python Preprocessing.py --settings_file normtogether
```

The results of preprocessing will be saved in `<seconddata>` folder.


### Training
<b>If you would like to train the MCCED model by yourself (`train.txt` is the setting file) :</b>
```
python MCCED_model.py --settings_file train
```
The trained model weights will be saved in `<resultdata>` folder.

<b>If you would like to use weights which we have got by training the model:</b>
 <br>you can download weights directly in test stage from the `<resultdata/>` fold where we provide a whole training result.



 ### Anomaly detection
 Firstly, you should change the weights path in `universal_find_best_in_trained_models.py`:
 ```
 model_path="../resultdata/11_17_21_24conditional_results/models/model.h5"
 ```
 The path should be changed based on your own cases.

 <br>Secondly, begin test:
 ```
 python anomaly_detect.py --settings_file train
 ```
 It will search the best model from ten models based on the best F1 score. Meanwhile, it will select the threshold by grid search.

 <br>Lastly, the F1 score, recall, precision of the best model with the right threshold will be printed in the screen.


 










