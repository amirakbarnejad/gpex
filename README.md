# GPEX
## Paper Results
You can find the notebooks in the "PaperResults/" directory.
For instance, the notebook for explaining the classifier on cifar10 is located in "PaperResults/Cifar10/demo_explainANN_classifier.ipynb" and the notebook
for explaining the attention subcomponent is located in "PaperResults/Cifar10/demo_explainANN_attention.ipynb".

To run the notebooks, you firstly  need to run the following python script to that downloads the models and datasets:

```shell
python setup_paperresults.py
```
If you want to download the material for a specific dataset, you can pass in one argument which has to either "cifar10", "mnist", "kather", or "dogswolves".
For example:
```shell
python setup_paperresults.py cifar10
```


 
