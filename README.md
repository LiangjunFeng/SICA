# SICA
----------------------------------
slow independent component analysis, SICA algorithm

Code for a paper
A new independent component analysis for time series feature extraction with the concurrent 
consideration of high-order statistic and slowness

### Description
----------------------------------
SICA is a new algorithm for feature extraction wich conbines the advantages of SFA and FastICA.

SICA could sort the extracted features acoording to the slowest and extracted more useful features
than FastICA.

And here are some reults in the paper

![](https://github.com/LiangjunFeng/SICA/blob/master/results/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202018-04-21%20%E4%B8%8B%E5%8D%887.13.42.png)



where the SICA could figure out the right orignal features but the FastICA can't

### SICA package
-------------------------------------
the SICA has been packed for convient use.

the are two differnent algorithm about SICA in the package:

"sica" for unsurpvised learning cases

"ssica" for surpvised learning cases

``` 
pip install SICA
```

``` 
from SICA import sica
res = sica.SICA.fit_trainsform(traindata)


from SICA import ssica
traindata = ssica.fit_trainsform(traindata,trainlabel)
testdata = ssica.trainsform(testdata)
```




