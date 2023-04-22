# crack-detection

This GitHub repository is the base code of the [paper](https://doi.org/10.1177/1748006X221140966) that compares two approaches for crack detection using a [dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/1) of concrete images having cracks. 

- `pipeline_ml.py`: a pipeline for the implementation of Machine Learning models (SVM, MLP, Random Forest, AdaBoost, K-Nearest Neighbors). The features were extracted using MATLAB Image Processing Toolbox, and the results are in the folder `features`.
- `pipeline_dl.py`: a pipeline for the implementation of Deep Learning models, a subset of ML, but that does not require the manual feature extraction step. The images used are in the folder `images`.

Also included is the source code for the generation of the plots presented in the paper in `generate_images.py`.

## Use
The code was tested in Python 3.9. To install the required packages simply run:

```
pip install -r requirements.txt
```