Gaussian Mixture Model build in Numpy for Clustering - Base problem category as per Ready Tensor specifications.

- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- clustering

This is a Culstering Model that uses Guassian Mixtures implemented through Numpy.

The algorithm aims to partition n observations into k clusters in which each observation belongs to the cluster with highest probability, assuming each of the k clusters has a multivariable gaussian distribution.

The data preprocessing step includes:

- for numerical variables
  - TruncatedSVD
  - Standard scale data

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as car, iris, penguins, statlog, steel_plate_fault, and wine. Additionally, we also used various synthetically generated datasets such as two concentric (noisy) circles, four worms (four crescent-moon shaped clusters), and unequal variance gaussian blobs.

This Clustering Model is written using Python as its programming language. Numpy is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT.
