# Approche-hybride-poltergeist
1. SUMMARY
POLTERGEIST is a behavioral development defect that makes software maintenance difficult and, consequently, negatively impacts software quality. Detection approaches for this defect are either manual searches or traditional heuristic methods based solely on the structural aspects of the code. We have proposed a hybrid model that relies on unsupervised and semi-supervised learning combined with manual validation to detect this defect. This model is an assembly of two essential models: IsolationForest for isolating potential poltergeists and the beta variational autoencoder for automatic validation.
We evaluated our model on the Apache-Ant 1.6.2 project, which contains 871 Java files.
Our approach validated 34 poltergeists, increasing the number of poltergeists by 9. This study opens up research avenues on machine learning methods in the detection of poltergeists.

2. ARCHITECTURE
   Behaviordetector is composed of three modules. The first module isolates potential defects using the Isolation Forest algorithm. The second module involves manual validation, which aims to extract the true poltergeists from the set of proposals made by Isolation Forest. Finally, a beta variational autoencoder is used to evaluate the manual proposals.
   
3. REPRODUCTION
The apache-ant_annotated.csv can be used for supervised learning. Reproduction is possible thanks to the Behaviordetector.ipynb file. The poltergeists, false poltergeists isolated by the Isolation Forest algorithm, and the files that are not poltergeists can be found in the folders File_apache-ant-poltergeist, File_apache-ant-falsepositifs, and File_apache-ant-truenegatifs, respectively.
We describe the process of acquiring the various datasets. The initial dataset is found in the project apache-ant-1.6.2-src.zip from https://archive.apache.org/dist/ant/source/. We extracted the Java files into the dataset File_apache-ant-1.6.2. Using the Isolation Forest algorithm, we obtained the set of files that are not poltergeists, i.e., the true negatives, which we named File_apache-ant-truenegatifs. Manual validation automatically confirmed the false positives and poltergeists contained in the respective datasets File_apache-ant-falsepositifs and File_apache-ant-poltergeist. Each dataset is associated with a CSV file.
