# Approche-hybride-poltergeist
1. SUMMARY
POLTERGEIST is a behavioral development defect that makes software maintenance difficult and, consequently, negatively impacts software quality. Detection approaches for this defect are either manual searches or traditional heuristic methods based solely on the structural aspects of the code. We have proposed a hybrid model that relies on unsupervised and semi-supervised learning combined with manual validation to detect this defect. This model is an assembly of two essential models: IsolationForest for isolating potential poltergeists and the beta variational autoencoder for automatic validation.
We evaluated our model on the Apache-Ant 1.6.2 project, which contains 871 Java files.
Our approach validated 25 poltergeists, which can serve as supervised learning data. This study opens up research avenues on machine learning methods in the detection of poltergeists.

2. ARCHITECTURE
   Behaviordetector is composed of three modules. The first module isolates potential defects using the Isolation Forest algorithm. The second module involves manual validation, which aims to extract the true poltergeists from the set of proposals made by Isolation Forest. Finally, a beta variational autoencoder is used to evaluate the manual proposals.

3. REPRODUCTION
he file apache-ant_annotated.csv can be used for supervised learning. Reproduction is possible thanks to the Behaviordetector.ipynb file. The poltergeists, false poltergeists isolated by the Isolation Forest algorithm, and the files that are not poltergeists can be found in the folders File_apache-ant-1.6.2_poltergeists, File_apache-ant-1.6.2_falsepositif, and File_apache-ant-1.6.2_datanegatif, respectively. .
