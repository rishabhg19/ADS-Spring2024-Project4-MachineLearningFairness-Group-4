# Project 4: Machine Learning Fairness

### [Project Description](doc/project4_desc.md)

Term: Spring 2024

+ Team #
+ Project title: Machine Learning Fairness Algorithms
+ Team members
	+ Rishabh Ganesh
	+ Qu Fei An
	+ Rhoan Lee
	+ Yerin Cho
+ Project summary: In this project, we use algorithms to improve the fairness of machine learning classifiers on a Compas dataset about recidivism. This dataset is known to be unfair, so machine learning projects involving this dataset benefit from using algorithms that improve fair classification. The first algorithm we look at is classification without disparate mistreatment. Here, we implement the algorithm so that the output of the classifier mitigates disparate impact and mistreatment, making the results more fair. The second algorithm we look at is a fairness aware classifier with a prejudice remover regularizer. Here, we implement the algorithm that is aware of its own fairness by using a loss function during training that is meant to minimize a prejudice index. Our results from implementing the aforementioned algorithms and evaluating their performance on the Compas recidivism dataset can be found in the reports in the `doc` folder of this repository.
	
**Contribution statement**: Qu Fei An worked on algorithm 1 and the closing remarks for the presentation. Rishabh Ganesh read the paper on algorithm 2 and worked on the implementation, evaluation, and analysis for algorithm 2. Rishabh Ganesh also coordinated team meetings and task delegation. Additionally, Rishabh Ganesh wrote the project summary. Rhoan Lee also read the paper on algorithm 2, debugged the implementation, and did evaluation and analysis for algorithm 2.

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
