# DynamicRelevanceForRareEvents

Dynamic Relevance and Pre-processing Strategies for prediction of rare events.
-------------------------------
Dynamic Relevance module in Python is used to predict rare events in a target continous variables when time compoenent is associated.

This module is an experimental research on prediction of rare events in a continous target variable when time component is associated, using Dynamic Relevance and Sliding Window.


-------------------------------
Dynamic Relevance Module Structure
-------------------------------
This module contains 2 sub modules and 6 files

    1: datasets - Folder  contains bike sharing dataset.

    2: PhiRelevance -
    This module allows to create PhiRelevance Function equivalent to Fortran and C implementation in Python.
    C Implementation: https://www.dcc.fc.up.pt/~rpribeiro/uba/
    Fortran Implementation: https://github.com/paobranco/UBL/tree/master/src

    For more details on usage of PhiRelevance refer readMe.txt in PhiRelevance module

    3: DataVisulaization.py
    This Python file is used for visualizing data for bike-sharing dataset.
    It uses Heatmaps, Histograms, Boxplots, Scatter plots to analyse features in dataset


    4: ReleBasedOverSampling.py
    Python implementation of Pre-processing strategy for regression problem that uses relevance values of continous target variable.

    5: SmoteRSampling.py
    Python implementation of Pre-processing strategy for regression problem works similar to Smote for classification problem. This strategy also uses Relevance to create Synthetic cases.

    6: SlidingWindowUtility.py
    Implement the concept of Dynamic Relevance for Data streams using Sliding Window to predict rare events when time component is assocaited with continous target variable.

    7: TestUtilityModule.py
    Test file to check different methods in PhiRelevance module.

    8: Final Results.docx
    Document contain results, tested for different parameters using SlidingWindowUtility.py


-------------------------------
Prerequisites
-------------------------------
Python interpreter - version 3.*
Fortran Compiler

For Windows:
    Visual studio or any Python IDE providing .dll libraries to parse files in PhiRelevance.


modules required:
    warnings, matplotlib, os, numpy, sklearn, smogn, seaborn, tabulate

-------------------------------
Usage
-------------------------------
For Extreme Methods usind Dynamic Relevance:
In Command Prompt run:

python SlidingWindowUtility.py -dataFile ./datasets/bike-sharing/hour.csv -outputLabel temp -dateLabel dteday -method extremes -methodLevel both -coef 0.05 -slidingWindowSize 120 -dynamicUtility 1 -regressionAlgo 3 -sampling 1

To undersatand clearly the user command line arguments: python SlidingWindowUtility.py -h or --help

-------------------------------
For Testing Relevance Function in PhiRelevance Module

In Command Prompt run:

python TestUtilityModule.py

-------------------------------
help
-------------------------------
python SlidingWindowUtility.py -h or --help

-------------------------------
Author details
-------------------------------

Durga Prasad Rangavajjala

University of Ottawa

drang041@uottawa.ca

-----------------------------------

Supervisor: Paulo Branco

University of Ottawa

-------------------------------
