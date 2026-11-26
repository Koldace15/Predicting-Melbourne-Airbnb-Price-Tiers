Ethan Lee , Khoi Minh , Aaron Abbott 

Research question: Can Melbourne Airbnb listings be accurately classified into
low, medium, and high price tiers, and which listing attributes are most
influential in distinguishing these tiers?


1. FILES SUBMITTED
------------------
code.ipynb   - single Jupyter notebook containing all code used in the report
README.txt   - this file


2. DATA
-------
Inside Airbnb, Melbourne, Victoria, Australia - detailed listings file
(listings.csv, 16 June 2026 snapshot, 25,728 listings).
Source: http://insideairbnb.com/get-the-data/

The notebook looks for the data in this order:
  1. listings.csv in the same folder as the notebook (or uploaded to the
     Colab session), otherwise
  2. Google Drive at /content/drive/MyDrive/COMP20008/listings.csv
     (Colab will ask permission to mount Drive).
If your copy is elsewhere, edit INPUT_PATH in the first code cell.


3. REQUIREMENTS
---------------
Python 3.10+ with: pandas, numpy, scipy, scikit-learn, matplotlib
(the standard Google Colab environment already has all of these).


4. HOW TO RUN
-------------
Open code.ipynb in Google Colab or Jupyter and run all cells top to bottom
(Colab: Runtime > Run all). Cells must run in order: later sections read
airbnb_cleaned.csv, which the preprocessing section writes.
A full run takes roughly 5-7 minutes; most of this is the KNN and Decision
Tree grid searches. A fixed random seed (random_state = 42) is used for the
train/test split, cross-validation folds, Decision Tree and bootstrap, so
results are reproducible.


5. NOTEBOOK STRUCTURE
---------------------
Section 1 - Preprocessing
  - Load data; drop 6,553 listings with no price (19,175 remain)
  - 2b: evidence for the 3 rejected candidate steps (price outliers > $2,000,
    property_type consolidation, dropping listings with no reviews)
  - 3.1 Chosen step 1: parse bathroom counts from bathrooms_text
  - 3.2 Chosen step 2: impute bedrooms/beds/bathrooms with room-type medians
  - 3.3 Chosen step 3: review gaps (has_reviews flag + median rating fill)
  - 4: create tercile price_tier target, one-hot encode room_type
  - 5: export airbnb_cleaned.csv

Section 2 - Correlation Analysis
  - 7-variable set incl. price_tier and distance_to_cbd (haversine)
  - Pearson, Spearman, MI and NMI (min-normalised) for all 21 pairs,
    with N/A where a method is not appropriate
  - Heatmaps of all four measures

Section 3 - Supervised Learning and Evaluation
  - 15 predictors; stratified 80/20 split (15,340 train / 3,835 test)
  - KNN (StandardScaler inside a Pipeline) and Decision Tree tuned with
    5-fold stratified CV, macro-F1 scoring; every setting tried is reported
  - Default vs chosen hyperparameters
  - Test-set evaluation vs majority-class baseline; per-class
    precision/recall/F1, confusion matrices
  - Bootstrap 95% CIs (1,000 resamples) for macro-F1 and the paired
    KNN - Decision Tree difference
  - Decision Tree Gini importances and KNN permutation importances

Section 4 - Feature Selection
  - Filter method: MI between each predictor and price_tier (training data)
  - Embedded method: Decision Tree Gini importance
  - Top-3 comparison and hard-case listing (id 2571693)


6. HOW OUTPUTS MAP TO THE REPORT
--------------------------------
Report item          Notebook location                         Output file
Table 1              Preprocessing 2b, 3.1, 3.2, 3.3 outputs   -
Table 2              Preprocessing 3.1, 3.2, 3.3 outputs       -
Figure 1             Correlation section 4                     correlation_heatmaps.png
Section 4.2 values   Correlation section 3 (results table)     -
Table 3              Supervised Learning 3 and 4               knn_cv_results.csv,
                                                               dt_cv_results.csv
Figure 2             Supervised Learning 4                     dt_tuning.png
Table 4              Supervised Learning 5                     hyperparameter_effects.csv
Table 5              Supervised Learning 6 and 7               model_summary.csv
Section 4.3 errors   Supervised Learning 6                     confusion_matrices.png,
                                                               test_predictions.csv
Section 4.3 features Supervised Learning 8                     dt_feature_importances.csv,
                                                               knn_permutation_importances.csv,
                                                               feature_influence.png
Table 6              Feature Selection 1 and 2                 feature_selection_comparision.csv
Table 7              Feature Selection 3                       test_predictions.csv

Also produced (not shown in the report): knn_tuning.png (KNN validation
curve) and airbnb_cleaned.csv (cleaned dataset, 19,175 rows).

All numbers quoted in the report are printed in the notebook cell outputs.
