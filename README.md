Feng, Jean, and Noah Simon. 2018. “Gradient-Based Regularization Parameter Selection for Problems With Nonsmooth Penalty Functions.” Journal of Computational and Graphical Statistics: A Joint Publication of American Statistical Association, Institute of Mathematical Statistics, Interface Foundation of North America 27 (2): 426–35. https://doi.org/10.1080/10618600.2017.1390470.

External packages needed:
* CVXPY is required for all scripts
* Install Spearmint https://github.com/JasperSnoek/spearmint
* Biopython is required to run the colitis data analysis. Also need to download
     * the geneset data from Molecular Signatures Database
     * GDS1615 from the Gene Expression Omnibus database

Create folders:
* In this directory, create
  * spearmint_descent directory
  * results directory
  * For each example, it will need its own results folder:
      * e.g. results/matrix_completion_groups/tmp
  * To run the Colitis data example, create a realdata directory and put the downloaded
      data into this directory

Section 3 results were generated as follows:
* Elastic Net:
    python elasticnet_eval.py
* Sparse group lasso:
    python sgl_eval.py
* Sparse Additive models:
    python sparse_add_models_eval.py

Section 4 results were generated as follows:
    python realdata_eval.py

By default, the code solves the joint optimization problem via gradient descent.
Specify different solvers (grid search, Spearmint, and nelder-mead) and problem settings via command options.
