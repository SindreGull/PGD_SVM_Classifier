# PGD_SVM_Classifier

This project implements a custom Projected Gradient Descent solver for Support Vector Machines on linearly separable datasets (will son be expanded to non-linear). Built entirely from scratch using NumPy, among other things, the solver uses advanced techniques like:

- Barzilai–Borwein step length adaptation
- Exact line search for step size
- An improved projection algorithm for box constraints and the equality constraint on dual variables


## Technologies & tools used

- Python 
- NumPy  
- Scipy: for basic optimization utilities, but not solvers
- Matplotlib 


## Skills

- Solving the dual SVM problem using constrained optimization
- Implementing the PGD algorithm with numerical stability and convergence checks
- Recovering the primal variables ($w, b$) from the dual solution
- Creating synthetic datasets with precise control over margin, class distribution, and noise


## Files in this repository

-  codebaseop2.py contains the codebase for the implementation of the solver
-  notebook.ipynb contains the main function, ie.l a demonstration notebook for testing and visualizing the solver
-  test_data.py contains code for generating the linearly separable dataset
-  README.md is this file   
