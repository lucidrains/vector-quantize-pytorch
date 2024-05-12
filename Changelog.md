# Work on LatentQuantize

[]

* What is the difference between codebook size, dim, num codebooks ?? How to compute those, in particular need to check coherence between case [5,5,5] and 5 (should lead to the same...)

[Commit 5]

* Clean the _equal_level part

[Commit 4]

* Add test for 'int' case: fails! Issue with CPU memory... uncovering an error in building the levels in that case: it uses the 'dim' arguments passed in the __init__, instead of the number of codebooks or codebook_dim?
* Fix: same level case, by simply elimintating the '_equal_levels' variable, and ensuring that both cases are treated the same way. In particular, 'values_per_latent' parameter has the same type and shape (that is, a ParameterList)

[Commit 3]

* The dim attribute is not optional: change of the docstring, remove the use of 'default' computation.
* Add a test dedicated to "same level" case of LatentQuantization --> uncover an error in the quantize method.

[Commit 2]

* Format and lint of code
* Remove: helper function 'exists' and 'defaults'. They added a level of complexity without adding lisibility (in particular 'default' and its absence of docstring)

[Commit 1]

* Add a test file to check the refacto.
* Add radon to compute complexities of code, and Rich for better output.
* Add wily to follow the improvement of the metrics through the refacto.
* Add Marimo for better notebooks (maybe not used)





<!-- # Work on README

Using the pytest-examples plugin for pytest, creating a basic test suite for this project is easy, as the tests are already existing in the README.
To improve the tests (and the examples!), here are a few changes:

* In all the file, using print statements to truly validate the code. Instead of comments.
* Line 81, shapes were wrong. -->