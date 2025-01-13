Validation
==========

A `test script <https://github.com/vantage6/v6-glm-py/blob/main/test/test.py>`_ is
available in the `test` directory. It contains `pytest` unit tests and can be run with
the following command:

.. code-block:: bash

    pytest test/test.py

Be sure to install ``pytest`` before running this command. The script will run the
GLM algorithm via the vantage6 ``MockAlgorithmClient``.

The following tests are performed:

- Test if the algorithm converges for a Poisson use case.
- Test if the algorithm converges for a Gaussian use case.
- Test if the algorithm converges for a binomial use case.
- Test if the algorithm converges for a relative survival use case.
- Test if the algorithm gives the expected betas after a single iteration (Poisson).
- Test if the algorithm works with a few more complicated formulas (binomial).
- Verify that the appropriate errors are raised:

  - If the user provides incorrect input
  - If the dataset is too small (including check on null values)
  - If the number of parameters is too high compared to the number of observations
  - If the number of organizations included is too low
  - If the user provides non-allowed columns to use
