Validation
==========

A `test script <https://github.com/vantage6/v6-glm-py/blob/main/test/test.py>`_ is
available in the `test` directory. It contains `pytest` unit tests and can be run with
the following command:

.. code-block:: bash

    pytest test/test.py

Be sure to install ``pytest`` before running this command. The script will run the
GLM algorithm via the vantage6 ``MockAlgorithmClient``.

.. TODO describe the tests that are done in the test script