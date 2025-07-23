How to use
==========

Input arguments
---------------

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Argument
     - Type
     - Description
   * - ``family``
     - string
     - Function family to use. Possible values are ``gaussian``, ``binomial``,
       ``poisson``, and ``survival``.
   * - ``outcome_variable``
     - string | null
     - The outcome variable to use in the model. If not given, a formula should be
       provided.
   * - ``predictor_variables``
     - list of strings | null
     - Predictor variables to use in the model. If not given, a formula should be
       provided.
   * - ``formula``
     - string | null
     - Wilkinson formula to use in the model. If not given, the outcome and predictor
       variables should be provided. The formula should adhere to the
       `formulaic formula grammar <https://matthewwardrop.github.io/formulaic/latest/guides/grammar/>`_.
   * - ``categorical_predictors``
     - list of strings | null
     - Column names of predictors that should be treated as categorical. If not given,
       numerical columns are treated as continuous. Text-based columns are treated as
       categorical automatically.
   * - ``category_reference_values``
     - dict | null
     - The reference values for the categorical predictors. If not given, the first
       category is used as the reference value. If you have a column 'A' and you want
       to use 'x' as the reference value, you should provide ``{'A': 'x'}``.
   * - ``survival_sensor_column``
     - string | null
     - The column name of the sensor that should be used for the survival analysis. Only
       used/required when the family is ``survival``.
   * - ``tolerance_level``
     - float
     - The tolerance level to use for the convergence of the algorithm.
   * - ``max_iterations``
     - integer
     - The maximum number of iterations to use for the algorithm.
   * - ``organizations_to_include``
     - List of integers
     - Which organizations to include in the computation.
   * - ``link_function``
     - string | null
     - The link function to use for the binomial model. If not given, the default link function
       for the family is used. Possible values are ``log``, ``logit``.


Python client example
---------------------

To understand the information below, you should be familiar with the vantage6
framework. If you are not, please read the `documentation <https://docs.vantage6.ai>`_
first, especially the part about the
`Python client <https://docs.vantage6.ai/en/main/user/pyclient.html>`_.

The code below runs the GLM algorithm with a standard `v6 dev` network running on the
local machine. It authenticates with the server, sends a GLM tasks and fetches the
results. The data that is being used in this example can be found in the
`Poisson test data directory <https://github.com/vantage6/v6-glm-py/tree/main/test/poisson>`_
of this algorithm's repository.

.. code-block:: python

  from vantage6.client import Client

  server = 'http://localhost'
  port = 7601
  api_path = '/api'
  username = 'dev_admin'
  password = 'password'

  # Create connection with the vantage6 server
  client = Client(server, port, api_path)
  client.authenticate(username, password)

  input_ = {
    'method': 'glm',
    'kwargs': {
        "family": "poisson",
        "outcome_variable": "num_awards",
        "predictor_variables": ["prog", "math"],
        "category_reference_values": {"prog": "General"},
    }
  }

  my_task = client.task.create(
      collaboration=1,
      organizations=[2],
      name='GLM task',
      description='Federated Generalized Linear Model (GLM) task',
      image='harbor2.vantage6.ai/algorithms/glm',
      input_=input_,
      databases = [{"label": "glm"}],
  )

  task_id = my_task.get('id')
  results = client.wait_for_results(task_id)