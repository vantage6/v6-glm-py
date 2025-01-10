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


Python client example
---------------------

To understand the information below, you should be familiar with the vantage6
framework. If you are not, please read the `documentation <https://docs.vantage6.ai>`_
first, especially the part about the
`Python client <https://docs.vantage6.ai/en/main/user/pyclient.html>`_.

.. TODO Some explanation of the code below

.. code-block:: python

  from vantage6.client import Client

  server = 'http://localhost'
  port = 7601
  api_path = '/api'
  private_key = None
  username = 'devadmin'
  password = 'password'

  # Create connection with the vantage6 server
  client = Client(server, port, api_path)
  client.setup_encryption(private_key)
  client.authenticate(username, password)

  input_ = {
    'method': 'glm',
    'args': [],
    'kwargs': {
        'outcome_variable': 'my_value',
        'predictor_variables': 'my_value',
        'survival_sensor_column': 'my_value',
        'family': 'my_value',
        'tolerance_level': 'my_value',
        'max_iterations': 'my_value',
        'organizations_to_include': 'my_value',
    },
    'output_format': 'json'
  }

  my_task = client.task.create(
      collaboration=1,
      organizations=[2],
      name='GLM task',
      description='Federated Generalized Linear Model (GLM) task',
      image='harbor2.vantage6.ai/algorithms/glm-py',
      input_=input_,
  )

  task_id = my_task.get('id')
  results = client.wait_for_results(task_id)