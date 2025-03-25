Privacy
=======

.. _privacy-guards:

Guards
------

There are several guards in place to protect the privacy of individuals in the database:

- **Minimum number of data rows to participate**: A node will only participate if it
  contains at least `n` data rows. This is to prevent nodes with very little data from
  participating in the computation. By default, the minimum number of data rows is set
  to 10. Node administrators can change this minimum by adding the following to their
  node configuration file:

  .. code:: yaml

    algorithm_env:
      GLM_MINIMUM_ROWS: 10

  Note that the algorithm also checks for each of the columns involved in generating the
  model that the column contains at least this number of non-null values. If not, the
  computation is also refused.

- **Minimum number of organizations to participate**: The minimum number of
  organizations to participate in a GLM computation is set to 3. This is to prevent
  that a single organization can try to infer the data of only one other organization
  involved in the computation. Node administrators can change this minimum by adding the
  following to their node configuration file. Note however that this check is only
  performed by the node executing the central part of the algorithm.

  .. code:: yaml

    algorithm_env:
      GLM_MINIMUM_ORGANIZATIONS: 3

- **Check parameters vs observations ratio**: If the number of parameters is high
  compared to the number of observations, the computation will not be allowed. This is
  to prevent that data may be inferred from an overfitted model. The maximum ratio of
  parameters vs observations is set to 10%. Node administrators can change this ratio
  using the following configuration:

  .. code:: yaml

    algorithm_env:
      GLM_MAX_PCT_VARS_VS_OBS: 10

- **Setting the allowed columns**: The node administrator can set on which
  columns they want to allow or disallow computation by
  adding the following to the node configuration file:

  .. code:: yaml

    algorithm_env:
      GLM_ALLOWED_COLUMNS: "ageGroup,isOverweight"
      GLM_DISALLOWED_COLUMNS: "age,weight"

  This configuration will ensure that only the columns `ageGroup` and `isOverweight`
  are allowed to be used in the computations. The columns `age`
  and `weight` are disallowed and will not be used. Usually, there
  should either be an allowed or disallowed list, but not both: if there is an explicit
  allowed list, all other columns are automatically disallowed.

Data sharing
------------

For each iteration, two subtasks are created whose main job is to compute the local beta
coefficients and the local deviance, respectively. On top of that, the subtasks also
share some metadata with the central part of the task to ensure that the computation is
done correctly. The following data is shared between the parties:

- **Subtask 1**:

  - *Beta coefficients*: To compute the beta computations in the central part, several
    matrix multiplication results are shared, namely *X(T) W X* and *X(T) W z*. See
    the `Cellamare et al. (2022) <https://www.mdpi.com/1999-4893/15/7/243>`_ paper for
    more information.
  - *Dispersion*: For Gaussian distributions, the dispersion is shared. This is used to
    compute the standard error.
  - *Number of observations*: The number of rows used in the computation.
  - *Number of variables*: The number of parameters used in the computation, given by
    the design matrix.
  - *The sum of the predictor variable*. This is used to compute the global average of
    the predictor variable, which is used in the computation of the null deviance.

- **Subtask 2**:

  - *Local deviance*: The local deviance of the current iteration.
  - *Local deviance of the previous iteration*: The local deviance of the previous
    iteration.
  - *Local null deviance*: The local null deviance of the current iteration.

It is unlikely that data is inferred from these results if the amount of data is
large enough.

Vulnerabilities to known attacks
--------------------------------

.. list-table::
    :widths: 25 10 65
    :header-rows: 1

    * - Attack
      - Risk eliminated?
      - Risk analysis
    * - Reconstruction
      - ⚠
      - Reconstruction may be possible in an iterative process where one data station
        iteratively modifies their own data to match the partial results from another
        data station. This would be a brute force attack that likely requires many
        iterations. The risk can be reduced by limiting the number of tasks that a
        user can make.
    * - Differencing
      - ✔
      - The shared statistics are derived from the data in such a way that when a single
        data point is added, that data point is not derivable from the shared statistics.
    * - Deep Leakage from Gradients (DLG)
      - ✔
      - Only statistics derived from the gradient are shared, from which the gradient
        can not be reconstructed.
    * - Generative Adversarial Networks (GAN)
      - ✔
      -
    * - Model Inversion
      - ✔
      -
    * - Watermark Attack
      - ✔
      -