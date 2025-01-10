Implementation
==============

Overview
--------

For details on the mathematical background of this implementation, please refer to the
`Cellomare et al. (2022) <https://www.mdpi.com/1999-4893/15/7/243>`_ paper. Here, we
will not describe the mathematical background, but rather the flow of the algorithm.

The GLM algorithm is executed iteratively. After initialization, the
algorithm computes partial beta values on each node. These partial beta values are then
aggregated on the central node. With the overall beta values, each node can compute the
local deviance. Next, the central node computes the global deviance. If the deviance
is virtually the same as the previous iteration, the algorithm stops. Otherwise, the
algorithm again requests the nodes to compute the partial beta values, etc.

.. uml::

  !theme superhero-outline

  caption The central part of the algorithm is responsible for the \
          orchestration and aggregation\n of the algorithm. The partial \
          parts are executed on each node.

  |client|
  :request analysis;

  |central|
  :Collect organizations
  in collaboration;
  :Start iteration;
  :Create partial tasks;

  |partial|
  :Check if request complies
  with privacy settings;

  |partial|
  :Compute partial beta values;

  |central|
  :Compute overall beta values;
  :Create new partial tasks;

  |partial|
  :Compute the deviance using
  the overall beta values;

  |central|
  :Compute overall deviance;
  :If algorithm has converged
  or max iterations have been
  reached, stop. Otherwise,
  start a new iteration with
  new betas;

  |client|
  :Receive results;

Partials
--------

Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``compute_local_betas``
~~~~~~~~~~~~~~~~

This function iteratively computes the partial beta values on each node.  To achieve
this, the following steps are executed:

1. The data is loaded and the design matrix is computed using the formula
2. Several privacy checks are executed - see the :ref:`privacy guards <privacy-guards>`
   section for more information.
3. The `eta` values are computed using the overall beta values provided by the central
   function. In the first iteration, the family's link function is used to compute the
   `eta` values.
4. The `mu`, `z` and `W` values are computed using the `eta` values.
5. These values are used to compute the new partial beta values.
6. The partial beta values are returned to the central function, alongside some
   metadata, being the dispersion, number of observations, number of variables and the
   sum of the outcome variable.

``compute_local_deviance``
~~~~~~~~~~~~~~~~

The local deviance function computes the deviance on each node. The deviance is computed
using the overall beta values provided by the central function. The following steps are
executed:

1. The data is loaded and the design matrix is computed using the formula
2. Several privacy checks are executed - see the :ref:`privacy guards <privacy-guards>`
   section for more information. Note that these should not yield different results than
   the checks in the `compute_local_betas` function - unless the data provided to the
   node has changed in the meantime (for instance, if the node was restarted).
3. The `eta` values are computed using the overall beta values provided by the central
   function. The central function provides the betas from the previous iteration as well
   as the current iteration. These are used to compute the old and new `eta` values.
4. The `mu` values are computed using the `eta` values, for the old and new `eta`
   values.
5. The local deviance is computed using the `mu` values and the outcome variable.
6. The null deviance is computed using the global average of the outcome variable.
7. The local deviance of the current iteration, the previous iteration, and the local
   null deviance are returned to the central function.

Central (``glm``)
-----------------

The central part is responsible for the orchestration and aggregation of the algorithm.
It executes the following steps:

1. Collect organizations in collaboration.
2. Start an iteration, which consists of the following steps:
    1. Create partial task to compute local betas.
    2. Collect the partial beta results.
    3. Compute the overall beta values. Also, compute the overall dispersion, number of
       observations, number of variables, and the average of the outcome variable.
    4. Create new partial tasks to compute the local deviance.
    5. Collect the partial deviance results.
    6. Compute the overall deviance. This is simply the sum of the local deviances.
    7. If the deviance changes very little (below the tolerance threshold), the
       algorithm has converged. If the algorithm has converged or the maximum number of
       iterations has been reached, the algorithm stops. Otherwise, start a new
       iteration.
3. Use the final overall beta values to compute standard errors, Z-values and p-values.
4. Return the overall beta values together with the standard errors, Z-values, and
   p-values. Also, return the dispersion, number of observations, number of variables,
   number of iterations, deviance, null deviance, and whether the algorithm has
   converged or not.

