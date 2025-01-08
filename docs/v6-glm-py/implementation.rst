Implementation
==============

Overview
--------

This is an algorithm that is executed iteratively. After initialization, the
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

Central (``glm``)
-----------------
The central part is responsible for the orchestration and aggregation of the algorithm.

.. Describe the central function here.

Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``compute_local_betas``
~~~~~~~~~~~~~~~~

.. Describe the partial function.

