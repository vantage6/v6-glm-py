"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""

from typing import Any

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

from .common import Family, get_family
from .constants import DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE


# TODO implement offset - below is R example. Not sure how to implement the offset in
# the formula in Python. Maybe it is not needed?

# Consider a formula that includes an offset term:

# formula <- y ~ x1 + x2 + offset(log(exposure))
# data <- data.frame(y = c(1, 2, 3), x1 = c(4, 5, 6), x2 = c(7, 8, 9), exposure = c(10, 20, 30))

# In this case, model.frame(formula, data = data) creates a data frame that includes y, x1, x2,
# and log(exposure). The model.offset function then extracts the log(exposure) values as the offset.


@algorithm_client
def glm(
    client: AlgorithmClient,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str = Family.GAUSSIAN.value,
    dstar: str = None,
    types: list[str] = None,
    tolerance_level: int = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    organizations_to_include: list[int] = None,
) -> Any:
    """Central part of the algorithm
    TODO docstring
    """
    # select organizations to include
    if not organizations_to_include:
        organizations = client.organization.list()
        organizations_to_include = [
            organization.get("id") for organization in organizations
        ]

    # Iterate to find the coefficients
    iteration = 0
    betas = None
    while iteration < max_iterations:
        # print iteration header
        _log_header(iteration)

        # compute beta coefficients
        partial_betas = _compute_partial_betas(
            client,
            outcome_variable,
            predictor_variables,
            family,
            dstar,
            types,
            is_first_iteration=iteration == 0,
            organizations_to_include=organizations_to_include,
            betas=betas,
        )
        info(" - Partial betas obtained!")

        # compute central betas from the partial betas
        info("Computing central betas")
        betas = _compute_central_betas(partial_betas, family, dstar)
        info(" - Central betas obtained!")

        converged = True  ## TODO add convergence criterion
        if converged:
            break
        iteration += 1

    # return the final results of the algorithm
    return {}


def _compute_central_betas(
    partial_betas: dict,
    family_str: str,
    dstar: str,
):
    """TODO docstring"""
    family = get_family(family_str)
    return []


def _compute_partial_betas(
    client: AlgorithmClient,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    dstar: str,
    types: list[str],
    is_first_iteration: bool,
    organizations_to_include: list[int],
    betas: list[int] | None = None,
):
    """TODO docstring"""
    info("Defining input parameters")
    input_ = {
        "method": "compute_partial_betas",
        "kwargs": {
            "outcome_variable": outcome_variable,
            "predictor_variables": predictor_variables,
            "family": family,
            "dstar": dstar,
            "types": types,
            "is_first_iteration": is_first_iteration,
            "beta_coefficients": betas,
        },
    }

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="My subtask",
        description="This is a very important subtask",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")
    return results


def _log_header(num_iteration: int):
    info("")
    info("#" * 60)
    info(f"# Starting iteration {num_iteration}")
    info("#" * 60)
    info("")
