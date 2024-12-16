"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""

from functools import reduce
from typing import Any
import numpy as np
import pandas as pd
import scipy.stats as stats
from pprint import pprint  # TODO remove this import when done debugging

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
    iteration = 1
    betas = None
    while iteration <= max_iterations:
        converged, new_betas, deviance = _do_iteration(
            iteration=iteration,
            client=client,
            outcome_variable=outcome_variable,
            predictor_variables=predictor_variables,
            family=family,
            dstar=dstar,
            types=types,
            tolerance_level=tolerance_level,
            organizations_to_include=organizations_to_include,
            betas_old=betas,
        )
        betas = new_betas["beta_estimates"]

        # terminate if converged or reached max iterations
        if converged:
            info(" - Converged!")
            break
        if iteration == max_iterations:
            warn(" - Maximum number of iterations reached!")
            break
        iteration += 1

    # after the iteration, return the final results
    info("Preparing final results")
    betas = pd.Series(betas)
    std_errors = pd.Series(new_betas["std_error_betas"])
    zvalue = betas / std_errors
    if new_betas["is_dispersion_estimated"]:
        # TODO this code needs to be checked when running with Gaussian family
        pvalue = 2 * stats.t.cdf(
            -np.abs(zvalue), betas["num_observations"] - betas["num_variables"]
        )
    else:
        pvalue = 2 * stats.norm.cdf(-np.abs(zvalue))

    # add back indices to pvalue
    pvalue = pd.Series(pvalue, index=betas.index)

    # create dataframe with results
    results = pd.DataFrame(
        {
            "beta": betas,
            "std_error": std_errors,
            "z_value": zvalue,
            "p_value": pvalue,
        }
    )

    return {
        "coefficients": results.to_dict(),
        "details": {
            "converged": converged,
            "iterations": iteration,
            "dispersion": new_betas["dispersion"],
            "is_dispersion_estimated": new_betas["is_dispersion_estimated"],
            "deviance": deviance["new"],
            "null_deviance": deviance["null"],
            "num_observations": new_betas["num_observations"],
            "num_variables": new_betas["num_variables"],
        },
    }


def _do_iteration(
    iteration: int,
    client: AlgorithmClient,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    dstar: str,
    types: list[str],
    tolerance_level: int,
    organizations_to_include: list[int],
    betas_old: dict | None = None,
) -> bool:
    """TODO docstring"""
    # print iteration header
    _log_header(iteration)

    # compute beta coefficients
    partial_betas = _compute_local_betas(
        client,
        outcome_variable,
        predictor_variables,
        family,
        dstar,
        types,
        iter_num=iteration,
        organizations_to_include=organizations_to_include,
        betas=betas_old,
    )
    info(" - Partial betas obtained!")

    # compute central betas from the partial betas
    info("Computing central betas")
    new_betas = _compute_central_betas(partial_betas, family)
    info(" - Central betas obtained!")

    # compute the deviance for each of the nodes
    info("Computing deviance")
    deviance_partials = _compute_partial_deviance(
        client=client,
        outcome_variable=outcome_variable,
        predictor_variables=predictor_variables,
        family=family,
        iter_num=iteration,
        dstar=dstar,
        beta_estimates=new_betas["beta_estimates"],
        beta_estimates_previous=betas_old,
        weighted_derivative_mu=new_betas["normalized_weighted_y_sum"],
        types=types,
        organizations_to_include=organizations_to_include,
    )
    print("deviance_partials")
    pprint(deviance_partials)

    total_deviance = _compute_deviance(deviance_partials)
    info(" - Deviance computed!")
    pprint(total_deviance)

    # check if the algorithm has converged
    converged = False
    if total_deviance["new"] == 0:
        # Safety check to avoid division by zero
        converged = True
    convergence_criterion = (
        abs(total_deviance["old"] - total_deviance["new"]) / total_deviance["new"]
    )
    if convergence_criterion < tolerance_level:
        converged = True
    return converged, new_betas, total_deviance


def _compute_central_betas(
    partial_betas: list[dict],
    family: str,
):
    """TODO docstring"""

    # sum the contributions of the partial betas
    info("Summing contributions of partial betas")
    sum_weights = sum([partial["weights_sum"] for partial in partial_betas])
    if sum_weights == 0:
        error("Sum of weights is zero. Cannot compute central betas.")
        exit(1)

    normalized_weighted_y_sum = sum(
        [partial["weighted_sum_of_y"] / sum_weights for partial in partial_betas]
    )
    XTX_sum = reduce(
        lambda x, y: x + y, [pd.DataFrame(partial["XTX"]) for partial in partial_betas]
    )
    XTz_sum = reduce(
        lambda x, y: x + y, [pd.DataFrame(partial["XTz"]) for partial in partial_betas]
    )
    dispersion_sum = sum([partial["dispersion"] for partial in partial_betas])
    num_observations = sum([partial["num_observations"] for partial in partial_betas])
    num_variables = partial_betas[0]["num_variables"]

    # TODO no idea if this is correct. It's just a translation of the R code
    if family == Family.POISSON.value or family == Family.BINOMIAL.value:
        dispersion = 1
        # TODO we can probably remove this and use the family object instead
        is_dispersion_estimated = False
    else:
        dispersion = dispersion_sum / (num_observations - num_variables)
        is_dispersion_estimated = True

    info("Updating betas")

    XTX_np = XTX_sum.to_numpy()
    XTz_np = XTz_sum.to_numpy()

    beta_estimates = np.linalg.solve(XTX_np, XTz_np).flatten()
    std_error_betas = np.sqrt(np.diag(np.linalg.inv(XTX_np) * dispersion))

    # add the indices back to the beta estimates
    indices = pd.DataFrame(partial_betas[0]["XTX"]).index
    beta_estimates = pd.Series(beta_estimates, index=indices)
    std_error_betas = pd.Series(std_error_betas, index=indices)

    return {
        "beta_estimates": beta_estimates.to_dict(),
        "std_error_betas": std_error_betas.to_dict(),
        "dispersion": dispersion,
        "is_dispersion_estimated": is_dispersion_estimated,
        "num_observations": num_observations,
        "num_variables": num_variables,
        "normalized_weighted_y_sum": normalized_weighted_y_sum,
    }


def _compute_deviance(
    partial_deviances: list[dict],
):
    """TODO docstring"""
    total_deviance_null = sum(
        [partial["deviance_null"] for partial in partial_deviances]
    )
    total_deviance_old = sum([partial["deviance_old"] for partial in partial_deviances])
    total_deviance_new = sum([partial["deviance_new"] for partial in partial_deviances])
    return {
        "null": total_deviance_null,
        "old": total_deviance_old,
        "new": total_deviance_new,
    }


def _compute_local_betas(
    client: AlgorithmClient,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    dstar: str,
    types: list[str],
    iter_num: int,
    organizations_to_include: list[int],
    betas: list[int] | None = None,
):
    """TODO docstring"""
    info("Defining input parameters")
    print(betas)
    input_ = {
        "method": "compute_local_betas",
        "kwargs": {
            "outcome_variable": outcome_variable,
            "predictor_variables": predictor_variables,
            "family": family,
            "dstar": dstar,
            "types": types,
            "is_first_iteration": iter_num == 1,
            "beta_coefficients": betas,
        },
    }
    print(input_)

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="Partial betas subtask",
        description=f"Subtask to compute partial betas - iteration {iter_num}",
    )

    # wait for node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")
    return results


def _compute_partial_deviance(
    client: AlgorithmClient,
    outcome_variable: str,
    predictor_variables: list[str],
    family: str,
    iter_num: int,
    dstar: str,
    beta_estimates: pd.Series,
    beta_estimates_previous: pd.Series,
    weighted_derivative_mu: list[int],
    types: list[str],
    organizations_to_include: list[int],
):
    """TODO docstring"""
    info("Defining input parameters")
    input_ = {
        "method": "compute_local_deviance",
        "kwargs": {
            "outcome_variable": outcome_variable,
            "predictor_variables": predictor_variables,
            "family": family,
            "is_first_iteration": iter_num == 1,
            "dstar": dstar,
            "beta_coefficients": beta_estimates,
            "beta_coefficients_previous": beta_estimates_previous,
            "weighted_derivative_mu": weighted_derivative_mu,
        },
    }

    # create a subtask for all organizations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=organizations_to_include,
        name="Partial deviance subtask",
        description=f"Subtask to compute partial deviance - iteration {iter_num}",
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
