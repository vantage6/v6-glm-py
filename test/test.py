"""
Run pytest to test the GLM algorithm locally using MockNetwork.

    uv sync --group dev
    uv run pytest test/test.py -v
"""

import os
import pytest

import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

from vantage6.algorithm.mock.network import MockNetwork
from vantage6.algorithm.tools.exceptions import (
    UserInputError,
    PrivacyThresholdViolation,
    NodePermissionException,
)

current_path = Path(__file__).parent
DATABASE_LABEL = "Database"


def _load_family_dfs(family: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (current_path / family / "a.csv").exists():
        raise FileNotFoundError(f"Data files not found for family {family}!")
    return (
        pd.read_csv(current_path / family / "a.csv"),
        pd.read_csv(current_path / family / "b.csv"),
        pd.read_csv(current_path / family / "c.csv"),
    )


def get_mock_network(
    family: str,
    dfs: list[pd.DataFrame] | None = None,
) -> tuple[MockNetwork, object, list[dict], list[int]]:
    if dfs is None:
        dfs = list(_load_family_dfs(family))
    network = MockNetwork(
        datasets=[
            {DATABASE_LABEL: {"database": dfs[0]}},
            {DATABASE_LABEL: {"database": dfs[1]}},
            {DATABASE_LABEL: {"database": dfs[2]}},
        ],
        module_name="v6-glm-py",
    )
    client = network.user_client
    databases = [{"type": "dataframe", "dataframe_id": network.hq.dataframes[0]["id"]}]
    org_ids = [organization["id"] for organization in client.organization.list()]
    return network, client, databases, org_ids


def get_mock_network_poisson():
    return get_mock_network("poisson")


def get_mock_network_binomial(dfs: list[pd.DataFrame] | None = None):
    return get_mock_network("binomial", dfs)


def get_mock_network_gaussian():
    return get_mock_network("gaussian")


def get_mock_network_survival():
    return get_mock_network("survival")


def test_central_1_iteration():
    _, client, databases, org_ids = get_mock_network_poisson()
    central_task = client.task.create(
        method="glm",
        arguments={
            "outcome_variable": "num_awards",
            "predictor_variables": ["prog", "math"],
            "family": "poisson",
            "max_iterations": 1,
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    details = results[0]["details"]
    assert details["converged"] == False
    assert details["iterations"] == 1
    assert details["dispersion"] == 1
    assert details["is_dispersion_estimated"] == False
    assert details["num_observations"] == 200
    assert details["num_variables"] == 4


def test_central_until_convergence_poisson(assert_almost_equal: callable):
    """Test the GLM algorithm with Poisson family until convergence"""
    _, client, databases, org_ids = get_mock_network_poisson()
    central_task = client.task.create(
        method="glm",
        arguments={
            "outcome_variable": "num_awards",
            "predictor_variables": ["prog", "math"],
            "family": "poisson",
            "category_reference_values": {"prog": "General"},
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    details = results[0]["details"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -5.24712)
    assert_almost_equal(coefficients["beta"]["prog[T.Academic]"], 1.08385)
    assert_almost_equal(coefficients["beta"]["prog[T.Vocational]"], 0.36980)
    assert_almost_equal(coefficients["beta"]["math"], 0.07015)
    assert_almost_equal(coefficients["p_value"]["Intercept"], 1.6013e-15)
    assert_almost_equal(coefficients["p_value"]["prog[T.Academic]"], 0.002483)
    assert_almost_equal(coefficients["p_value"]["prog[T.Vocational]"], 0.4018)
    assert_almost_equal(coefficients["p_value"]["math"], 3.62500e-11)
    assert_almost_equal(coefficients["std_error"]["Intercept"], 0.6584)
    assert_almost_equal(coefficients["std_error"]["prog[T.Academic]"], 0.3583)
    assert_almost_equal(coefficients["std_error"]["prog[T.Vocational]"], 0.44107)
    assert_almost_equal(coefficients["std_error"]["math"], 0.010599)
    assert_almost_equal(coefficients["z_value"]["Intercept"], -7.96887)
    assert_almost_equal(coefficients["z_value"]["prog[T.Academic]"], 3.025401)
    assert_almost_equal(coefficients["z_value"]["prog[T.Vocational]"], 0.83844)
    assert_almost_equal(coefficients["z_value"]["math"], 6.6186)

    assert details["converged"] == True
    assert details["iterations"] == 5
    assert details["dispersion"] == 1
    assert details["is_dispersion_estimated"] == False
    assert_almost_equal(details["deviance"], 189.4496)
    assert_almost_equal(details["null_deviance"], 287.67223)
    assert details["num_observations"] == 200
    assert details["num_variables"] == 4


def test_central_binomial_missing_categorical_reference_values(
    assert_almost_equal: callable,
):
    df_a, df_b, df_c = _load_family_dfs("binomial")
    df_a = df_a.loc[df_a["rank"] != 4].reset_index(drop=True)
    _, client, databases, org_ids = get_mock_network_binomial([df_a, df_b, df_c])

    central_task = client.task.create(
        method="glm",
        arguments={
            "outcome_variable": "admit",
            "predictor_variables": ["gre", "gpa", "rank"],
            "family": "binomial",
            "categorical_predictors": ["rank"],
            "category_reference_values": {"rank": 1},
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    details = results[0]["details"]

    assert_almost_equal(coefficients["beta"]["Intercept"], -3.89)
    assert_almost_equal(coefficients["beta"]["gre"], 0.00232)
    assert_almost_equal(coefficients["beta"]["gpa"], 0.765)
    assert_almost_equal(coefficients["beta"]["rank[T.2]"], -0.677)
    assert_almost_equal(coefficients["beta"]["rank[T.3]"], -1.34)
    assert_almost_equal(coefficients["beta"]["rank[T.4]"], -1.57)

    assert_almost_equal(coefficients["std_error"]["Intercept"], 1.15)
    assert_almost_equal(coefficients["std_error"]["gre"], 0.00113)
    assert_almost_equal(coefficients["std_error"]["gpa"], 0.337)
    assert_almost_equal(coefficients["std_error"]["rank[T.2]"], 0.316)
    assert_almost_equal(coefficients["std_error"]["rank[T.3]"], 0.345)
    assert_almost_equal(coefficients["std_error"]["rank[T.4]"], 0.476)

    assert_almost_equal(coefficients["z_value"]["Intercept"], -3.39)
    assert_almost_equal(coefficients["z_value"]["gre"], 2.06)
    assert_almost_equal(coefficients["z_value"]["gpa"], 2.27)
    assert_almost_equal(coefficients["z_value"]["rank[T.2]"], -2.14)
    assert_almost_equal(coefficients["z_value"]["rank[T.3]"], -3.87)
    assert_almost_equal(coefficients["z_value"]["rank[T.4]"], -3.30)

    assert_almost_equal(coefficients["p_value"]["Intercept"], 0.000698)
    assert_almost_equal(coefficients["p_value"]["gre"], 0.0390)
    assert_almost_equal(coefficients["p_value"]["gpa"], 0.0231)
    assert_almost_equal(coefficients["p_value"]["rank[T.2]"], 0.0323)
    assert_almost_equal(coefficients["p_value"]["rank[T.3]"], 0.000107)
    assert_almost_equal(coefficients["p_value"]["rank[T.4]"], 0.000953)

    assert details["converged"] == True
    assert details["iterations"] == 4
    assert_almost_equal(details["dispersion"], 1.00)
    assert details["is_dispersion_estimated"] == False
    assert_almost_equal(details["deviance"], 439)
    assert_almost_equal(details["null_deviance"], 478)
    assert details["num_observations"] == 379
    assert details["num_variables"] == 6


def test_central_until_convergence_gaussian(assert_almost_equal: callable):
    _, client, databases, org_ids = get_mock_network_gaussian()
    central_task = client.task.create(
        method="glm",
        arguments={
            "outcome_variable": "Volume",
            "predictor_variables": ["Girth"],
            "family": "gaussian",
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -36.94346)
    assert_almost_equal(coefficients["beta"]["Girth"], 5.0658564)
    assert_almost_equal(coefficients["p_value"]["Intercept"], 4.86364e-23)
    assert_almost_equal(coefficients["p_value"]["Girth"], 2.229553e-37)
    assert_almost_equal(coefficients["std_error"]["Intercept"], 2.339522)
    assert_almost_equal(coefficients["std_error"]["Girth"], 0.171981)
    assert_almost_equal(coefficients["z_value"]["Intercept"], -15.7910)
    assert_almost_equal(coefficients["z_value"]["Girth"], 29.45576)
    details = results[0]["details"]
    assert details["converged"] == True
    assert_almost_equal(details["deviance"], 1048.6051)
    assert_almost_equal(details["dispersion"], 17.47675)
    assert details["is_dispersion_estimated"] == True
    assert details["iterations"] == 2
    assert_almost_equal(details["null_deviance"], 16212.1677)
    assert details["num_observations"] == 62
    assert details["num_variables"] == 2


def test_central_until_convergence_binomial(assert_almost_equal: callable):
    _, client, databases, org_ids = get_mock_network_binomial()
    central_task = client.task.create(
        method="glm",
        arguments={
            "outcome_variable": "admit",
            "predictor_variables": ["gre", "gpa", "rank"],
            "family": "binomial",
            "categorical_predictors": ["rank"],
            "category_reference_values": {"rank": 1},
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -3.989979)
    assert_almost_equal(coefficients["beta"]["gre"], 0.002264)
    assert_almost_equal(coefficients["beta"]["gpa"], 0.804038)
    assert_almost_equal(coefficients["beta"]["rank[T.2]"], -0.675442)
    assert_almost_equal(coefficients["beta"]["rank[T.3]"], -1.340204)
    assert_almost_equal(coefficients["beta"]["rank[T.4]"], -1.551464)
    assert_almost_equal(coefficients["p_value"]["Intercept"], 0.000465027)
    assert_almost_equal(coefficients["p_value"]["gre"], 0.038465)
    assert_almost_equal(coefficients["p_value"]["gpa"], 0.01538789)
    assert_almost_equal(coefficients["p_value"]["rank[T.2]"], 0.032828)
    assert_almost_equal(coefficients["p_value"]["rank[T.3]"], 0.00010394)
    assert_almost_equal(coefficients["p_value"]["rank[T.4]"], 0.00020471)
    assert_almost_equal(coefficients["std_error"]["Intercept"], 1.139950)
    assert_almost_equal(coefficients["std_error"]["gre"], 0.00109399)
    assert_almost_equal(coefficients["std_error"]["gpa"], 0.331819)
    assert_almost_equal(coefficients["std_error"]["rank[T.2]"], 0.31648)
    assert_almost_equal(coefficients["std_error"]["rank[T.3]"], 0.34530)
    assert_almost_equal(coefficients["std_error"]["rank[T.4]"], 0.41783)
    assert_almost_equal(coefficients["z_value"]["Intercept"], -3.500)
    assert_almost_equal(coefficients["z_value"]["gre"], 2.069863)
    assert_almost_equal(coefficients["z_value"]["gpa"], 2.423)
    assert_almost_equal(coefficients["z_value"]["rank[T.2]"], -2.13417)
    assert_almost_equal(coefficients["z_value"]["rank[T.3]"], -3.88120)
    assert_almost_equal(coefficients["z_value"]["rank[T.4]"], -3.71313)

    details = results[0]["details"]
    assert details["converged"] == True
    assert_almost_equal(details["deviance"], 458.51749)
    assert details["dispersion"] == 1
    assert details["is_dispersion_estimated"] == False
    assert details["iterations"] == 4
    assert_almost_equal(details["null_deviance"], 499.9765)
    assert details["num_observations"] == 400
    assert details["num_variables"] == 6


def test_central_until_convergence_binomial_rr(assert_almost_equal: callable):
    _, client, databases, org_ids = get_mock_network_binomial()
    central_task = client.task.create(
        method="glm",
        arguments={
            "outcome_variable": "admit",
            "predictor_variables": ["gre", "gpa", "rank"],
            "family": "binomial",
            "categorical_predictors": ["rank"],
            "category_reference_values": {"rank": 1},
            "link_function": "log",
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -3.22909)

    rr = {var: round(float(np.exp(coef)), 6) for var, coef in coefficients["beta"].items()}
    assert_almost_equal(rr["Intercept"], 0.0396)
    assert_almost_equal(rr["gre"], 1.0023)
    assert_almost_equal(rr["gpa"], 2.2053)
    assert_almost_equal(rr["rank[T.2]"], 0.1610)
    assert_almost_equal(rr["rank[T.3]"], 0.0993)
    assert_almost_equal(rr["rank[T.4]"], 0.0852)

    details = results[0]["details"]
    assert details["converged"] == True
    assert details["dispersion"] == 1
    assert details["is_dispersion_estimated"] == False


def test_central_binomial_with_formula(assert_almost_equal: callable):
    _, client, databases, org_ids = get_mock_network_binomial()
    central_task = client.task.create(
        method="glm",
        arguments={
            "formula": "admit ~ gre + gpa + I(rank == 3)",
            "family": "binomial",
            "categorical_predictors": ["rank"],
            "category_reference_values": {"rank": 1},
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -5.03136)
    assert_almost_equal(coefficients["beta"]["gre"], 0.0024847)
    assert_almost_equal(coefficients["beta"]["gpa"], 0.8674160)
    assert_almost_equal(coefficients["beta"]["I(rank == 3)"], -0.65508)
    assert_almost_equal(coefficients["p_value"]["Intercept"], 4.085592e-06)
    assert_almost_equal(coefficients["p_value"]["gre"], 0.02005)
    assert_almost_equal(coefficients["p_value"]["gpa"], 0.00770)
    assert_almost_equal(coefficients["p_value"]["I(rank == 3)"], 0.01118)
    assert_almost_equal(coefficients["std_error"]["Intercept"], 1.092117)
    assert_almost_equal(coefficients["std_error"]["gre"], 0.0010685)
    assert_almost_equal(coefficients["std_error"]["gpa"], 0.3255194)
    assert_almost_equal(coefficients["std_error"]["I(rank == 3)"], 0.258222)
    assert_almost_equal(coefficients["z_value"]["Intercept"], -4.60698)
    assert_almost_equal(coefficients["z_value"]["gre"], 2.32533)
    assert_almost_equal(coefficients["z_value"]["gpa"], 2.66471)
    assert_almost_equal(coefficients["z_value"]["I(rank == 3)"], -2.5368)

    details = results[0]["details"]
    assert details["converged"] == True
    assert_almost_equal(details["deviance"], 473.571245172)
    assert details["dispersion"] == 1
    assert details["is_dispersion_estimated"] == False
    assert details["iterations"] == 4
    assert_almost_equal(details["null_deviance"], 499.9765)
    assert details["num_observations"] == 400
    assert details["num_variables"] == 4


def test_central_binomial_with_formula_2(assert_almost_equal: callable):
    _, client, databases, org_ids = get_mock_network_binomial()
    central_task = client.task.create(
        method="glm",
        arguments={
            "formula": "admit ~ rank + I(log(gre) + gpa)",
            "family": "binomial",
            "categorical_predictors": ["rank"],
            "category_reference_values": {"rank": 1},
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -9.10991938)
    assert_almost_equal(coefficients["beta"]["I(log(gre) + gpa)"], 0.942246)
    assert_almost_equal(coefficients["beta"]["rank[T.2]"], -0.67615)
    assert_almost_equal(coefficients["beta"]["rank[T.3]"], -1.35706)
    assert_almost_equal(coefficients["beta"]["rank[T.4]"], -1.55692)

    details = results[0]["details"]
    assert_almost_equal(details["deviance"], 458.8201015594518)
    assert details["iterations"] == 4
    assert_almost_equal(details["null_deviance"], 499.9765)
    assert details["num_observations"] == 400
    assert details["num_variables"] == 5


def test_central_survival(assert_almost_equal: callable):
    _, client, databases, org_ids = get_mock_network_survival()
    central_task = client.task.create(
        method="glm",
        arguments={
            "outcome_variable": "status",
            "predictor_variables": ["sex", "age"],
            "family": "survival",
            "survival_sensor_column": "survival_sensor_column",
            "categorical_predictors": ["sex"],
            "category_reference_values": {"sex": 0},
        },
        organizations=[org_ids[0]],
        databases=databases,
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -1.29697)
    assert_almost_equal(coefficients["beta"]["age"], 0.0102645)
    assert_almost_equal(coefficients["beta"]["sex[T.1]"], 0.0215628)
    assert_almost_equal(coefficients["p_value"]["Intercept"], 3.6631e-05)
    assert_almost_equal(coefficients["p_value"]["age"], 0.14887)
    assert_almost_equal(coefficients["p_value"]["sex[T.1]"], 0.8594638)
    assert_almost_equal(coefficients["std_error"]["Intercept"], 0.31420694)
    assert_almost_equal(coefficients["std_error"]["age"], 0.007110791)
    assert_almost_equal(coefficients["std_error"]["sex[T.1]"], 0.121785)
    assert_almost_equal(coefficients["z_value"]["Intercept"], -4.1277603)
    assert_almost_equal(coefficients["z_value"]["age"], 1.4435127068)
    assert_almost_equal(coefficients["z_value"]["sex[T.1]"], 0.177056648)

    details = results[0]["details"]
    assert details["converged"] == True
    assert_almost_equal(details["deviance"], 407.594184)
    assert details["dispersion"] == 1
    assert details["is_dispersion_estimated"] == False
    assert details["iterations"] == 9
    assert_almost_equal(details["null_deviance"], 380.052351)
    assert details["num_observations"] == 1000
    assert details["num_variables"] == 3


def test_wrong_user_input():
    _, client, databases, org_ids = get_mock_network_poisson()
    arguments = {
        "outcome_variable": "num_awards",
        "predictor_variables": ["prog", "math"],
        "family": "poisson",
        "category_reference_values": {"prog": "General"},
    }
    task_kwargs = {
        "method": "glm",
        "organizations": [org_ids[0]],
        "databases": databases,
    }

    with pytest.raises(
        UserInputError,
        match="Family non-existing-family not supported. Please provide one of the "
        "supported families: poisson, binomial, gaussian, survival",
    ):
        wrong_arguments = deepcopy(arguments)
        wrong_arguments["family"] = "non-existing-family"
        client.task.create(arguments=wrong_arguments, **task_kwargs)

    with pytest.raises(
        UserInputError,
        match="Either formula or outcome and predictor variables should be provided. "
        "Neither is provided.",
    ):
        wrong_arguments = deepcopy(arguments)
        wrong_arguments.pop("predictor_variables")
        wrong_arguments.pop("outcome_variable")
        client.task.create(arguments=wrong_arguments, **task_kwargs)

    os.environ["GLM_ALLOWED_COLUMNS"] = "bla,bla2"
    with pytest.raises(NodePermissionException):
        client.task.create(arguments=arguments, **task_kwargs)
    os.environ["GLM_ALLOWED_COLUMNS"] = "prog,math,num_awards"
    client.task.create(arguments=arguments, **task_kwargs)
    del os.environ["GLM_ALLOWED_COLUMNS"]

    os.environ["GLM_DISALLOWED_COLUMNS"] = "prog"
    with pytest.raises(NodePermissionException):
        client.task.create(arguments=arguments, **task_kwargs)
    os.environ["GLM_DISALLOWED_COLUMNS"] = "bla,bla2"
    client.task.create(arguments=arguments, **task_kwargs)
    del os.environ["GLM_DISALLOWED_COLUMNS"]

    with pytest.raises(UserInputError):
        survival_arguments = deepcopy(arguments)
        survival_arguments["family"] = "survival"
        client.task.create(arguments=survival_arguments, **task_kwargs)

    df_a, df_b, df_c = _load_family_dfs("poisson")
    _, client, databases, org_ids = get_mock_network_poisson([df_a.head(3), df_b, df_c])
    with pytest.raises(PrivacyThresholdViolation):
        client.task.create(arguments=arguments, **task_kwargs)

    df_a = deepcopy(df_a)
    df_a["prog"] = None
    _, client, databases, org_ids = get_mock_network_poisson([df_a, df_b, df_c])
    with pytest.raises(PrivacyThresholdViolation):
        client.task.create(arguments=arguments, **task_kwargs)

    many_var_df = deepcopy(df_a)
    for col in range(20):
        many_var_df[f"col_{col}"] = 1
    _, client, databases, org_ids = get_mock_network_poisson(
        [many_var_df, many_var_df, many_var_df]
    )
    with pytest.raises(PrivacyThresholdViolation):
        extra_arguments = deepcopy(arguments)
        extra_arguments["predictor_variables"].extend([f"col_{col}" for col in range(20)])
        client.task.create(arguments=extra_arguments, **task_kwargs)

    df_a, df_b, _ = _load_family_dfs("poisson")
    network = MockNetwork(
        datasets=[
            {DATABASE_LABEL: {"database": df_a}},
            {DATABASE_LABEL: {"database": df_b}},
        ],
        module_name="v6-glm-py",
    )
    client = network.user_client
    databases = [{"type": "dataframe", "dataframe_id": network.hq.dataframes[0]["id"]}]
    org_ids = [organization["id"] for organization in client.organization.list()]
    with pytest.raises(UserInputError):
        client.task.create(
            method="glm",
            arguments=arguments,
            organizations=[org_ids[0]],
            databases=databases,
        )
