"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python test.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""

import os
import pytest
from pathlib import Path
from copy import deepcopy

from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
from vantage6.algorithm.tools.exceptions import (
    UserInputError,
    PrivacyThresholdViolation,
    NodePermissionException,
)

# get path of current directory
current_path = Path(__file__).parent


def get_mock_client(family: str):
    # check if data file exists
    if not (current_path / family / "a.csv").exists():
        raise FileNotFoundError(f"Data files not found for family {family}!")
    return MockAlgorithmClient(
        datasets=[
            # Data for first organization
            [
                {
                    "database": current_path / family / "a.csv",
                    "db_type": "csv",
                    "input_data": {},
                }
            ],
            # Data for second organization
            [
                {
                    "database": current_path / family / "b.csv",
                    "db_type": "csv",
                    "input_data": {},
                }
            ],
            # Data for third organization
            [
                {
                    "database": current_path / family / "c.csv",
                    "db_type": "csv",
                    "input_data": {},
                }
            ],
        ],
        module="v6-glm-py",
    )


def get_mock_client_poisson():
    return get_mock_client("poisson")


def get_mock_client_binomial():
    return get_mock_client("binomial")


def get_mock_client_gaussian():
    return get_mock_client("gaussian")


def get_mock_client_survival():
    return get_mock_client("survival")


def get_org_ids(client: MockAlgorithmClient):
    organizations = client.organization.list()
    return [organization["id"] for organization in organizations]


# client = get_mock_client_poisson()
# org_ids = get_org_ids(client)
# central_task = client.task.create(
#     input_={
#         "method": "glm",
#         "kwargs": {
#             "outcome_variable": "num_awards",
#             "predictor_variables": ["prog", "math"],
#             "family": "poisson",
#             "category_reference_values": {"prog": "General"},
#         },
#     },
#     organizations=[org_ids[0]],
# )
# results = client.wait_for_results(central_task.get("id"))
# print(results)


def test_central_1_iteration():
    client = get_mock_client_poisson()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "outcome_variable": "num_awards",
                "predictor_variables": ["prog", "math"],
                # "survival_sensor_column": "some_value",
                "family": "poisson",
                # "tolerance_level": "some_value",
                "max_iterations": 1,
                # "organizations_to_include": "some_value",
            },
        },
        organizations=[org_ids[0]],
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
    client = get_mock_client_poisson()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "outcome_variable": "num_awards",
                "predictor_variables": ["prog", "math"],
                "family": "poisson",
                "category_reference_values": {"prog": "General"},
            },
        },
        organizations=[org_ids[0]],
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    details = results[0]["details"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -5.24712)
    assert_almost_equal(coefficients["beta"]["prog[T.Academic]"], 1.08385)
    assert_almost_equal(coefficients["beta"]["prog[T.Vocational]"], 0.36980)
    assert_almost_equal(coefficients["beta"]["math"], 0.07015)
    assert_almost_equal(coefficients["p_value"]["Intercept"], 1.6013e-15)
    assert_almost_equal(coefficients["p_value"]["prog[T.Academic]"], 0.00248)
    assert_almost_equal(coefficients["p_value"]["prog[T.Vocational]"], 0.4017)
    assert_almost_equal(coefficients["p_value"]["math"], 3.62500e-11)
    assert_almost_equal(coefficients["std_error"]["Intercept"], 0.6584)
    assert_almost_equal(coefficients["std_error"]["prog[T.Academic]"], 0.3582)
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


def test_central_until_convergence_gaussian(assert_almost_equal: callable):
    client = get_mock_client_gaussian()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "outcome_variable": "Volume",
                "predictor_variables": ["Girth"],
                "family": "gaussian",
            },
        },
        organizations=[org_ids[0]],
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
    client = get_mock_client_binomial()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "outcome_variable": "admit",
                "predictor_variables": ["gre", "gpa", "rank"],
                "family": "binomial",
                "categorical_predictors": ["rank"],
                "category_reference_values": {"rank": 1},
            },
        },
        organizations=[org_ids[0]],
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


def test_central_binomial_with_formula(assert_almost_equal: callable):
    client = get_mock_client_binomial()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "formula": "admit ~ gre + gpa + I(rank == 3)",
                "family": "binomial",
                "categorical_predictors": ["rank"],
                "category_reference_values": {"rank": 1},
            },
        },
        organizations=[org_ids[0]],
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
    client = get_mock_client_binomial()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "formula": "admit ~ rank + I(log(gre) + gpa)",
                "family": "binomial",
                "categorical_predictors": ["rank"],
                "category_reference_values": {"rank": 1},
            },
        },
        organizations=[org_ids[0]],
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
    client = get_mock_client_survival()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "outcome_variable": "status",
                "predictor_variables": ["sex", "age"],
                "family": "survival",
                "survival_sensor_column": "survival_sensor_column",
                "categorical_predictors": ["sex"],
                "category_reference_values": {"sex": 0},
            },
        },
        organizations=[org_ids[0]],
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
    client = get_mock_client_poisson()
    org_ids = get_org_ids(client)
    input_ = {
        "method": "glm",
        "kwargs": {
            "outcome_variable": "num_awards",
            "predictor_variables": ["prog", "math"],
            "family": "poisson",
            "category_reference_values": {"prog": "General"},
        },
    }
    # Test with non-existing family
    with pytest.raises(
        UserInputError,
        match="Family non-existing-family not supported. Please provide one of the "
        "supported families: poisson, binomial, gaussian, survival",
    ):
        wrong_input = deepcopy(input_)
        wrong_input["kwargs"]["family"] = "non-existing-family"
        client.task.create(
            input_=wrong_input,
            organizations=[org_ids[0]],
        )

    # Test without providing formula or outcome and predictor variables
    with pytest.raises(
        UserInputError,
        match="Either formula or outcome and predictor variables should be provided. "
        "Neither is provided.",
    ):
        wrong_input = deepcopy(input_)
        wrong_input["kwargs"].pop("predictor_variables")
        wrong_input["kwargs"].pop("outcome_variable")
        client.task.create(
            input_=wrong_input,
            organizations=[org_ids[0]],
        )

    # test if env var is set that only certain columns are allowed
    os.environ["GLM_ALLOWED_COLUMNS"] = "bla,bla2"
    with pytest.raises(NodePermissionException):
        client.task.create(
            input_=input_,
            organizations=[org_ids[0]],
        )
    # check that it works if only used columns are allowed
    os.environ["GLM_ALLOWED_COLUMNS"] = "prog,math,num_awards"
    client.task.create(
        input_=input_,
        organizations=[org_ids[0]],
    )
    del os.environ["GLM_ALLOWED_COLUMNS"]

    # test if env var is set that certain columns are not allowed
    os.environ["GLM_DISALLOWED_COLUMNS"] = "prog"
    with pytest.raises(NodePermissionException):
        client.task.create(
            input_=input_,
            organizations=[org_ids[0]],
        )
    # check that it works if only non-used columns are disallowed
    os.environ["GLM_DISALLOWED_COLUMNS"] = "bla,bla2"
    client.task.create(
        input_=input_,
        organizations=[org_ids[0]],
    )
    del os.environ["GLM_DISALLOWED_COLUMNS"]

    # check that if family is survival, there is an error if survival_sensor_column is
    # not provided
    with pytest.raises(UserInputError):
        input_survival = deepcopy(input_)
        input_survival["kwargs"]["family"] = "survival"
        client.task.create(
            input_=input_survival,
            organizations=[org_ids[0]],
        )

    # test with very little data for one organization
    first_party_data = deepcopy(client.datasets_per_org[org_ids[0]])
    client.datasets_per_org[org_ids[0]][0] = first_party_data[0].head(3)
    with pytest.raises(PrivacyThresholdViolation):
        client.task.create(
            input_=input_,
            organizations=[org_ids[0]],
        )

    # test if one of the used columns contains lots of null data
    client.datasets_per_org[org_ids[0]][0] = deepcopy(first_party_data[0])
    client.datasets_per_org[org_ids[0]][0]["prog"] = None
    with pytest.raises(PrivacyThresholdViolation):
        client.task.create(
            input_=input_,
            organizations=[org_ids[0]],
        )

    # test that using too many variables relative to observations is not allowed
    client.datasets_per_org[org_ids[0]][0] = deepcopy(first_party_data[0])
    for col in range(20):
        client.datasets_per_org[org_ids[0]][0][f"col_{col}"] = 1
    with pytest.raises(PrivacyThresholdViolation):
        input_extra_vars = deepcopy(input_)
        input_extra_vars["kwargs"]["predictor_variables"].extend(
            [f"col_{col}" for col in range(20)]
        )
        client.task.create(
            input_=input_extra_vars,
            organizations=[org_ids[0]],
        )

    # test that running GLM on two organizations is not allowed
    client = MockAlgorithmClient(
        datasets=[
            # Data for first organization
            [
                {
                    "database": current_path / "poisson" / "a.csv",
                    "db_type": "csv",
                    "input_data": {},
                }
            ],
            # Data for second organization
            [
                {
                    "database": current_path / "poisson" / "b.csv",
                    "db_type": "csv",
                    "input_data": {},
                }
            ],
        ],
        module="v6-glm-py",
    )
    org_ids = get_org_ids(client)
    with pytest.raises(
        UserInputError,
    ):
        client.task.create(
            input_=input_,
            organizations=[org_ids[0]],
        )
