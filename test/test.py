"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python test.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""

from pprint import pprint
from pathlib import Path
import numpy as np
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

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


def get_org_ids(client: MockAlgorithmClient):
    organizations = client.organization.list()
    return [organization["id"] for organization in organizations]


# client = get_mock_client_binomial()
# org_ids = get_org_ids(client)
# central_task = client.task.create(
#     input_={
#         "method": "glm",
#         "kwargs": {
#             "outcome_variable": "admit",
#             "predictor_variables": ["gre", "gpa", "rank"],
#             "family": "binomial",
#             "categorical_predictors": ["rank"],
#             "category_reference_values": {"rank": 1},
#         },
#     },
#     organizations=[org_ids[0]],
# )
# results = client.wait_for_results(central_task.get("id"))
# pprint(results)
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
pprint(results)


def test_central_1_iteration():
    client = get_mock_client_poisson()
    org_ids = get_org_ids(client)
    central_task = client.task.create(
        input_={
            "method": "glm",
            "kwargs": {
                "outcome_variable": "num_awards",
                "predictor_variables": ["prog", "math"],
                # "dstar": "some_value",
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
                # "category_reference_values": {"prog": "General"},
            },
        },
        organizations=[org_ids[0]],
    )
    results = client.wait_for_results(central_task.get("id"))

    coefficients = results[0]["coefficients"]
    assert_almost_equal(coefficients["beta"]["Intercept"], -36.94346)
    assert_almost_equal(coefficients["beta"]["Girth"], 5.0658564)
    assert_almost_equal(coefficients["p_value"]["Intercept"], 7.621448e-12)
    assert_almost_equal(coefficients["p_value"]["Girth"], 8.64433e-19)
    assert_almost_equal(coefficients["std_error"]["Intercept"], 3.36514)
    assert_almost_equal(coefficients["std_error"]["Girth"], 0.247376)
    assert_almost_equal(coefficients["z_value"]["Intercept"], -10.97827)
    assert_almost_equal(coefficients["z_value"]["Girth"], 20.47828)
    details = results[0]["details"]
    assert details["converged"] == True
    assert_almost_equal(details["deviance"], 524.3025)
    assert_almost_equal(details["dispersion"], 18.079397)
    assert details["is_dispersion_estimated"] == True
    assert details["iterations"] == 2
    assert_almost_equal(details["null_deviance"], 8106.08387)
    assert details["num_observations"] == 31
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
