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

## Mock client
client = MockAlgorithmClient(
    datasets=[
        # Data for first organization
        [{"database": current_path / "a.csv", "db_type": "csv", "input_data": {}}],
        # Data for second organization
        [{"database": current_path / "b.csv", "db_type": "csv", "input_data": {}}],
        # Data for third organization
        [{"database": current_path / "c.csv", "db_type": "csv", "input_data": {}}],
    ],
    module="v6-glm-py",
)

# # list mock organizations
organizations = client.organization.list()
# print(organizations)
org_ids = [organization["id"] for organization in organizations]

# # Run the central method **for 1 iteration**
# central_task = client.task.create(
#     input_={
#         "method": "glm",
#         "kwargs": {
#             "outcome_variable": "num_awards",
#             "predictor_variables": ["prog", "math"],
#             # "dstar": "some_value",
#             # "types": "some_value",
#             "family": "poisson",
#             # "tolerance_level": "some_value",
#             "max_iterations": 1,
#             # "organizations_to_include": "some_value",
#         },
#     },
#     organizations=[org_ids[0]],
# )
# results = client.wait_for_results(central_task.get("id"))
# pprint(results)

# coefficients = results[0]["coefficients"]
# details = results[0]["details"]
# np.testing.assert_almost_equal(coefficients["beta"]["Intercept"], -2.6049735531)
# np.testing.assert_almost_equal(coefficients["beta"]["prog[T.General]"], -0.96139767)
# np.testing.assert_almost_equal(
#     coefficients["beta"]["prog[T.Vocational]"], -0.63360800191
# )
# np.testing.assert_almost_equal(coefficients["beta"]["math"], 0.0521480764677)
# np.testing.assert_almost_equal(
#     coefficients["p_value"]["Intercept"], 3.5776068337621e-05
# )
# np.testing.assert_almost_equal(
#     coefficients["p_value"]["prog[T.General]"], 0.00161174281
# )
# np.testing.assert_almost_equal(
#     coefficients["p_value"]["prog[T.Vocational]"], 0.02498002174875599
# )
# np.testing.assert_almost_equal(coefficients["p_value"]["math"], 2.640565207667894e-07)
# np.testing.assert_almost_equal(coefficients["std_error"]["Intercept"], 0.63025719201367)
# np.testing.assert_almost_equal(
#     coefficients["std_error"]["prog[T.General]"], 0.30484045913928837
# )
# np.testing.assert_almost_equal(
#     coefficients["std_error"]["prog[T.Vocational]"], 0.2826447521963971
# )
# np.testing.assert_almost_equal(coefficients["std_error"]["math"], 0.010130863564717027)
# np.testing.assert_almost_equal(coefficients["z_value"]["Intercept"], -4.1331913163552)
# np.testing.assert_almost_equal(coefficients["z_value"]["prog[T.General]"], -3.153773216)
# np.testing.assert_almost_equal(
#     coefficients["z_value"]["prog[T.Vocational]"], -2.2417115371628973
# )
# np.testing.assert_almost_equal(coefficients["z_value"]["math"], 5.147446329196461)

# assert details["converged"] == False
# assert details["iterations"] == 1
# assert details["dispersion"] == 1
# assert details["is_dispersion_estimated"] == False
# np.testing.assert_almost_equal(details["deviance"], 231.74880221325174)
# np.testing.assert_almost_equal(details["null_deviance"], 287.67223445286476)
# assert details["num_observations"] == 200
# assert details["num_variables"] == 4

# Run the central method **for 1 iteration**
central_task = client.task.create(
    input_={
        "method": "glm",
        "kwargs": {
            "outcome_variable": "num_awards",
            "predictor_variables": ["prog", "math"],
            # "dstar": "some_value",
            # "types": "some_value",
            "family": "poisson",
            # "tolerance_level": "some_value",
            # "max_iterations": 3,
            # "organizations_to_include": "some_value",
        },
    },
    organizations=[org_ids[0]],
)
results = client.wait_for_results(central_task.get("id"))
pprint(results)
exit(1)  ## TODO remove to execute rest of the tests

coefficients = results[0]["coefficients"]
details = results[0]["details"]
np.testing.assert_almost_equal(coefficients["beta"]["Intercept"], -2.6049735531)
np.testing.assert_almost_equal(coefficients["beta"]["prog[T.General]"], -0.96139767)
np.testing.assert_almost_equal(
    coefficients["beta"]["prog[T.Vocational]"], -0.63360800191
)
np.testing.assert_almost_equal(coefficients["beta"]["math"], 0.0521480764677)
np.testing.assert_almost_equal(
    coefficients["p_value"]["Intercept"], 3.5776068337621e-05
)
np.testing.assert_almost_equal(
    coefficients["p_value"]["prog[T.General]"], 0.00161174281
)
np.testing.assert_almost_equal(
    coefficients["p_value"]["prog[T.Vocational]"], 0.02498002174875599
)
np.testing.assert_almost_equal(coefficients["p_value"]["math"], 2.640565207667894e-07)
np.testing.assert_almost_equal(coefficients["std_error"]["Intercept"], 0.63025719201367)
np.testing.assert_almost_equal(
    coefficients["std_error"]["prog[T.General]"], 0.30484045913928837
)
np.testing.assert_almost_equal(
    coefficients["std_error"]["prog[T.Vocational]"], 0.2826447521963971
)
np.testing.assert_almost_equal(coefficients["std_error"]["math"], 0.010130863564717027)
np.testing.assert_almost_equal(coefficients["z_value"]["Intercept"], -4.1331913163552)
np.testing.assert_almost_equal(coefficients["z_value"]["prog[T.General]"], -3.153773216)
np.testing.assert_almost_equal(
    coefficients["z_value"]["prog[T.Vocational]"], -2.2417115371628973
)
np.testing.assert_almost_equal(coefficients["z_value"]["math"], 5.147446329196461)

assert details["converged"] == False
assert details["iterations"] == 1
assert details["dispersion"] == 1
assert details["is_dispersion_estimated"] == False
np.testing.assert_almost_equal(details["deviance"], 231.74880221325174)
np.testing.assert_almost_equal(details["null_deviance"], 287.67223445286476)
assert details["num_observations"] == 200
assert details["num_variables"] == 4


# Run the partial method for all organizations
task = client.task.create(
    input_={
        "method": "compute_local_betas",
        "kwargs": {
            # TODO add sensible values
            "outcome_variable": "num_awards",
            "predictor_variables": ["prog", "math"],
            "family": "poisson",
            "is_first_iteration": True,
            "beta_coefficients": [],
            # "dstar": "some_value",
            # "types": "some_value",
            "weights": None,
        },
    },
    organizations=[org_ids[0]],
    # organizations=org_ids,
)
print(task)

# Get the results from the task
results = client.wait_for_results(task.get("id"))
# results = json.loads(results)
print(results)

results_node1 = results[0]
np.testing.assert_almost_equal(results_node1["XTX"]["Intercept"]["Intercept"], 22.6)
np.testing.assert_almost_equal(
    results_node1["XTX"]["Intercept"]["prog[T.General]"], 4.5
)
np.testing.assert_almost_equal(
    results_node1["XTX"]["Intercept"]["prog[T.Vocational]"], 8.1
)
np.testing.assert_almost_equal(results_node1["XTX"]["Intercept"]["math"], 1014.1)

np.testing.assert_almost_equal(
    results_node1["XTX"]["prog[T.General]"]["Intercept"], 4.5
)
np.testing.assert_almost_equal(
    results_node1["XTX"]["prog[T.General]"]["prog[T.General]"], 4.5
)
np.testing.assert_almost_equal(
    results_node1["XTX"]["prog[T.General]"]["prog[T.Vocational]"], 0.0
)
np.testing.assert_almost_equal(results_node1["XTX"]["prog[T.General]"]["math"], 198.3)

np.testing.assert_almost_equal(
    results_node1["XTX"]["prog[T.Vocational]"]["Intercept"], 8.1
)
np.testing.assert_almost_equal(
    results_node1["XTX"]["prog[T.Vocational]"]["prog[T.General]"], 0.0
)
np.testing.assert_almost_equal(
    results_node1["XTX"]["prog[T.Vocational]"]["prog[T.Vocational]"], 8.1
)
np.testing.assert_almost_equal(
    results_node1["XTX"]["prog[T.Vocational]"]["math"], 343.7
)

np.testing.assert_almost_equal(results_node1["XTX"]["math"]["Intercept"], 1014.1)
np.testing.assert_almost_equal(results_node1["XTX"]["math"]["prog[T.General]"], 198.3)
np.testing.assert_almost_equal(
    results_node1["XTX"]["math"]["prog[T.Vocational]"], 343.7
)
np.testing.assert_almost_equal(results_node1["XTX"]["math"]["math"], 46004.5)

np.testing.assert_almost_equal(
    results_node1["XTz"]["num_awards"]["Intercept"], -13.70316036674478
)
np.testing.assert_almost_equal(
    results_node1["XTz"]["num_awards"]["prog[T.General]"], -3.94857852
)
np.testing.assert_almost_equal(
    results_node1["XTz"]["num_awards"]["prog[T.Vocational]"], -8.56251525
)
np.testing.assert_almost_equal(results_node1["XTz"]["num_awards"]["math"], -575.214766)

np.testing.assert_almost_equal(results_node1["dispersion"], 5.321407624633432)
assert results_node1["num_observations"] == 66
assert results_node1["num_variables"] == 4
assert results_node1["weighted_sum_of_y"] == 16
assert results_node1["weights_sum"] == 66
