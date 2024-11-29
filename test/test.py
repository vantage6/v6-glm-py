"""
Run this script to test your algorithm locally (without building a Docker
image) using the mock client.

Run as:

    python test.py

Make sure to do so in an environment where `vantage6-algorithm-tools` is
installed. This can be done by running:

    pip install vantage6-algorithm-tools
"""

from pathlib import Path
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

# # Run the central method on 1 node and get the results
# central_task = client.task.create(
#     input_={
#         "method":"glm",
#         "kwargs": {
#             # TODO add sensible values
#             "outcome_variable": "some_value",
#             "predictor_variables": "some_value",
#             "dstar": "some_value",
#             "types": "some_value",
#             "family": "some_value",
#             "tolerance_level": "some_value",
#             "max_iterations": "some_value",
#             "organizations_to_include": "some_value",

#         }
#     },
#     organizations=[org_ids[0]],
# )
# results = client.wait_for_results(central_task.get("id"))
# print(results)

# Run the partial method for all organizations
task = client.task.create(
    input_={
        "method": "compute_partial_betas",
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
print(results)
