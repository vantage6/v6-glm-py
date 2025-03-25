# Constants for main function arguments
# # maximum number of iterations
DEFAULT_MAX_ITERATIONS = 25
# # tolerance level, used to check for convergence
DEFAULT_TOLERANCE = 1e-8

# names of environment variables that can be defined to override default values
# # minimum number of rows in the dataframe
ENVVAR_MINIMUM_ROWS = "GLM_MINIMUM_ROWS"
# # whitelist of columns allowed to be requested
ENVVAR_ALLOWED_COLUMNS = "GLM_ALLOWED_COLUMNS"
# # blacklist of columns not allowed to be requested
ENVVAR_DISALLOWED_COLUMNS = "GLM_DISALLOWED_COLUMNS"
# # minimum number of organizations to include in the analysis
ENVVAR_MINIMUM_ORGANIZATIONS = "GLM_MINIMUM_ORGANIZATIONS"
# # maximum percentage of number of variables relative to number of observations
# # allowed in the model. If the number of variables exceeds this percentage,
# # the model will not be run due to risks of data leakage through overfitting.
ENVVAR_MAX_PCT_PARAMS_OVER_OBS = "GLM_MAX_PCT_VARS_VS_OBS"

# default values for environment variables
# # minimum number of rows in the dataframe
DEFAULT_MINIMUM_ROWS = 10
# # minimum number of organizations to include in the analysis
DEFAULT_MINIMUM_ORGANIZATIONS = 3
# # maximum percentage of number of variables relative to number of observations
DEFAULT_MAX_PCT_PARAMS_VS_OBS = 10

# constants that cannot be reset by user
# # number of significant digits in the final output
SIGNIFICANT_DIGITS_FINAL_OUTPUT = 6
