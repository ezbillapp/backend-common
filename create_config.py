import json
import sys
from typing import Dict

import boto3

CHALICE_CONGIFIG_TEMPLATE_FILE = ".chalice/template.config.json"
CHALICE_CONGIFIG_FILE = ".chalice/config.json"


from config import ENV_VARS, PROJECT_NAME, REGION_NAME


def _get_key(stage, var):
    """Gets the key of the environment variable"""
    return f"/{PROJECT_NAME}/{stage}/{var}"


def _get_ssm_value(client, stage, var):
    """Gets the value of the environment variable"""
    return client.get_parameter(Name=_get_key(stage, var), WithDecryption=True)["Parameter"][
        "Value"
    ]


def gen_env_vars_dict(stage: str) -> Dict[str, str]:
    """Generates a dictionary of environment variables"""
    client = boto3.client("ssm", region_name=REGION_NAME)
    return {var: _get_ssm_value(client, stage, var) for var in ENV_VARS}


def save_env_vars_dict(template_path, config_path, env_vars):
    """Writes all environment variables on build before deployment"""
    with open(template_path, "r") as f:
        content = json.loads(f.read())
        content["stages"]["dev"]["environment_variables"] = env_vars
        json.dump(content, open(config_path, "w"))


def main():
    """Writes all environment variables on build before deployment"""
    stage = str(sys.argv[1])

    env_vars = gen_env_vars_dict(stage)
    env_vars["STAGE"] = stage
    save_env_vars_dict(CHALICE_CONGIFIG_TEMPLATE_FILE, CHALICE_CONGIFIG_FILE, env_vars)


if __name__ == "__main__":
    main()
