import json
import logging
import sys
from typing import Dict

import boto3

_logger = logging.getLogger(__name__)

CHALICE_CONGIFIG_TEMPLATE_FILE = ".chalice/template.config.json"
CHALICE_CONGIFIG_FILE = ".chalice/config.json"


from config import ENV_VARS, PROJECT_NAME, REGION_NAME


def _get_key(stage, var):
    """Gets the key of the environment variable"""
    return f"/{PROJECT_NAME}/{stage}/{var}"


def _get_ssm_value(client, stage, var):
    """Gets the value of the environment variable"""
    try:
        return client.get_parameter(Name=_get_key(stage, var), WithDecryption=True)["Parameter"][
            "Value"
        ]
    except client.exceptions.ParameterNotFound:
        _logger.error("Environment variable %s not found", var)
        return None


def gen_env_vars_dict(stage: str) -> Dict[str, str]:
    """Generates a dictionary of environment variables"""
    client = boto3.client("ssm", region_name=REGION_NAME)
    res = {}
    for var in ENV_VARS:
        val = _get_ssm_value(client, stage, var)
        if val is not None:
            res[var] = val
    return res


def save_env_vars_dict(template_path, config_path, stage, env_vars, subnets, security_groups):
    """Writes all environment variables on build before deployment"""
    with open(template_path, "r") as f:
        content = json.loads(f.read() or "{}")
        if "stages" not in content:
            content["stages"] = {}
        if stage not in content["stages"]:
            content["stages"][stage] = {}
        content["stages"][stage]["environment_variables"] = env_vars
        content["stages"][stage]["subnet_ids"] = subnets
        content["stages"][stage]["security_group_ids"] = security_groups
        json.dump(content, open(config_path, "w"))


def main():
    """Writes all environment variables on build before deployment"""
    stage = str(sys.argv[1])

    env_vars = gen_env_vars_dict(stage)
    subnets = [env_vars.pop(env_var) for env_var in ["SUBNET_1", "SUBNET_2", "SUBNET_3"]]
    security_groups = [env_vars.pop(env_var) for env_var in ["SECURITY_GROUP"]]
    env_vars["STAGE"] = stage
    save_env_vars_dict(
        CHALICE_CONGIFIG_TEMPLATE_FILE,
        CHALICE_CONGIFIG_FILE,
        stage,
        env_vars,
        subnets,
        security_groups,
    )


if __name__ == "__main__":
    main()
