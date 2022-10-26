import os
import re
import sys
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool

parent_dir = os.path.abspath(os.getcwd())
sys.path.append(parent_dir)
from chalicelib.schema import connection_uri  # pylint: disable=wrong-import-position
from chalicelib.schema.models import Base  # NOQA # pylint: disable=wrong-import-position

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config: Any = context.config  # pylint: disable=no-member


# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.
# Database configuration from env


config.set_main_option(
    "sqlalchemy.url",
    connection_uri,
)


def get_excludes_from_config(config_, type_="tables"):
    excludes = config_.get(type_, None)
    if excludes is not None:
        excludes = excludes.split(",")
    return excludes


excluded_tables = get_excludes_from_config(config.get_section("exclude"), "tables")
excluded_indices = get_excludes_from_config(config.get_section("exclude"), "indices")


def include_object(obj, name, type_, reflected, compare_to):
    if type_ == "table":
        for table_pat in excluded_tables:
            if re.match(table_pat, name):
                return False
    elif type_ == "index":
        for index_pat in excluded_indices:
            if re.match(index_pat, name):
                return False
    return True


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
