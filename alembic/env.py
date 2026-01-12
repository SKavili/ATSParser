"""Alembic environment configuration."""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine
from alembic import context

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.database.connection import Base

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override sqlalchemy.url with our settings
# Use a synchronous driver (pymysql) for Alembic, since it runs in sync mode.
# Escape % signs to prevent ConfigParser interpolation issues
mysql_url = settings.mysql_url.replace("+aiomysql", "+pymysql").replace("%", "%%")
config.set_main_option("sqlalchemy.url", mysql_url)

# Import all models so that Base.metadata is fully populated.
# IMPORTANT: This must import the module that defines *all* ORM models
# that should be managed by Alembic. If you add new models/modules,
# ensure they are imported here or from app.database.models.
from app.database import models  # noqa: F401

# Use the same Base.metadata that your ORM models are attached to.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

