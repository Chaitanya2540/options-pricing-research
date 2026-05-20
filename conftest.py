# Top-level pytest config. The pyproject.toml [tool.pytest.ini_options] block
# adds src/ to the python path, so tests can `from options.pricing.black_scholes import ...`
# without an editable install. This file exists so pytest discovers conftest
# fixtures from any depth.
