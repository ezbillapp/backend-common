setup-virtualenv:
	python3 -m pip install virtualenv
	virtualenv --python=python3.8 venv
	. venv/bin/activate # BUG Not work
	pip install --upgrade pip

install-dev-deps: setup-virtualenv
	pip install --upgrade pip
	pip install -r requirements-dev.txt

install: setup-virtualenv
	pip install -r requirements.txt

prepare-dev-env: install-dev-deps install
	pre-commit install

lint:
	pylint app.py
	pylint chalicelib
	pylint tests

typehint:
	mypy .

test:
	pytest --cov=chalicelib tests/
	coverage html

lint-format:
	black --exclude versions -l 100 --check .

review: lint-format lint typehint test

format:
	isort .
	black --exlude versions -l 100 .

clean: check
	find chalicelib -type f -name "*.pyc" -delete
	find tests -type f -name "*.pyc" -delete
	find chalicelib -type d -name "__pycache__" -delete
	find tests -type d -name "__pycache__" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache

check:
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

re-create-db: check
	PGPASSWORD=$$DB_PASSWORD dropdb -h $$DB_HOST -p $$DB_PORT -U $$DB_USER $$DB_NAME
	PGPASSWORD=$$DB_PASSWORD createdb -h $$DB_HOST -p $$DB_PORT -U $$DB_USER $$DB_NAME

	alembic -c chalicelib/alembic.ini upgrade head

migrate-db:
	alembic -c chalicelib/alembic.ini revision --autogenerate
	alembic -c chalicelib/alembic.ini upgrade head


run:
	chalice local --stage local
