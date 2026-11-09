.PHONY: help setup lint format type test coverage validation data train train-all evaluation certify sanity compare report run diagram docker-build docker-up clean all changelog cut-release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies (with the validation group) and pre-commit hooks
	uv sync --dev --group validation
	uv run pre-commit install

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run tests
	uv run pytest

coverage: ## Tests with coverage gate
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

validation: ## Parity tests against torchattacks, SciPy and statsmodels
	uv run pytest -m validation -v

data: ## Download and verify CIFAR-10 (163 MiB, outside the repository)
	uv run python -m adversarial_robustness_lab data

train: ## Train one zoo model: make train NAME=wrn16_2_pgd
	uv run python benchmarks/train_zoo.py $(NAME) --resume

train-all: ## Train the whole zoo, one run at a time
	scripts/train_queue.sh small_cnn_standard wrn16_2_standard wrn16_2_pgd wrn16_2_fgsm_rs wrn16_2_trades wrn16_2_noise025 wrn16_2_noise050 rn18_standard

evaluation: ## Robust accuracy of every model on the full test set
	uv run python benchmarks/robustness.py

compare: ## Our attacks against torchattacks on the same models
	uv run python benchmarks/compare_torchattacks.py

certify: ## Randomized smoothing certification
	uv run python benchmarks/certify.py

sanity: ## Gradient-masking sanity checks
	uv run python benchmarks/sanity.py

report: ## Print the README tables from benchmarks/results
	uv run python benchmarks/report.py

run: ## Run the HTTP service (CHECKPOINT=path)
	uv run python -m adversarial_robustness_lab serve --checkpoint $(CHECKPOINT)

diagram: ## Render every diagram in docs/architecture to SVG (house theme, white background)
	@for f in docs/architecture/*.mmd; do \
		npx -y @mermaid-js/mermaid-cli -q -i "$$f" -o "$${f%.mmd}.svg" -c mermaid.config.json -b white || exit 1; \
	done

docker-build: ## Build the image
	docker build -t adversarial-robustness-lab .

docker-up: ## Run via docker compose (ARL_CHECKPOINT=path)
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)

changelog: ## Regenerate CHANGELOG.md from Conventional Commits (git-cliff)
	git cliff --output CHANGELOG.md && perl -0pi -e 's/\n+\z/\n/' CHANGELOG.md

cut-release: ## Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional)
	scripts/release.sh $(RELEASE_VERSION)
