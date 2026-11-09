.DEFAULT_GOAL := parse

DOCS_DIR   := docs
PARSED_DIR := parsed
ARGS       ?=
PDFS       := $(wildcard $(DOCS_DIR)/*.pdf) $(wildcard $(DOCS_DIR)/*.PDF)
JSONS      := $(patsubst $(DOCS_DIR)/%.pdf,$(PARSED_DIR)/%.json,$(filter %.pdf,$(PDFS))) \
              $(patsubst $(DOCS_DIR)/%.PDF,$(PARSED_DIR)/%.json,$(filter %.PDF,$(PDFS)))

.PHONY: parse all test clean help

## parse: Batch extract pending PDFs in docs/ into parsed/ (loads VLM once; pass ARGS="..." for flags)
parse: $(PARSED_DIR)
	uv run pdf-extraction-pipeline $(DOCS_DIR) --output-dir $(PARSED_DIR) $(ARGS)

## all: Target-based build using Make dependency tracking
all: $(JSONS)

## Pattern rule: Parse an individual PDF (e.g. 'make parsed/sample.json')
$(PARSED_DIR)/%.json: $(DOCS_DIR)/%.pdf | $(PARSED_DIR)
	uv run pdf-extraction-pipeline $< --output-dir $(PARSED_DIR) $(ARGS)

$(PARSED_DIR)/%.json: $(DOCS_DIR)/%.PDF | $(PARSED_DIR)
	uv run pdf-extraction-pipeline $< --output-dir $(PARSED_DIR) $(ARGS)

$(PARSED_DIR):
	mkdir -p $(PARSED_DIR)

$(DOCS_DIR):
	mkdir -p $(DOCS_DIR)

## test: Run test suite
test:
	uv run pytest

## clean: Remove parsed JSON outputs and test artifacts
clean:
	rm -rf $(PARSED_DIR) .pytest_cache

## help: Display available targets
help:
	@echo "Available make targets:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/## //'
