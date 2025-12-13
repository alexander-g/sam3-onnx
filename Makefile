ifneq ($(OS),Windows_NT)
  # On Unix-based systems, use ANSI codes
  BLUE = \033[36m
  BOLD_BLUE = \033[1;36m
  RED = \033[31m
  YELLOW = \033[33m
  BOLD = \033[1m
  NC = \033[0m
endif

escape = $(subst $$,\$$,$(subst ",\",$(subst ',\',$(1))))

define exec
	@echo "$(BOLD_BLUE)$(call escape,$(1))$(NC)"
	@$(1)
endef

help:
	@echo "$(BLUE)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-].+:.*?# .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?# "}; \
		{printf "  $(YELLOW)%-23s$(NC) %s\n", $$1, $$2}'

clone:
	$(call exec,test -d sam3 || git clone https://github.com/wkentaro/sam3.git -b onnx)
	$(call exec,cd sam3 && git pull origin onnx)

build: clone  # build
	$(call exec,uv sync)

lint:  # lint
	$(call exec,ruff format --check)
	$(call exec,ruff check)
	$(call exec,ty check)

format:  # format
	$(call exec,ruff format)
	$(call exec,ruff check --fix)
