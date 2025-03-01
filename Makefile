# Variáveis
PYTHON = python3
SCRIPT = main.py

# Alvo principal
run:
	$(PYTHON) $(SCRIPT) path1 path2

# Alvo para rodar com um caminho customizado
run_custom:
	$(PYTHON) $(SCRIPT) $(path1) $(path2)

# Alvo para rodar o script com argumentos dinamicamente
run_with_args:
	$(PYTHON) $(SCRIPT) $(arg1) $(arg2)
