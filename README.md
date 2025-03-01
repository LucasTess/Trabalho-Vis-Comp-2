# Trabalho-Vis-Comp-2
Segundo trabalho da disciplina de visão computacional na turma de 2024/02
Neste trabalho nós temos que criar um algoritmo para calcular a homografia entre duas imagens, usando:

- DLT Normalizado 
- RANSAC para eliminar os outliers
- IMPORTANTE: Não é preciso incluir otimização. Se o algortimo for bem feito, o resultado mesmo sem otimização já é excelente. 


# README

Este repositório contém um script Python (`main.py`) que requer dois caminhos como argumentos de entrada: `path1` e `path2`. Este README fornece informações sobre como rodar o script utilizando o `Makefile` para facilitar o processo.

## Requisitos

Antes de rodar o script, certifique-se de que você tem o Python 3 instalado em seu ambiente. Além disso, o script `main.py` depende de algumas bibliotecas externas, então você pode precisar instalar as dependências com:

```bash
pip install -r requirements.txt
```



## Estrutura de Diretórios

    .
    ├── Makefile
    ├── README.md
    ├── imgs
    │   ├── batman.jpg
    │   ├── comicsStarWars01.jpg
    │   ├── comicsStarWars02.jpg
    │   ├── dueto1.jpg
    │   ├── dueto2.jpg
    │   ├── elefanto1.jpg
    │   ├── elefanto2.jpg
    │   ├── livro_001.jpg
    │   ├── livro_002.jpg
    │   ├── outdoor_batman.jpg
    │   ├── outdoors01.jpg
    │   └── outdoors02.jpg
    ├── main.py
    ├── requirements.txt
    └── utils
        ├── aux_functions.py
        ├── dlt_functions.py
        ├── parser.py
        ├── plot_images.py
        └── ransac_functions.py

- `main.py`: O script Python principal.
- `Makefile`: Arquivo que contém os comandos de execução.

## Comandos Disponíveis no Makefile

### 1. `make run`

Este comando executa o script `main.py` com dois argumentos padrão (`path1` e `path2`). Você pode substituir `path1` e `path2` pelos caminhos reais dos arquivos ou diretórios que deseja passar como parâmetros.

```bash
make run_custom path1=<caminho1> path2=<caminho2>
```

### 2. Rode apenas o python no terminal sem makefile

```bash
python3 main.py <caminho1> <caminho2>
#exemplo
python main.py imgs/batman.jpg imgs/outdoor_batman.jpg
```
