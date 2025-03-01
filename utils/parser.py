from argparse import Namespace, ArgumentParser

class Parser:
    def __init__(self):
        self.description = self.set_description()
        self.parser = ArgumentParser(description=self.description)

    def set_description(self):
        self.description = "Forneça o caminho de duas imagens, para treino e consulta"

    def get_file(self) -> Namespace:
        self.parser.add_argument('train_image', type=str, help="Caminho para a imagem de treino")
        self.parser.add_argument('query_image', type=str, help='Caminho para a imagem de consulta')
        
        return self.parser.parse_args()
