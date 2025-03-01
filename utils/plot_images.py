import matplotlib.pyplot as plt
import cv2

def plot_images(
        image_1: cv2.typing.MatLike,
        image_2: cv2.typing.MatLike,
        image_3: cv2.typing.MatLike,
        image_4: cv2.typing.MatLike,
    ) -> None:

    fig, _ = plt.subplots(2, 2, figsize=(10, 7))
    fig.add_subplot(2, 2, 1)
    plt.imshow(image_3, 'gray')
    fig.add_subplot(2, 2, 2)
    plt.title('Primeira imagem')
    plt.imshow(image_1, 'gray')
    fig.add_subplot(2, 2, 3)
    plt.title('Segunda imagem')
    plt.imshow(image_2, 'gray')
    fig.add_subplot(2, 2, 4)
    plt.title('Primeira imagem após transformação')
    plt.imshow(image_4, 'gray')

    plt.tight_layout()
    plt.show()
