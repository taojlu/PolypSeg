import matplotlib.pyplot as plt


def img_mask_plot(image, mask):
    fix, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(image)
    ax[0].set_title("raw image")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("raw image")
    plt.show()
