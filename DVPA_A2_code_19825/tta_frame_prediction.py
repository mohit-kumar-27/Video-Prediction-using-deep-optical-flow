import glob
import cv2
import regex as re
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import scipy.ndimage
from flownet2.Flownet2ControllerTTA import FlowControllerTTA


def deep_optical_flow(model_path, firstImage, secondImage, lr, num_iter, image_ind, dataset):
    """
    FlowNet2 Deep Optical flow estimation between firstImage and secondImage
    :param model_path: Path to pretrained optical flow model
    :param firstImage: First image
    :param secondImage: Second Image
    :param lr: Learning rate of fine tuning
    :param num_iter: Number of iterations of fine tuning
    :param image_ind: Current image index
    :param dataset: Dataset name
    :return:
    """

    # Calculate flow
    flow_controller = FlowControllerTTA(model_path)
    flow_controller.test_time_adaptation(firstImage, secondImage, num_iter, lr)
    optical_flow = flow_controller.predict(firstImage, secondImage)

    # Plot, visualize and save the optical flow
    flow_image = flow_controller.convert_flow_to_image(optical_flow)
    if not os.path.exists(f"./results/tta/optical_flow/{dataset}/"):
        os.makedirs(f"./results/tta/optical_flow/{dataset}/")
    cv2.imwrite(f'./results/tta/optical_flow/{dataset}/flow_map_{image_ind}.png', flow_image)
    # cv2.imshow("Flow image", flow_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # Kernels for finding gradients Ix, Iy, It
    kernel_x = np.array([[-1, 1]])
    kernel_y = np.array([[-1], [1]])
    kernel_t = np.array([[1]])

    # kernel_x = np.array([[-1., 1.], [-1., 1.]]) / 4
    # kernel_y = np.array([[-1., -1.], [1., 1.]]) / 4
    # kernel_t = np.array([[1., 1.], [1., 1.]]) / 4

    firstImage_grayscale = firstImage[:, :, 0]
    secondImage_grayscale = secondImage[:, :, 0]
    Ix = scipy.ndimage.convolve(input=firstImage_grayscale, weights=kernel_x, mode="nearest")
    Iy = scipy.ndimage.convolve(input=firstImage_grayscale, weights=kernel_y, mode="nearest")
    It = scipy.ndimage.convolve(
        input=secondImage_grayscale, weights=kernel_t, mode="nearest"
    ) + scipy.ndimage.convolve(input=firstImage_grayscale, weights=-kernel_t, mode="nearest")

    flow = [optical_flow[:, :, 0], optical_flow[:, :, 1]]
    I = [Ix, Iy, It]

    return flow, I

LR = 1e-4
NUM_ITER = 1

def sphere_frame_prediction(model_path="./flownet2/pretrained_models/FlowNet2_checkpoint.pth.tar"):
    """
    Sphere dataset interpolation of Frame N+1 from Frame N and Frame N+2
    :param model_path: Path to pretrained optical flow model
    :return: None
    """

    images = glob.glob("./input/sphere/*.ppm")
    images.sort(key=lambda f: int(re.sub("\D", "", f)))

    for ind in range(0, len(images) - 2, 1):
        firstImage = cv2.imread(images[ind])
        secondImage = cv2.imread(images[ind + 1])
        thirdImage = cv2.imread(images[ind + 2],flags=cv2.IMREAD_GRAYSCALE)

        forward_flow, If = deep_optical_flow(model_path, firstImage, secondImage, LR, NUM_ITER, ind, "sphere")
        img = predict_frame(secondImage, forward_flow, If, ind, "sphere")
        compare_images(img, thirdImage)

def corridor_frame_prediction(model_path="./flownet2/pretrained_models/FlowNet2_checkpoint.pth.tar"):
    """
    Corridor dataset interpolation of Frame N+1 from Frame N and Frame N+2
    :param model_path: Path to pretrained optical flow model
    :return: None
    """

    images = glob.glob("./input/corridor/*.pgm")
    images.sort(key=lambda f: int(re.sub("\D", "", f)))

    for ind in range(0, len(images) - 2, 1):
        firstImage = cv2.imread(images[ind])
        secondImage = cv2.imread(images[ind + 1])
        thirdImage = cv2.imread(images[ind + 2],flags=cv2.IMREAD_GRAYSCALE)

        forward_flow, If = deep_optical_flow(model_path, firstImage, secondImage, LR, NUM_ITER, ind, "corridor")

        img = predict_frame(secondImage, forward_flow, If, ind, "corridor")
        compare_images(img, thirdImage)

def predict_frame(secondImage, forward_flow, If, image_ind, dataset):
    """
    Future frame prediction
    :param secondImage: Second image (Frame N+1)
    :param forward_flow: Optical flow from Frame N to Frame N+1
    :param If: Forward gradients [Ix, Iy, It]
    :param image_ind: Current image index
    :param dataset: Dataset name
    """

    uf, vf = forward_flow
    uf = -1*uf
    vf = -1*vf
    x, y = np.float32(np.meshgrid(np.arange(secondImage.shape[0]), np.arange(secondImage.shape[1])))
    x1, y1 = np.float32(x + uf), np.float32(y + vf)
    warped_image = cv2.remap(secondImage, x1, y1, interpolation=cv2.INTER_CUBIC)

    if not os.path.exists(f"./results/tta/predicted_frames/{dataset}/"):
        os.makedirs(f"./results/tta/predicted_frames/{dataset}/")
    cv2.imwrite(f"./results/tta/predicted_frames/{dataset}/forward_prediction_{image_ind+2}.png",
    warped_image,)

    return cv2.imread(f"./results/tta/predicted_frames/{dataset}/forward_prediction_{image_ind+2}.png",flags=cv2.IMREAD_GRAYSCALE)

# This function computes the mean square error between two images
def mse(imageA, imageB):

    err = np.sum((np.array(imageA, dtype=np.float32) - np.array(imageB, dtype=np.float32)) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

# This function compares two images in terms of mse and ssim and plots them
def compare_images(imageA, imageB):

    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA,cmap = plt.cm.gray)
    plt.axis("off")
    plt.title(f'MSE: {m:.2f}')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB,cmap = plt.cm.gray)
    plt.axis("off")
    plt.title(f"SSIM: {s:.2f}")
    plt.show()


corridor_frame_prediction(model_path="./flownet2/pretrained_models/FlowNet2_checkpoint.pth.tar")

sphere_frame_prediction(model_path="./flownet2/pretrained_models/FlowNet2_checkpoint.pth.tar")
