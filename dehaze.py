import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img, title='Image', cmap=None):
    '''
    inputs:
        img: the input image
        title: the title for the displayed image
        cmap: GRAY or Color
    outputs:
        None
    '''
    plt.figure()
    plt.title(title)
    try:
        if cmap:
            # plt.imshow(img.astype(np.uint8), cmap=cmap)
            plt.imshow(img, cmap=cmap)
        else:
            # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
    except:
        if cmap:
            plt.imshow(img.astype(np.uint8), cmap=cmap)
            # plt.imshow(img, cmap=cmap)
        else:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
    plt.axis("off")
    plt.show()


def DarkChannelPrior(img, patchSize):
    '''
    inputs:
        img: the input image
        patchSize: integer, represents size of area (patchSize, patchSize)
    outputs:
        darkChannel: the dark channel of the input image
    '''
    width, height = img.shape[:2]
    # print('img.shape: ', img.shape)

    darkChannel = np.zeros((width, height), dtype=np.float32)

    pad = int(patchSize / 2)
    # pad_size = (vert_start, vert_end), (horz_start, horz_end), layer)
    pading_size = ((pad, pad), (pad, pad), (0, 0))
    padded_img = np.pad(img, pading_size, mode='edge')
    # display_img(padded_img)
    # print('padded_img.shape: ', padded_img.shape)

    for i in range(width):
        for j in range(height):
            # REF: eq 5 pg 3/13
            minPixel = np.min(padded_img[i:i + patchSize, j:j + patchSize, :])
            darkChannel[i, j] = minPixel
    # print('darkChannel.shape: ', darkChannel.shape)

    return(darkChannel)


def EstimateTheAtmosphericLight(img, darkChannel, topPercent):
    '''
    inputs:
        img: the input image
        darkChannel: the dark channel of the input image
        topPercent: integer, top percent of pixel lightness to use
    outputs:
        atmosphericLight: value of topPercent of pixels
    '''
    width, height = img.shape[:2]

    flat_img = img.reshape(width*height, 3)
    flat_darkChannel = darkChannel.reshape(-1)

    top_Percent = int(width * height * topPercent)

    sortedIndex = (-flat_darkChannel).argsort()
    top_Percent_Index = sortedIndex[:top_Percent]

    B_max, G_max, R_max = 0, 0, 0
    atmosphericLight = np.array([0, 0, 0])
    for index in top_Percent_Index:
        B, G, R = flat_img[index]
        if B > B_max:
            atmosphericLight[0] = B
            B_max = B
        if G > G_max:
            atmosphericLight[1] = G
            G_max = G
        if R > R_max:
            atmosphericLight[2] = R
            R_max = R

    return(atmosphericLight)


def EstimatingTheTransmission(img, atmosphericLight, w, patchSize):
    '''
    inputs:
        img: the input image
        atmosphericLight: intensity of topPercent of pixels
        w: number between 0 and 1, represents amount of haze to leave
        patchSize: integer, represents size of area (patchSize, patchSize)
    outputs:
        transmissionMap: the transmission map of the input image
    '''
    # eq. 12
    I_y = img / atmosphericLight
    darkChannel = DarkChannelPrior(I_y, patchSize)
    transmissionMap = 1 - w * darkChannel
    transmissionMap = transmissionMap.astype(np.float32)

    return(transmissionMap)


def SoftMatting(img, transmissionMap, radius, eps):
    '''
    inputs:
        img: the input image
        transmissionMap: the transmission map of the input image
        radius: radius used in the guidedFilter
        eps: regularization term for the guidedFilter
    outputs:
        softMatMap: the soft mat transmission map
    '''
    # REF: http://kaiminghe.com/publications/pami12guidedfilter.pdf
    softMatMap = cv2.ximgproc.guidedFilter(img, transmissionMap, radius, eps)
    return(softMatMap)


def RecoveringTheSceneRadiance(img, atmosphericLight, softMat, t0):
    '''
    inputs:
        img: the input image
        atmosphericLight: intensity of topPercent of pixels
        softMat: the soft mat transmission map
        t0: the lower bound
    outputs:
        radianceMap: the radiance map of the input image
    '''
    width, height = softMat.shape
    colorChannel = img.shape[2]

    for i in range(width):
        for j in range(height):
            if softMat[i, j] > 1:
                softMat[i, j] = 1
            elif softMat[i, j] < t0:
                softMat[i, j] = t0

    softMat_color = np.zeros_like(img, dtype=np.float32)
    for color in range(colorChannel):
        softMat_color[:, :, color] = softMat

    radiance_img = (img - atmosphericLight) / softMat_color + atmosphericLight

    for i in range(width):
        for j in range(height):
            for color in range(colorChannel):
                if radiance_img[i, j, color] > 255:
                    radiance_img[i, j, color] = 255
                elif radiance_img[i, j, color] < 0:
                    radiance_img[i, j, color] = 0

    radianceMap = np.array(radiance_img, np.int32)

    return(radianceMap)


# color enhancement options
def YUV_image(img):
    image_yuv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    # image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return(image)


def LAB_image(img):
    image_lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    image_lab[..., 0] = cv2.equalizeHist(image_lab[..., 0])
    image = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)
    return(image)

if __name__ == "__main__":
    # ----- Input Variables ------
    # images from paper
    # path = 'input_img/pumpkins.jpg'
    # path = 'input_img/tiananmen1.png'

    # images produced
    path = 'images/2.jpg'
    # path = 'input_img/12.jpg'
    # path = 'input_img/14.jpg'
    # path = 'input_img/20.jpg'
    # path = 'input_img/31.jpg'

    img = cv2.imread(path)
    # DarkChannelPrior and transmissionMap
    patchSize = 15
    # atmosphericLight
    topPercent = 0.001
    # transmissionMap
    w = 0.95
    # SoftMatting
    radius = 20
    eps = 10e-3
    # RecoveringTheSceneRadiance
    t0 = 0.1
    # ----- End of Input Variables ------
    print('DarkChannelProir started')
    darkChannel = DarkChannelPrior(img, patchSize)
    print('DarkChannelProir completed')

    print('EstimateTheAtmosphericLight started')
    atmosphericLight = EstimateTheAtmosphericLight(img, darkChannel, topPercent)
    print('EstimateTheAtmosphericLight completed')

    print('EstimatingTheTransmission started')
    transmissionMap = EstimatingTheTransmission(img, atmosphericLight, w, patchSize)
    print('EstimatingTheTransmission completed')

    print('SoftMatting started')
    softMat = SoftMatting(img, transmissionMap, radius, eps)
    print('SoftMatting completed')

    print('RecoveringTheSceneRadiance started')
    radianceMap = RecoveringTheSceneRadiance(img, atmosphericLight, softMat, t0)
    print('RecoveringTheSceneRadiance completed')

    YUV_img = YUV_image(radianceMap)
    LAB_img = LAB_image(radianceMap)

    # display images
    display_img(darkChannel, title='darkChannel', cmap='gray')
    print('atmosphericLight: \n', atmosphericLight)
    display_img(transmissionMap, title='transmissionMap', cmap='gray')
    display_img(softMat, title='softMat')
    display_img(radianceMap, title='radianceMap')
    display_img(YUV_img, title='YUV_img')
    display_img(LAB_img, title='LAB_img')

    # save images
    # cv2.imwrite('darkChannel.jpg', darkChannel)
    # cv2.imwrite('transmissionMap.jpg', transmissionMap*255)
    # cv2.imwrite('softMat.jpg', softMat*255)
    # cv2.imwrite('radianceMap.jpg', radianceMap)
    # cv2.imwrite('YUV_img.jpg', YUV_img)
    # cv2.imwrite('LAB_img.jpg', LAB_img)

    # res = path[:-4] + '_res' + path[-4:]
    # res_img = cv2.imread(res)
    # display_img(res_img, title='res_img')
    print('Haze removal completed')
