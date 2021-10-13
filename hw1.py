"""
ITS8030: Homework 1

Please implement all functions below.

For submission create a project called its8030-2021-hw1 and put the solution in there.

Please note that NumPy arrays and PyTorch tensors share memory represantation, so when converting a
torch.Tensor type to numpy.ndarray, the underlying memory representation is not changed.

There is currently no existing way to support both at the same time. There is an open issue on
PyTorch project on the matter: https://github.com/pytorch/pytorch/issues/22402

There is also a deeper problem in Python with types. The type system is patchy and generics
has not been solved properly. The efforts to support some kind of generics for Numpy are
reflected here: https://github.com/numpy/numpy/issues/7370 and here: https://docs.google.com/document/d/1vpMse4c6DrWH5rq2tQSx3qwP_m_0lyn-Ij4WHqQqRHY
but there is currently no working solution. For Dicts and Lists there is support for generics in the 
typing module, but not for NumPy arrays.
"""
import cv2
import numpy as np
import util
import math

"""
Task 1: Convolution

Implement the function 

convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int, kernel_height : int, add : bool, in_place:bool) -> np.ndarray

to convolve an image with a kernel of size kernel_height*kernel_width.
Use zero-padding around the borders for simplicity (what other options would there be?).
Here:

    image is a 2D matrix of class double
    kernel is a 2D matrix with dimensions kernel_width and kernel_height
    kernel_width and kernel_height are the width and height of the kernel respectively

(Note: in the general case, they are not equal and may not be always odd, so you have to ensure that they are odd.)

    if add is true, then 128 is added to each pixel for the result to get rid of negatives.
    if in_place is False, then the output image should be a copy of the input image. The default is False,
    i.e. the operations are performed on the input image.

Write a general convolution function that can handle all possible cases as mentioned above.
You can get help from the convolution part of the function mean_blur_image (to be implemented in a lab)
to write this function.
"""
def convolution(image : np.ndarray, kernel : np.ndarray, kernel_width : int,
                kernel_height : int, add : bool, in_place : bool = False) -> np.ndarray :
    
    if(kernel_height % 2 == 0 or kernel_width % 2 == 0): 
        raise Exception("kernel's width and height must be odd.")
    
    if add == True:
        add_val = 128
    else:
        add_val = 0
    
    kernel = np.flipud(np.fliplr(kernel))  # flip kernel for convolution
    
    height = image.shape[0]
    width = image.shape[1] # get sizes of image
    channels = image.shape[2]

    
    # for padding, the offset for 1 side
    p_width = (kernel_width - 1) // 2
    p_height = (kernel_height - 1) // 2
    
    # create an array of zeroes with padding on each side of TYPE DOUBLE

    pad_img = np.zeros((height+2*p_height,width+2*p_width,channels),dtype=np.double)
    
    # copy image into the center of the new padded image
    
    pad_img[p_height:p_height+height,p_width:p_width+width,0:channels] = image
    
    if in_place == False:
        convoluted_img = np.zeros_like(image, dtype=np.double)
        for i in range(channels):
            for c in range(height):       # columns of img
                for r in range(width):    # rows of img
                    convoluted_img[c,r,i] = np.sum(pad_img[c:c+kernel_height,r:r+kernel_width,i]*kernel)
        return convoluted_img + add_val
    else:
        for i in range(channels):
            for c in range(height):       # columns of img
                for r in range(width):    # rows of img
                    image[c,r,i] = np.sum(pad_img[c:c+kernel_height,r:r+kernel_width,i]*kernel) + add_val

"""
Task 2: Gaussian blur

Implement the function

gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray 

to Gaussian blur an image. "sigma" is the standard deviation of the Gaussian.
Use the function mean_blur_image as a template, create a 2D Gaussian filter
as the kernel and call the convolution function of Task 1.
Normalize the created kernel using the function normalize_kernel() (to
be implemented in a lab) before convolution. For the Gaussian kernel, use
kernel size = 2*radius + 1 (same as the Mean filter) and radius = int(math.ceil(3 * sigma))
and the proper normalizing constant.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0,
and save as "task2.png".
"""
def gaussian_blur_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    kernelRadius = int(math.ceil(3 * sigma)) 
    kernelSize = 2*kernelRadius + 1 # must be odd
    
    kernel = util.gkern(kernelSize,sigma)
    
    if in_place == False:
        return convolution(image, kernel, kernelSize, kernelSize, False)
    else:
        convolution(image, kernel, kernelSize, kernelSize, False, True)


"""
Task 3: Separable Gaussian blur

Implement the function

separable_gaussian_blur_image (image : np.ndarray, sigma : float, in_place : bool) -> np.ndarray

to Gaussian blur an image using separate filters. "sigma" is the standard deviation of the Gaussian.
The separable filter should first Gaussian blur the image horizontally, followed by blurring the
image vertically. Call the convolution function twice, first with the horizontal kernel and then with
the vertical kernel. Use the proper normalizing constant while creating the kernel(s) and then
normalize using the given normalize_kernel() function before convolution. The final image should be
identical to that of gaussian_blur_image.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0, and save as "task3.png".
"""
# Too slow...
def separable_gaussian_blur_image (image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :  
    
    kernelRadius = int(math.ceil(3 * sigma)) 
    kernelSize = 2*kernelRadius + 1 # must be odd
    
    krn_x, krn_y = util.get_separable_gkerns(kernelSize, sigma)
    
    if in_place == False:
        one = convolution(image, krn_x, kernelSize,1, False)
        return convolution(one, krn_y, 1,kernelSize,False)
    else:
        convolution(image, krn_x, kernelSize,1, False, True)
        convolution(image, krn_y, 1, kernelSize, False, True)


"""
Task 4: Image derivatives

Implement the functions

first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray
first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray and
second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray

to find the first and second derivatives of an image and then Gaussian blur the derivative
image by calling the gaussian_blur_image function. "sigma" is the standard deviation of the
Gaussian used for blurring. To compute the first derivatives, first compute the x-derivative
of the image (using the horizontal 1*3 kernel: [-1, 0, 1]) followed by Gaussian blurring the
resultant image. Then compute the y-derivative of the original image (using the vertical 3*1
kernel: [-1, 0, 1]) followed by Gaussian blurring the resultant image.
The second derivative should be computed by convolving the original image with the
2-D Laplacian of Gaussian (LoG) kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] and then applying
Gaussian Blur. Note that the kernel values sum to 0 in these cases, so you don't need to
normalize the kernels. Remember to add 128 to the final pixel values in all 3 cases, so you
can see the negative values. Note that the resultant images of the two first derivatives
will be shifted a bit because of the uneven size of the kernels.

To do: Compute the x-derivative, the y-derivative and the second derivative of the image
"cactus.jpg" with a sigma of 1.0 and save the final images as "task4a.png", "task4b.png"
and "task4c.png" respectively.
"""
def first_deriv_image_x(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    hz_kernel = np.array([[-1,0,1]])
    if in_place == False:
        res = convolution(image,hz_kernel,3,1,True)
        return gaussian_blur_image(res,sigma)
    else:
        convolution(image,hz_kernel,3,1,True,True)
        gaussian_blur_image(image,sigma,True)

def first_deriv_image_y(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    vt_kernel = np.array([[-1],[0],[1]])
    if in_place == False:
        res = convolution(image,vt_kernel,1,3,True)
        return gaussian_blur_image(res,sigma)
    else:
        convolution(image,vt_kernel,1,3,True,True)
        gaussian_blur_image(image,sigma,True)

def second_deriv_image(image : np.ndarray, sigma : float, in_place : bool = False) -> np.ndarray :
    LoG_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    if in_place == False:
        res = convolution(image,LoG_kernel,3,3,True)
        return gaussian_blur_image(res,sigma)
    else:
        convolution(image,LoG_kernel,3,3,True,True)
        gaussian_blur_image(image,sigma,True)

"""
Task 5: Image sharpening

Implement the function
sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray
to sharpen an image by subtracting the Gaussian-smoothed second derivative of an image, multiplied
by the constant "alpha", from the original image. "sigma" is the Gaussian standard deviation. Use
the second_deriv_image implementation and subtract back off the 128 that second derivative added on.

To do: Sharpen "yosemite.png" with a sigma of 1.0 and alpha of 5.0 and save as "task5.png".
"""
def sharpen_image(image : np.ndarray, sigma : float, alpha : float, in_place : bool = False) -> np.ndarray :
    second_derivative = second_deriv_image(image,sigma)
    subtrahend = gaussian_blur_image(second_derivative,sigma) - 128
    if in_place == False:
        return (image - alpha*(subtrahend))
    else:
        image = image - alpha*(subtrahend)


"""
Task 6: Edge Detection

Implement 
sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray
to compute edge magnitude and orientation information. Convert the image into grayscale.
Use the standard Sobel masks in X and Y directions:
[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] respectively to compute
the edges. Note that the kernel values sum to 0 in these cases, so you don't need to normalize the
kernels before convolving. Divide the image gradient values by 8 before computing the magnitude and
orientation in order to avoid spurious edges. sobel_image should then display both the magnitude and
orientation of the edges in the image.

To do: Compute Sobel edge magnitude and orientation on "cactus.jpg" and save as "task6.png".
"""
def sobel_image(image : np.ndarray, in_place : bool = False) -> np.ndarray :
    
    # expects only grayscale 2D image of type double.
    
    # the method takes in a colored double image, therefore a new axis is created (to be used in convolution)
    # convert image to grayscale and add 1 empty dimension
    #grayscale = cv2.cvtColor(util.img_to_uint8(image), cv2.COLOR_RGB2GRAY)
    grayscale = image[..., np.newaxis]
    
    # Sobel kernels in X and Y
    x_sobel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype = np.double) 
    y_sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.double)
    
    # Convolve X and y
    G_x = convolution(grayscale, x_sobel, 3, 3, False)
    G_y = convolution(grayscale, y_sobel, 3, 3, False)
    
    # Calculate magnitude (we divide by 8 to avoid spurious edges)
    
    theta = np.arctan2((G_y), (G_x)) # arctan2 to avoid division by 0
    if in_place == False:
        magnitude = np.sqrt( np.square(G_x/8) + np.square( G_y/8) ).astype(np.double)
        return magnitude[:,:,0], theta[:,:,0]

    else:
        image[:,:] = np.sqrt(np.square(G_x/8) + np.square( G_y/8) ).astype(np.double)[:,:,0]
        return theta[:,:,0]

"""
Task 7: Bilinear Interpolation

Implement the function
bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray

to compute the linearly interpolated pixel value at the point (x,y) using bilinear interpolation.
Both x and y are real values. Put the red, green, and blue interpolated results in the vector "rgb".

To do: The function rotate_image will be implemented in a lab and it uses bilinear_interpolation
to rotate an image. Rotate the image "yosemite.png" by 20 degrees and save as "task7.png".
"""
def bilinear_interpolation(image : np.ndarray, x : float, y : float) -> np.ndarray :
    
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))

    x1 = x0 + 1
    y1 = y0 + 1
    const = 1/((x1 - x0)*(y1 - y0))
    a1 = np.array([x1 - x, x - x0], dtype = np.double)
    a2 = np.array([[y1 - y],[y - y0]], dtype = np.double)
    intrp = np.zeros(3, dtype = np.double)
    
    # With values of angles other than 90,180,270 etc algorithm would ask for
    # values outside of image's boundaries. These do not exist, therefore I put 255 on every channel instead (white)
    if x1 <= (image.shape[0]-1) and y1 <= (image.shape[1]-1):
        for c in range(image.shape[2]):
            vals = np.array([[image[x0,y0,c], image[x0, y1,c]],[image[x1,y0,c], image[x1, y1,c]]], dtype = np.double)
            intrp[c] = const * np.matmul(np.matmul(a1, vals),a2)
    else:
        intrp += 255
    return intrp

def rotate_image_fast(interpolation_fn, image: np.ndarray, rotation_angle: float, in_place: bool = False) -> np.ndarray:
    """
    Function to rotate an image around its center and using an appropriate interpolation function with signature
    defined in hw1.py. Optimisations by A. Käver.
    """
    radians = math.radians(rotation_angle)
    image_copy = np.zeros_like(image)
    image_height, image_width, *_ = image.shape
    image_height_div2 = image_height / 2.0
    image_width_div2 = image_width / 2.0

    cos = math.cos(radians)
    sin = math.sin(radians)

    for r in range(image_height):
        x0 = r - image_height_div2
        x0_cos = x0 * cos
        x0_sin = x0 * sin
        for c in range(image_width):
            y0 = c - image_width_div2
            x1 = x0_cos - y0 * sin
            y1 = x0_sin + y0 * cos
            x1 += image_height_div2
            y1 += image_width_div2
            rgb = interpolation_fn(image, x1, y1)
            image_copy[r][c] = rgb
    return image_copy


 
"""
Task 8: Finding edge peaks

Implement the function
find_peaks_image(image : np.ndarray, thres : float, in_place : bool = False) -> np.ndarray
to find the peaks of edge responses perpendicular to the edges. The edge magnitude and orientation
at each pixel are to be computed using the Sobel operators. The original image is again converted
into grayscale in the starter code. A peak response is found by comparing a pixel's edge magnitude
to that of the two samples perpendicular to the edge at a distance of one pixel, which requires the
bilinear_interpolation function
(Hint: You need to create an image of magnitude values at each pixel to send as input to the
interpolation function).
If the pixel's edge magnitude is e and those of the other two are e1 and e2, e must be larger than
"thres" (threshold) and also larger than or equal to e1 and e2 for the pixel to be a peak response.
Assign the peak responses a value of 255 and everything else 0. Compute e1 and e2 as follows:

(please check the separate task8.pdf)

To do: Find the peak responses in "virgintrains.jpg" with thres = 40.0 and save as "task8.png".
What would be a better value for thres?
"""
def find_peaks_image(image : np.ndarray, thres : float, in_place : bool = False) -> np.ndarray :
    mag, theta = sobel_image(image)
    theta = util.rad2deg(theta)
    mag = mag[..., np.newaxis].astype(np.double)
    out_img = np.zeros_like(image)

    def compare_peaks(img):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if mag[y][x][0] > thres:
                    angle = theta[y][x]
                    e1x = y + 1 * np.cos(angle)
                    e1y = x + 1 * np.sin(angle)
                    e2x = y - 1 * np.cos(angle)
                    e2y = x - 1 * np.sin(angle)
                    e1_val = bilinear_interpolation(mag, e1x, e1y)
                    e2_val = bilinear_interpolation(mag, e2x, e2y)
                    if mag[y][x][0] >= e1_val[0]:
                        if mag[y][x][0] >= e2_val[0]:
                            img[y][x] = 255
                else:
                    img[y][x] = 0

    if in_place == True:
        compare_peaks(image)
    else:
        compare_peaks(out_img)
        return out_img

"""
Task 9 (a): K-means color clustering with random seeds (extra task)

Implement the function

random_seed_image(image : np.ndarray, num_clusters : int, in_place : bool = False) -> np.ndarray

to perform K-Means Clustering on a color image with randomly selected initial cluster centers
in the RGB color space. "num_clusters" is the number of clusters into which the pixel values
in the image are to be clustered. Use random.randint(0,255) to initialize each R, G and B value.
to create #num_clusters centers, assign each pixel of the image to its closest cluster center
and then update the cluster centers with the average of the RGB values of the pixels belonging
to that cluster until convergence. Use max iteration # = 100 and L1 distance between pixels,
i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30 (or anything suitable). Note: Your code
should account for the case when a cluster contains 0 pixels during an iteration. Also, since
this algorithm is random, you will get different resultant images every time you call the function.

To do: Perform random seeds clustering on "flowers.png" with num_clusters = 4 and save as "task9a.png".
"""
def random_seed_image(image : np.ndarray, num_clusters : int, in_place : bool = False) -> np.ndarray :
    "implement the function here"
    raise "not implemented yet!"

"""
Task 9 (b): K-means color clustering with pixel seeds (extra)

Implement the function
pixel_seed_image(image : np.ndarray, num_clusters: int, in_place : bool = False)
to perform K-Means Clustering on a color image with initial cluster centers sampled from the
image itself in the RGB color space. "num_clusters" is the number of clusters into which the
pixel values in the image are to be clustered. Choose a pixel and make its RGB values a seed
if it is sufficiently different (dist(L1) >= 100) from already-selected seeds. Repeat till
you get #num_clusters different seeds. Use max iteration # = 100 and L1 distance between pixels,
 i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
 the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30.

To do: Perform pixel seeds clustering on "flowers.png" with num_clusters = 5 and save as "task9b.png".
"""
def pixel_seed_image(image : np.ndarray, num_clusters: int, in_place : bool = False) -> np.ndarray :
    "implement the function here"
    raise "not implemented yet!"
