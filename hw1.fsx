(*
ITS8030: Homework 1

Please implement all functions below.

For submission create a project called its8030-2021-hw1 and put the solution in there.

*)

#load "Util.fsx"

open MathNet.Numerics.LinearAlgebra

open Util

(*
Task 1: Convolution

Implement the function below to convolve an image with a kernel.
Use zero-padding around the borders for simplicity (what other options would there be?).

if add is true, then 128 is added to each pixel for the result to get rid of negatives.

Write a general convolution function that can handle all possible cases as mentioned above.
You can get help from the convolution part of the function mean_blur_image (to be implemented in a lab)
to write this function.
*)
let convolution (kernel: Matrix<float>) (add: bool) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"

(*
Task 2: Gaussian blur

Implement the function to Gaussian blur an image. "sigma" is the standard deviation of the Gaussian.
Use the function mean_blur_image as a template, create a 2D Gaussian filter
as the kernel and call the convolution function of Task 1.
Normalize the created kernel using the function Kernel.normalize (to
be implemented in a lab) before convolution. For the Gaussian kernel, use
kernel size = 2*radius + 1 (same as the Mean filter) and radius = int(math.ceil(3 * sigma))
and the proper normalizing constant.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0,
and save as "task2.png".

*)
let gaussianBlurImage (sigma: float) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"

(*
Task 3: Separable Gaussian blur

Implement the function below to Gaussian blur an image using separate filters. "sigma" is the standard deviation of the Gaussian.
The separable filter should first Gaussian blur the image horizontally, followed by blurring the
image vertically. Call the convolution function twice, first with the horizontal kernel and then with
the vertical kernel. Use the proper normalizing constant while creating the kernel(s) and then
normalize using the given Kernel.normalize function before convolution. The final image should be
identical to that of gaussian_blur_image.

To do: Gaussian blur the image "songfestival.jpg" using this function with a sigma of 4.0, and save as "task3.png".
*)
let separableGaussianBlurImage (sigma: float) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"

(*
Task 4: Image derivatives

Implement the functions bewlo to find the first and second derivatives of an image and then Gaussian blur the derivative
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

*)
let firstDerivImageX (sigma: float) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"
let firstDerivImageY (sigma: float) (image: Matrix<float>) = failwith "Unimplemented"
let secondDerivImage (sigma: float) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"


(*
Task 5: Image sharpening

Implement the function below to sharpen an image by subtracting the Gaussian-smoothed second derivative of an image, multiplied
by the constant "alpha", from the original image. "sigma" is the Gaussian standard deviation. Use
the second_deriv_image implementation and subtract back off the 128 that second derivative added on.

To do: Sharpen "yosemite.png" with a sigma of 1.0 and alpha of 5.0 and save as "task5.png".
*)
let sharpenImage (sigma: float) (alpha: float) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"

(*
Task 6: Edge Detection

Implement the function below to compute edge magnitude and orientation information. Convert the image into grayscale.
Use the standard Sobel masks in X and Y directions:
[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] and [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] respectively to compute
the edges. Note that the kernel values sum to 0 in these cases, so you don't need to normalize the
kernels before convolving. Divide the image gradient values by 8 before computing the magnitude and
orientation in order to avoid spurious edges. sobel_image should then display both the magnitude and
orientation of the edges in the image.

To do: Compute Sobel edge magnitude and orientation on "cactus.jpg" and save as "task6.png".
*)
let sobelImage (image: MatrixImage) : MatrixImage = failwith "Unimplemented"

(*
Task 7: Bilinear Interpolation

Implement the function
bilinearInterpolation function below to compute the linearly interpolated pixel value at the point (x,y) using bilinear interpolation.
Both x and y are real values. Put the red, green, and blue interpolated results in the vector "rgb".

To do: The function rotateImage will be implemented in a lab and it uses bilinearInterpolation
to rotate an image. Rotate the image "yosemite.png" by 20 degrees and save as "task7.png".
*)
let bilinearInterpolation (x: float) (y: float) (image: MatrixImage) : Vector<float> = failwith "Unimplemented"

let rotateImage (angle: float) (image: MatrixImage) : MatrixImage =
    let rgb = bilinearInterpolation 0.5 0. image
    // To be implemented by the lecturer.
    failwith "Unimplemented"

(*
Task 8: Finding edge peaks

Implement the function below to find the peaks of edge responses perpendicular to the edges. The edge magnitude and orientation
at each pixel are to be computed using the Sobel operators. The original image is again converted
into grayscale in the starter code. A peak response is found by comparing a pixel's edge magnitude
to that of the two samples perpendicular to the edge at a distance of one pixel, which requires the
bilinearInterpolation function
(Hint: You need to create an image of magnitude values at each pixel to send as input to the
interpolation function).
If the pixel's edge magnitude is e and those of the other two are e1 and e2, e must be larger than
"thres" (threshold) and also larger than or equal to e1 and e2 for the pixel to be a peak response.
Assign the peak responses a value of 255 and everything else 0. Compute e1 and e2 as follows:

(please check the separate task8.pdf)

To do: Find the peak responses in "virgintrains.jpg" with thres = 40.0 and save as "task8.png".
What would be a better value for thres?
*)
let findPeakImage (thres: float) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"


(*
Task 9 (a): K-means color clustering with random seeds (extra task)

Implement the function below to perform K-Means Clustering on a color image with randomly selected initial cluster centers
in the RGB color space. "numClusters" is the number of clusters into which the pixel values
in the image are to be clustered. Use random.Next(0,255) to initialize each R, G and B value.
to create #num_clusters centers, assign each pixel of the image to its closest cluster center
and then update the cluster centers with the average of the RGB values of the pixels belonging
to that cluster until convergence. Use max iteration # = 100 and L1 distance between pixels,
i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30 (or anything suitable). Note: Your code
should account for the case when a cluster contains 0 pixels during an iteration. Also, since
this algorithm is random, you will get different resultant images every time you call the function.
*)

let randomSeedImage (numClusters: int) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"


(*
Task 9 (b): K-means color clustering with pixel seeds (extra)

Implement the function below to perform K-Means Clustering on a color image with initial cluster centers sampled from the
image itself in the RGB color space. "numClusters" is the number of clusters into which the
pixel values in the image are to be clustered. Choose a pixel and make its RGB values a seed
if it is sufficiently different (dist(L1) >= 100) from already-selected seeds. Repeat till
you get #num_clusters different seeds. Use max iteration # = 100 and L1 distance between pixels,
 i.e. dist = |Red1 - Red2| + |Green1 - Green2| + |Blue1 - Blue2|. The algorithm converges when
 the sum of the L1 distances between the new cluster centers and the previous cluster centers
is less than epsilon*num_clusters. Choose epsilon = 30.

To do: Perform pixel seeds clustering on "flowers.png" with num_clusters = 5 and save as "task9b.png".
*)
let pixelSeedImage (numClusters: int) (image: MatrixImage) : MatrixImage = failwith "Unimplemented"
