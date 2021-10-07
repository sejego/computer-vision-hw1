(* Utility functions. Implemented by Oluwandabira Alawode. *)

#r "nuget: SixLabors.ImageSharp, 1.0.3"

#r "nuget: MathNet.Numerics, 5.0.0-alpha02"
#r "nuget: MathNet.Numerics.FSharp, 5.0.0-alpha02"
#r "nuget: MathNet.Filtering, 0.7.0"

open System

open SixLabors.ImageSharp
open SixLabors.ImageSharp.PixelFormats

open MathNet.Numerics.LinearAlgebra


module Matrix =
    /// Return the Matrix's RowCount and ColumnCount as a tuple.
    let size (mat: Matrix<float>) = mat.RowCount, mat.ColumnCount

    /// <summary>Return the minimum and maximum values of the Matrix as a tuple <c>(min, max)</c>.</summary>
    let minmax mat =
        mat
        |> Matrix.fold
            (fun (min, max) i ->
                if i < min then (i, max)
                elif i > max then (min, i)
                else (min, max))
            (Double.MaxValue, Double.MinValue)

type ImageFormat =
    /// MatrixImage with 3 elements of the same size [| R; G; B |] with values (0.0 .. 255.0).</summary>
    | RGB
    /// MatrixImage with 1 element with float values (0.0 .. 255.0).
    | Grayscale
    /// MatrixImage that is not in RGB or Grayscale format.
    | Invalid

/// <summary>Alias of <c>Matrix&lt;float&gt; array</c>.</summary>
type MatrixImage = Matrix<float> array

/// <summary>Utility module for loading and saving <see cref="MatrixImage">MatrixImage</see>s.</summary>
/// <remarks>Uses ImageSharp to endcode and decode the images.</remarks>
[<RequireQualifiedAccess>]
module MatrixImage =
    type Rgb24 with
        /// <summary>Colorimetric (perceptual luminance-preserving) conversion to grayscale.</summary>
        /// <remarks>Calculated from https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale.</remarks>
        member x.Grayscale =
            let toLinear c = (float c) / 255.0

            let linear =
                0.2126 * (toLinear x.R)
                + 0.7152 * (toLinear x.G)
                + 0.0722 * (toLinear x.B)

            linear * 255.

    /// Scale all values to be between 0.0 and 255.0 inclusive.
    let scaleTo255 (m: Matrix<float>) =
        let (min, max) = Matrix.minmax m
        m.Map((fun v -> v * (max - min) / 255.), Zeros.AllowSkip)

    /// Detect the ImageFormat of a MatrixImage.
    let ImageFormat (img: MatrixImage) =
        match img.Length with
        | 1 -> Grayscale
        | 3 ->
            if (Array.distinctBy Matrix.size img).Length = 1 then
                RGB
            else
                Invalid
        | _ -> Invalid

    let private grayscaleFromRgb24 (img: Image<Rgb24>) : Matrix<float> =
        let data = DenseMatrix.zero img.Width img.Height

        for y in img.Height - 1 .. -1 .. 0 do
            let ps = img.GetPixelRowSpan(y).ToArray()

            for x in 0 .. img.Width - 1 do
                data.[x, y] <- ps.[x].Grayscale

        data

    let private fromRgb24 (img: Image<Rgb24>) : MatrixImage =
        let data =
            Array.init 3 (fun _ -> DenseMatrix.zero img.Width img.Height)

        for y in img.Height - 1 .. -1 .. 0 do
            let ps = (img.GetPixelRowSpan y).ToArray()

            for x in 0 .. img.Width - 1 do
                data.[0].[x, y] <- float ps.[x].R
                data.[1].[x, y] <- float ps.[x].G
                data.[2].[x, y] <- float ps.[x].B

        data

    /// <summary>Loads an image from the filesystem.</summary>
    /// <param name="str">Filepath of the image to be decoded.</param>
    /// <param name="format"><see cref="ImageFormat">ImageFormat</see> to decode into.</param>
    let Load (str: string) format =
        let img = Image.Load<Rgb24>(str)

        match format with
        | RGB -> fromRgb24 img
        | Grayscale -> [| grayscaleFromRgb24 img |]
        | Invalid -> failwith "Cannot intentionally load an invalid image."


    let private toRgb24 (img: MatrixImage) =
        match ImageFormat img with
        | Invalid -> None
        | format ->
            let bytes = Array.Parallel.map scaleTo255 img
            let (width, height) = Matrix.size img.[0]

            let data =
                Array.Parallel.init
                    (width * height)
                    (fun i ->
                        let x = i % width
                        let y = i / width

                        if format = Grayscale then
                            let b = byte bytes.[0].[x, y] in Rgb24(b, b, b)
                        else
                            Rgb24(byte bytes.[0].[x, y], byte bytes.[1].[x, y], byte bytes.[2].[x, y]))

            Some <| Image.LoadPixelData(data, width, height)

    /// <summary>Save an image to the filesystem.</summary>
    /// <remarks>Image encoding is detected from <paramref name="str"/>.</remarks>
    /// <param name="str">Filepath to save the image to.</param>
    /// <param name="img"><see cref="MatrixImage">Image</see> to be saved.</param>
    let Save (str: string) img =
        match toRgb24 img with
        | Some img -> img.Save(str)
        | None -> failwith "Cannot save an invalid image."

/// A module for creating and normalizing simple kernels.
module Kernel =
    /// <summary>Normalizes the kernel.</summary>
    /// <remarks>Returns original kernel if the sum of elements is 0.</remarks>
    let normalize (kernel: Matrix<float>) =
        let sum = Matrix.sum kernel

        if sum = 0. then
            kernel
        else
            kernel / sum

    /// <summary>Creates an identiy kernel.</summary>
    /// <param name="k">The created kernel will have length <c>2 * k + 1</c>.</param>
    let identity (k: int) =
        let n = 2 * k + 1
        let zeroes = DenseMatrix.zero n n
        zeroes.[k, k] <- 1.
        zeroes

    /// <summary>Creates a box blur kernel.</summary>
    /// <param name="k">The created kernel will have length <c>2 * k + 1</c>.</param>
    let box (k: int) =
        let n = 2 * k + 1
        let nn = float <| n * n
        DenseMatrix.init n n (fun _ _ -> 1. / nn)

    /// <summary>Creates a 2D Gaussian kernel.</summary>
    /// <remarks>Directly implements the 2D Gaussian function described in https://en.wikipedia.org/wiki/Gaussian_blur</remarks>
    /// <param name="k">The created kernel will have length <c>2 * k + 1</c>.</param>
    /// <param name="sigma">Standard deviation of the Gaussian destribution.</param>
    let gaussian (k: int) (sigma: float) =
        let n = 2 * k + 1

        let sig2 = sigma * sigma

        let gaussuianXY x y =
            let x = float x
            let y = float y
            let p = -(x * x + y * y) / (2. * sig2)
            (1. / 2. * (Math.PI) * sig2) * Math.E ** p

        DenseMatrix.init n n (fun x y -> gaussuianXY (x - k) (y - k))