using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;

namespace CircleDetectorLib
{
    public class EmguCvPrepration
    {
        /// <summary>
        /// Loads an image from the specified file path
        /// </summary>
        /// <param name="filePath">Path to the image file</param>
        /// <returns>The loaded image as Mat object</returns>
        public Mat LoadImageFromFile(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

            if (!System.IO.File.Exists(filePath))
                throw new System.IO.FileNotFoundException("Image file not found", filePath);

            return CvInvoke.Imread(filePath, ImreadModes.Color);
        }

        /// <summary>
        /// Saves an image to the specified file path
        /// </summary>
        /// <param name="image">Image to save</param>
        /// <param name="filePath">Path where the image should be saved</param>
        /// <returns>True if successful, false otherwise</returns>
        public bool SaveImageToFile(Mat image, string filePath)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "Image cannot be null");

            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

            try
            {
                CvInvoke.Imwrite(filePath, image);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Adjusts the brightness of an image
        /// </summary>
        /// <param name="image">Input image</param>
        /// <param name="factor">Brightness factor (> 1 increases brightness, < 1 decreases brightness)</param>
        /// <returns>The adjusted image</returns>
        public Mat AdjustBrightness(Mat image, double factor)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "Image cannot be null");

            if (factor <= 0)
                throw new ArgumentException("Factor must be greater than 0", nameof(factor));

            Mat result = new Mat();

            // Convert the image to a format that can be scaled
            Mat convertedImage = new Mat();
            image.ConvertTo(convertedImage, DepthType.Cv32F);

            // Scale the image - directly adjust brightness using convertedImage * factor
            CvInvoke.ConvertScaleAbs(image, result, factor, 0);

            // Alternative approach if ConvertScaleAbs doesn't work well for your use case:
            // Mat[] channels = new Mat[3];
            // CvInvoke.Split(convertedImage, channels);
            //
            // for (int i = 0; i < channels.Length; i++)
            // {
            //     channels[i] = channels[i] * factor;
            // }
            //
            // CvInvoke.Merge(channels, result);
            // 
            // foreach (var channel in channels)
            // {
            //     channel.Dispose();
            // }

            // Convert back to the original format
            convertedImage.ConvertTo(result, image.Depth);

            convertedImage.Dispose();

            return result;
        }

        /// <summary>
        /// Applies Gaussian blur to an image
        /// </summary>
        /// <param name="image">Input image</param>
        /// <param name="kernelSize">Kernel size (must be odd and greater than 1, e.g., 3, 5, 7)</param>
        /// <param name="sigma">Standard deviation of the Gaussian distribution</param>
        /// <returns>The blurred image</returns>
        public Mat ApplyGaussianBlur(Mat image, Size kernelSize, double sigma = 0)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "Image cannot be null");

            if (kernelSize.Width <= 1 || kernelSize.Height <= 1)
                throw new ArgumentException("Kernel size must be greater than 1", nameof(kernelSize));

            if (kernelSize.Width % 2 == 0 || kernelSize.Height % 2 == 0)
                throw new ArgumentException("Kernel dimensions must be odd numbers", nameof(kernelSize));

            Mat result = new Mat();
            CvInvoke.GaussianBlur(image, result, kernelSize, sigma);

            return result;
        }

        /// <summary>
        /// Applies Canny edge detection to an image
        /// </summary>
        /// <param name="image">Input image</param>
        /// <param name="threshold1">First threshold for the hysteresis procedure</param>
        /// <param name="threshold2">Second threshold for the hysteresis procedure</param>
        /// <param name="apertureSize">Aperture size for the Sobel operator (default is 3)</param>
        /// <param name="l2Gradient">Specifies whether to use L2 gradient norm (true) or L1 norm (false)</param>
        /// <returns>The edge image</returns>
        public Mat ApplyCannyEdgeDetection(Mat image, double threshold1, double threshold2, int apertureSize = 3, bool l2Gradient = false)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "Image cannot be null");

            if (threshold1 < 0 || threshold2 < 0)
                throw new ArgumentException("Thresholds must be non-negative", nameof(threshold1));

            if (apertureSize != 3 && apertureSize != 5 && apertureSize != 7)
                throw new ArgumentException("Aperture size must be 3, 5, or 7", nameof(apertureSize));

            // Convert to grayscale if needed
            Mat grayImage;
            if (image.NumberOfChannels == 3)
            {
                grayImage = new Mat();
                CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            }
            else
            {
                grayImage = image.Clone();
            }

            // Apply Canny edge detection
            Mat edges = new Mat();
            CvInvoke.Canny(grayImage, edges, threshold1, threshold2, apertureSize, l2Gradient);

            // Dispose of gray image if we created it
            if (image.NumberOfChannels == 3)
            {
                grayImage.Dispose();
            }

            return edges;
        }
    }
}