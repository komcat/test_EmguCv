using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace CircleDetectorLib
{
    public class EmguCvHough
    {
        /// <summary>
        /// Detects circles in an image using the Hough Circle Transform
        /// </summary>
        /// <param name="image">Input image (should be grayscale for best results)</param>
        /// <param name="minRadius">Minimum radius to detect</param>
        /// <param name="maxRadius">Maximum radius to detect</param>
        /// <param name="cannyThreshold">Threshold for the Canny edge detector</param>
        /// <param name="accumulatorThreshold">Accumulator threshold for circle detection</param>
        /// <param name="minDistBetweenCircles">Minimum distance between detected circles</param>
        /// <returns>Array of detected circles</returns>
        public CircleF[] DetectCircles(
            Mat image,
            int minRadius = 10,
            int maxRadius = 100,
            double cannyThreshold = 100.0,
            double accumulatorThreshold = 50.0,
            double minDistBetweenCircles = 20.0)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "Image cannot be null");

            if (minRadius <= 0)
                throw new ArgumentException("Minimum radius must be greater than 0", nameof(minRadius));

            if (maxRadius <= minRadius)
                throw new ArgumentException("Maximum radius must be greater than minimum radius", nameof(maxRadius));

            // Convert to grayscale if the image has multiple channels
            Mat grayImage = new Mat();
            if (image.NumberOfChannels == 3)
            {
                CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);
            }
            else
            {
                grayImage = image.Clone();
            }

            // Apply Gaussian blur to reduce noise
            Mat blurredImage = new Mat();
            CvInvoke.GaussianBlur(grayImage, blurredImage, new Size(5, 5), 1.5);

            // Use Hough Circle Transform to detect circles
            List<CircleF> circles = new List<CircleF>();

            // Create a Mat to store the detected circles
            using (Mat circlesMat = new Mat())
            {
                // HoughCircles parameters:
                // - blurredImage: input image (should be grayscale)
                // - circlesMat: output Mat of circles
                // - HoughModes.Gradient: detection method
                // - dp: resolution of accumulator (1 = same as input image)
                // - minDistBetweenCircles: minimum distance between circle centers
                // - cannyThreshold: higher threshold passed to Canny edge detector
                // - accumulatorThreshold: threshold for circle center detection
                // - minRadius: minimum circle radius
                // - maxRadius: maximum circle radius
                CvInvoke.HoughCircles(
                    blurredImage,
                    circlesMat,
                    HoughModes.Gradient,
                    1.0, // dp
                    minDistBetweenCircles,
                    cannyThreshold,
                    accumulatorThreshold,
                    minRadius,
                    maxRadius);

                // Convert the Mat result to CircleF array
                if (!circlesMat.IsEmpty)
                {
                    // Get the number of circles detected
                    int circlesCount = circlesMat.Cols;

                    // Create array to hold circle data
                    float[] circleData = new float[circlesCount * 3]; // Each circle has 3 values (x, y, radius)

                    // Convert circlesMat to 1D array of floats
                    circlesMat.CopyTo(circleData);

                    // Convert the float array to CircleF objects
                    for (int i = 0; i < circlesCount; i++)
                    {
                        int idx = i * 3;
                        PointF center = new PointF(circleData[idx], circleData[idx + 1]);
                        float radius = circleData[idx + 2];
                        circles.Add(new CircleF(center, radius));
                    }
                }
            }

            // Clean up
            grayImage.Dispose();
            blurredImage.Dispose();

            return circles.ToArray();
        }

        /// <summary>
        /// Draws detected circles on an image
        /// </summary>
        /// <param name="image">Image to draw circles on</param>
        /// <param name="circles">Array of circles to draw</param>
        /// <param name="circleColor">Color for the circles</param>
        /// <param name="thickness">Thickness of the circle outline (use -1 to fill the circle)</param>
        /// <returns>Image with circles drawn on it</returns>
        public Mat DrawCircles(Mat image, CircleF[] circles, MCvScalar circleColor, int thickness = 2)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "Image cannot be null");

            if (circles == null)
                throw new ArgumentNullException(nameof(circles), "Circles array cannot be null");

            // Create a copy of the image to draw on
            Mat outputImage = image.Clone();

            // Draw each circle
            foreach (CircleF circle in circles)
            {
                Point center = new Point((int)circle.Center.X, (int)circle.Center.Y);
                int radius = (int)circle.Radius;

                // Draw the circle outline
                CvInvoke.Circle(outputImage, center, radius, circleColor, thickness);

                // Draw the center point
                CvInvoke.Circle(outputImage, center, 2, new MCvScalar(0, 255, 0), -1);
            }

            return outputImage;
        }

        /// <summary>
        /// Gets information about detected circles
        /// </summary>
        /// <param name="circles">Array of detected circles</param>
        /// <returns>List of strings with circle information</returns>
        public List<string> GetCircleInformation(CircleF[] circles)
        {
            if (circles == null)
                throw new ArgumentNullException(nameof(circles), "Circles array cannot be null");

            List<string> circleInfo = new List<string>();

            for (int i = 0; i < circles.Length; i++)
            {
                CircleF circle = circles[i];
                string info = $"Circle {i + 1}: Center=({circle.Center.X:F1}, {circle.Center.Y:F1}), Radius={circle.Radius:F1}";
                circleInfo.Add(info);
            }

            return circleInfo;
        }

        /// <summary>
        /// Detects and processes circles in an image, saving the result
        /// </summary>
        /// <param name="inputImage">Input image</param>
        /// <param name="outputPath">Path to save the processed image</param>
        /// <param name="minRadius">Minimum radius to detect</param>
        /// <param name="maxRadius">Maximum radius to detect</param>
        /// <param name="cannyThreshold">Threshold for the Canny edge detector</param>
        /// <param name="accumulatorThreshold">Accumulator threshold for circle detection</param>
        /// <param name="minDistBetweenCircles">Minimum distance between detected circles</param>
        /// <returns>Information about detected circles</returns>
        public List<string> ProcessAndSaveCircleDetection(
            Mat inputImage,
            string outputPath,
            int minRadius = 10,
            int maxRadius = 100,
            double cannyThreshold = 100.0,
            double accumulatorThreshold = 50.0,
            double minDistBetweenCircles = 20.0)
        {
            if (inputImage == null)
                throw new ArgumentNullException(nameof(inputImage), "Input image cannot be null");

            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));

            // Detect circles
            CircleF[] circles = DetectCircles(
                inputImage,
                minRadius,
                maxRadius,
                cannyThreshold,
                accumulatorThreshold,
                minDistBetweenCircles);

            // Draw circles on the image
            Mat outputImage = DrawCircles(inputImage, circles, new MCvScalar(0, 0, 255), 2); // Red color for circles

            // Save the processed image
            CvInvoke.Imwrite(outputPath, outputImage);

            // Get information about detected circles
            List<string> circleInfo = GetCircleInformation(circles);

            // Clean up
            outputImage.Dispose();

            return circleInfo;
        }

        /// <summary>
        /// Detects circles from an image that has already been processed with edge detection
        /// </summary>
        /// <param name="edgeImage">Edge-detected input image</param>
        /// <param name="originalImage">Original image for drawing circles (optional, can be null)</param>
        /// <param name="outputPath">Path to save the result (optional, can be null)</param>
        /// <param name="minRadius">Minimum radius to detect</param>
        /// <param name="maxRadius">Maximum radius to detect</param>
        /// <param name="accumulatorThreshold">Accumulator threshold for circle detection</param>
        /// <param name="minDistBetweenCircles">Minimum distance between detected circles</param>
        /// <returns>Array of detected circles</returns>
        public CircleF[] FindCirclesFromEdgeImage(
            Mat edgeImage,
            Mat originalImage = null,
            string outputPath = null,
            int minRadius = 10,
            int maxRadius = 100,
            double accumulatorThreshold = 50.0,
            double minDistBetweenCircles = 20.0)
        {
            if (edgeImage == null)
                throw new ArgumentNullException(nameof(edgeImage), "Edge image cannot be null");

            // Use Hough Circle Transform to detect circles
            List<CircleF> circles = new List<CircleF>();

            // Create a Mat to store the detected circles
            using (Mat circlesMat = new Mat())
            {
                // Apply Hough Circle Transform
                CvInvoke.HoughCircles(
                    edgeImage,
                    circlesMat,
                    HoughModes.Gradient,
                    1.0, // dp
                    minDistBetweenCircles,
                    100.0, // Using a fixed value for Canny threshold since the image is already edge-detected
                    accumulatorThreshold,
                    minRadius,
                    maxRadius);

                // Process the detected circles
                if (!circlesMat.IsEmpty)
                {
                    // Get the number of circles detected
                    int circlesCount = circlesMat.Cols;

                    // Create array to hold circle data
                    float[] circleData = new float[circlesCount * 3]; // Each circle has 3 values (x, y, radius)

                    // Convert circlesMat to an array of floats
                    circlesMat.CopyTo(circleData);

                    // Convert the float array to CircleF objects
                    for (int i = 0; i < circlesCount; i++)
                    {
                        int idx = i * 3;
                        PointF center = new PointF(circleData[idx], circleData[idx + 1]);
                        float radius = circleData[idx + 2];
                        circles.Add(new CircleF(center, radius));
                    }
                }
            }

            CircleF[] circlesArray = circles.ToArray();

            // If original image and output path are provided, draw and save the result
            if (originalImage != null && !string.IsNullOrEmpty(outputPath))
            {
                using (Mat resultImage = DrawCircles(originalImage, circlesArray, new MCvScalar(0, 0, 255), 2))
                {
                    CvInvoke.Imwrite(outputPath, resultImage);
                }
            }

            // Print circle information to console
            Console.WriteLine($"Detected {circlesArray.Length} circles:");
            foreach (CircleF circle in circlesArray)
            {
                Console.WriteLine($"Circle: Center=({circle.Center.X:F1}, {circle.Center.Y:F1}), Radius={circle.Radius:F1}");
            }

            return circlesArray;
        }

        /// <summary>
        /// Detects circles directly from an input image
        /// </summary>
        /// <param name="inputImage">Input image</param>
        /// <param name="outputPath">Path to save the result (optional, can be null)</param>
        /// <param name="minRadius">Minimum radius to detect</param>
        /// <param name="maxRadius">Maximum radius to detect</param>
        /// <param name="cannyThreshold">Threshold for the Canny edge detector</param>
        /// <param name="accumulatorThreshold">Accumulator threshold for circle detection</param>
        /// <param name="minDistBetweenCircles">Minimum distance between detected circles</param>
        /// <returns>Array of detected circles</returns>
        public CircleF[] FindCirclesFromImage(
            Mat inputImage,
            string outputPath = null,
            int minRadius = 10,
            int maxRadius = 100,
            double cannyThreshold = 100.0,
            double accumulatorThreshold = 50.0,
            double minDistBetweenCircles = 20.0)
        {
            if (inputImage == null)
                throw new ArgumentNullException(nameof(inputImage), "Input image cannot be null");

            // Detect circles using the Hough Circle Transform
            CircleF[] circles = DetectCircles(
                inputImage,
                minRadius,
                maxRadius,
                cannyThreshold,
                accumulatorThreshold,
                minDistBetweenCircles);

            // If output path is provided, draw and save the result
            if (!string.IsNullOrEmpty(outputPath))
            {
                using (Mat resultImage = DrawCircles(inputImage, circles, new MCvScalar(0, 0, 255), 2))
                {
                    CvInvoke.Imwrite(outputPath, resultImage);
                }
            }

            // Print circle information to console
            Console.WriteLine($"Detected {circles.Length} circles:");
            foreach (CircleF circle in circles)
            {
                Console.WriteLine($"Circle: Center=({circle.Center.X:F1}, {circle.Center.Y:F1}), Radius={circle.Radius:F1}");
            }

            return circles;
        }
    }
}