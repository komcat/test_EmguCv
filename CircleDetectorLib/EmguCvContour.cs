using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace CircleDetectorLib
{
    public class EmguCvContour
    {
        /// <summary>
        /// Detects circles in an image using contour detection approach
        /// </summary>
        /// <param name="image">Input image (should be binary or edge image for best results)</param>
        /// <param name="minRadius">Minimum radius to consider as circle</param>
        /// <param name="maxRadius">Maximum radius to consider as circle</param>
        /// <param name="circularityThreshold">Threshold for circularity (1.0 is perfect circle, typically use > 0.8)</param>
        /// <returns>Array of detected circles</returns>
        public CircleF[] DetectCirclesUsingContours(
            Mat image,
            int minRadius = 10,
            int maxRadius = 100,
            double circularityThreshold = 0.8)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image), "Image cannot be null");

            if (minRadius <= 0)
                throw new ArgumentException("Minimum radius must be greater than 0", nameof(minRadius));

            if (maxRadius <= minRadius)
                throw new ArgumentException("Maximum radius must be greater than minimum radius", nameof(maxRadius));

            if (circularityThreshold <= 0 || circularityThreshold > 1.0)
                throw new ArgumentException("Circularity threshold must be between 0 and 1.0", nameof(circularityThreshold));

            // Make a copy to work with
            Mat processedImage = image.Clone();

            // Convert to binary if it's not already
            if (processedImage.NumberOfChannels == 3)
            {
                Mat grayImage = new Mat();
                CvInvoke.CvtColor(processedImage, grayImage, ColorConversion.Bgr2Gray);
                processedImage.Dispose();
                processedImage = grayImage;
            }

            // If the image is an edge image (like Canny output), we need to make it binary
            if (processedImage.Depth != DepthType.Cv8U)
            {
                Mat binaryImage = new Mat();
                CvInvoke.ConvertScaleAbs(processedImage, binaryImage, 1.0, 0);
                processedImage.Dispose();
                processedImage = binaryImage;
            }

            // Make sure we have white contours on black background
            Mat binary = new Mat();
            CvInvoke.Threshold(processedImage, binary, 127, 255, ThresholdType.Binary);
            processedImage.Dispose();

            // Find contours
            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(
                    binary,
                    contours,
                    null, // hierarchy, not needed
                    RetrType.List,
                    ChainApproxMethod.ChainApproxSimple);

                // Process contours to find circles
                List<CircleF> detectedCircles = new List<CircleF>();

                for (int i = 0; i < contours.Size; i++)
                {
                    using (VectorOfPoint contour = contours[i])
                    {
                        double area = CvInvoke.ContourArea(contour);

                        // Skip small contours
                        if (area < Math.PI * minRadius * minRadius)
                            continue;

                        // Skip large contours
                        if (area > Math.PI * maxRadius * maxRadius)
                            continue;

                        // Get the perimeter
                        double perimeter = CvInvoke.ArcLength(contour, true);

                        // Calculate circularity
                        // For a perfect circle: 4 * pi * area / (perimeter^2) = 1
                        double circularity = 4 * Math.PI * area / (perimeter * perimeter);

                        // Filter based on circularity
                        if (circularity < circularityThreshold)
                            continue;

                        // Calculate approximate circle parameters using moments
                        var moments = CvInvoke.Moments(contour);
                        double centerX = moments.M10 / moments.M00;
                        double centerY = moments.M01 / moments.M00;

                        // Calculate radius from area
                        double radius = Math.Sqrt(area / Math.PI);

                        // Alternative: calculate radius as average distance from center to contour points
                        double radiusSum = 0;
                        int pointCount = 0;

                        // Get points from contour
                        Point[] points = contour.ToArray();
                        foreach (Point p in points)
                        {
                            double dx = p.X - centerX;
                            double dy = p.Y - centerY;
                            double distance = Math.Sqrt(dx * dx + dy * dy);
                            radiusSum += distance;
                            pointCount++;
                        }

                        if (pointCount > 0)
                        {
                            // Average radius from points
                            double avgRadius = radiusSum / pointCount;

                            // Use average of area-based and point-based radius
                            radius = (radius + avgRadius) / 2;
                        }

                        // Filter based on radius constraints
                        if (radius < minRadius || radius > maxRadius)
                            continue;

                        // Add to detected circles
                        detectedCircles.Add(new CircleF(new PointF((float)centerX, (float)centerY), (float)radius));
                    }
                }

                binary.Dispose();

                return detectedCircles.ToArray();
            }
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
        /// Processes an image with contour-based circle detection
        /// </summary>
        /// <param name="inputImage">Input image</param>
        /// <param name="outputPath">Path to save the processed image</param>
        /// <param name="minRadius">Minimum radius of circles to detect</param>
        /// <param name="maxRadius">Maximum radius of circles to detect</param>
        /// <param name="circularityThreshold">Threshold for circularity (0.0-1.0)</param>
        /// <returns>List of strings with information about detected circles</returns>
        public List<string> ProcessAndSaveContourCircleDetection(
            Mat inputImage,
            string outputPath,
            int minRadius = 10,
            int maxRadius = 100,
            double circularityThreshold = 0.8)
        {
            if (inputImage == null)
                throw new ArgumentNullException(nameof(inputImage), "Input image cannot be null");

            if (string.IsNullOrEmpty(outputPath))
                throw new ArgumentException("Output path cannot be null or empty", nameof(outputPath));

            // Preprocess the image - convert to grayscale
            Mat grayImage = new Mat();
            if (inputImage.NumberOfChannels == 3)
            {
                CvInvoke.CvtColor(inputImage, grayImage, ColorConversion.Bgr2Gray);
            }
            else
            {
                grayImage = inputImage.Clone();
            }

            // Apply Gaussian blur to reduce noise
            Mat blurredImage = new Mat();
            CvInvoke.GaussianBlur(grayImage, blurredImage, new Size(5, 5), 1.5);
            grayImage.Dispose();

            // Apply Canny edge detection
            Mat edgeImage = new Mat();
            CvInvoke.Canny(blurredImage, edgeImage, 50, 150);
            blurredImage.Dispose();

            // Detect circles using contours
            CircleF[] circles = DetectCirclesUsingContours(
                edgeImage,
                minRadius,
                maxRadius,
                circularityThreshold);

            // Draw circles on the original image
            Mat outputImage = DrawCircles(inputImage, circles, new MCvScalar(0, 0, 255), 2); // Red color for circles

            // Save the processed image
            CvInvoke.Imwrite(outputPath, outputImage);

            // Get information about detected circles
            List<string> circleInfo = GetCircleInformation(circles);

            // Clean up
            edgeImage.Dispose();
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
        /// <param name="circularityThreshold">Threshold for circularity (0.0-1.0)</param>
        /// <returns>Array of detected circles</returns>
        public CircleF[] FindCirclesFromEdgeImage(
            Mat edgeImage,
            Mat originalImage = null,
            string outputPath = null,
            int minRadius = 10,
            int maxRadius = 100,
            double circularityThreshold = 0.8)
        {
            if (edgeImage == null)
                throw new ArgumentNullException(nameof(edgeImage), "Edge image cannot be null");

            // Detect circles using contours
            CircleF[] circles = DetectCirclesUsingContours(
                edgeImage,
                minRadius,
                maxRadius,
                circularityThreshold);

            // If original image and output path are provided, draw and save the result
            if (originalImage != null && !string.IsNullOrEmpty(outputPath))
            {
                using (Mat resultImage = DrawCircles(originalImage, circles, new MCvScalar(0, 0, 255), 2))
                {
                    CvInvoke.Imwrite(outputPath, resultImage);
                }
            }

            // Print circle information to console
            Console.WriteLine($"Detected {circles.Length} circles using contour method:");
            foreach (CircleF circle in circles)
            {
                Console.WriteLine($"Circle: Center=({circle.Center.X:F1}, {circle.Center.Y:F1}), Radius={circle.Radius:F1}");
            }

            return circles;
        }

        /// <summary>
        /// Computes maximum distance from a point to a set of points
        /// </summary>
        /// <param name="center">Center point</param>
        /// <param name="points">Array of points</param>
        /// <returns>Maximum distance</returns>
        private double MaxDistance(PointF center, Point[] points)
        {
            double maxDist = 0;
            foreach (Point p in points)
            {
                double dx = p.X - center.X;
                double dy = p.Y - center.Y;
                double dist = Math.Sqrt(dx * dx + dy * dy);
                if (dist > maxDist)
                    maxDist = dist;
            }
            return maxDist;
        }

        /// <summary>
        /// Computes minimum distance from a point to a set of points
        /// </summary>
        /// <param name="center">Center point</param>
        /// <param name="points">Array of points</param>
        /// <returns>Minimum distance</returns>
        private double MinDistance(PointF center, Point[] points)
        {
            double minDist = double.MaxValue;
            foreach (Point p in points)
            {
                double dx = p.X - center.X;
                double dy = p.Y - center.Y;
                double dist = Math.Sqrt(dx * dx + dy * dy);
                if (dist < minDist)
                    minDist = dist;
            }
            return minDist;
        }
    }
}