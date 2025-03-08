using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace CircleDetectorLib
{
    /// <summary>
    /// Specializes in finding a single circle closest to a target diameter and center of image
    /// using parallel processing for parameter optimization
    /// </summary>
    public class EmguCvSingleCircleFinderCenter
    {
        private readonly EmguCvHough _houghCircleDetector;
        private readonly EmguCvPrepration _imageProcessor;

        // Weight factors for scoring circles (can be adjusted)
        private double _sizeWeight = 0.7;  // How important size match is (0-1)
        private double _centerWeight = 0.3; // How important center proximity is (0-1)
        private bool SaveFileDuringIteration = false;
        public EmguCvSingleCircleFinderCenter(double sizeWeight = 0.7, double centerWeight = 0.3)
        {
            _houghCircleDetector = new EmguCvHough();
            _imageProcessor = new EmguCvPrepration();

            // Ensure weights are valid and sum to 1.0
            if (sizeWeight < 0 || centerWeight < 0 || (sizeWeight + centerWeight) != 1.0)
            {
                double sum = sizeWeight + centerWeight;
                _sizeWeight = sizeWeight / sum;
                _centerWeight = centerWeight / sum;
            }
            else
            {
                _sizeWeight = sizeWeight;
                _centerWeight = centerWeight;
            }
        }

        /// <summary>
        /// Result of a parameter combination test
        /// </summary>
        private class CircleSearchResult
        {
            public CircleF Circle { get; set; }
            public int Canny { get; set; }
            public int Accum { get; set; }
            public double SizeDifference { get; set; }
            public double CenterDistance { get; set; }
            public double Score { get; set; } // Combined score (lower is better)
        }

        /// <summary>
        /// Calculates the score for a circle based on size match and center proximity
        /// </summary>
        /// <param name="circle">The circle to evaluate</param>
        /// <param name="targetDiameter">Target diameter</param>
        /// <param name="imageCenter">Center point of the image</param>
        /// <param name="imageSize">Size of the image for normalization</param>
        /// <returns>Combined score (lower is better)</returns>
        private double CalculateCircleScore(
            CircleF circle,
            double targetDiameter,
            PointF imageCenter,
            Size imageSize)
        {
            // Calculate size difference as percentage of target diameter
            double sizeDiff = Math.Abs(circle.Radius * 2 - targetDiameter) / targetDiameter;

            // Calculate distance from image center, normalized by image diagonal
            double dx = circle.Center.X - imageCenter.X;
            double dy = circle.Center.Y - imageCenter.Y;
            double centerDist = Math.Sqrt(dx * dx + dy * dy);

            // Normalize by image diagonal (max possible distance)
            double diagonal = Math.Sqrt(imageSize.Width * imageSize.Width + imageSize.Height * imageSize.Height);
            double normalizedCenterDist = centerDist / (diagonal / 2);

            // Combined score (weighted sum)
            return (_sizeWeight * sizeDiff) + (_centerWeight * normalizedCenterDist);
        }

        /// <summary>
        /// Finds a single circle using a parallel parameter search approach that considers
        /// both size match and proximity to image center
        /// </summary>
        /// <param name="imagePath">Path to the input image</param>
        /// <param name="targetDiameter">Target diameter to find</param>
        /// <param name="outputPath">Path to save the result</param>
        /// <param name="diameterTolerance">Tolerance for the diameter</param>
        /// <param name="cannyStart">Starting value for Canny threshold (default: 10)</param>
        /// <param name="cannyEnd">Ending value for Canny threshold (default: 200)</param>
        /// <param name="cannyStep">Step size for Canny threshold (default: 2)</param>
        /// <param name="accumStart">Starting value for accumulator threshold (default: 10)</param>
        /// <param name="accumEnd">Ending value for accumulator threshold (default: 100)</param>
        /// <param name="accumStep">Step size for accumulator threshold (default: 2)</param>
        /// <param name="maxThreads">Maximum number of threads to use (default: number of processor cores)</param>
        /// <returns>The found circle or null if none found</returns>
        public CircleF? FindSingleCircleParallel(
            string imagePath,
            double targetDiameter,
            string outputPath = null,
            double diameterTolerance = 0,
            int cannyStart = 10,
            int cannyEnd = 200,
            int cannyStep = 2,
            int accumStart = 10,
            int accumEnd = 100,
            int accumStep = 2,
            int maxThreads = 0)
        {
            // If maxThreads is not specified, use the number of processors
            if (maxThreads <= 0)
            {
                maxThreads = Environment.ProcessorCount;
            }

            // Load the image
            Mat image = _imageProcessor.LoadImageFromFile(imagePath);

            // Calculate image center for distance measurements
            PointF imageCenter = new PointF(image.Width / 2f, image.Height / 2f);
            Size imageSize = new Size(image.Width, image.Height);

            // Create log directory for parameter search if outputPath is provided
            string logDir = null;
            if (!string.IsNullOrEmpty(outputPath))
            {
                logDir = System.IO.Path.Combine(
                    System.IO.Path.GetDirectoryName(outputPath),
                    "parameter_search_" + System.IO.Path.GetFileNameWithoutExtension(outputPath));

                if (!System.IO.Directory.Exists(logDir))
                {
                    System.IO.Directory.CreateDirectory(logDir);
                }
            }



            // Generate all parameter combinations
            var parameterCombinations = new List<(int Canny, int Accum)>();
            for (int canny = cannyStart; canny <= cannyEnd; canny += cannyStep)
            {
                for (int accum = accumStart; accum <= accumEnd; accum += accumStep)
                {
                    parameterCombinations.Add((canny, accum));
                }
            }

            int totalCombinations = parameterCombinations.Count;

            Console.WriteLine($"Starting parallel parameter search with {totalCombinations} combinations...");
            Console.WriteLine($"Canny range: {cannyStart}-{cannyEnd}, step {cannyStep}");
            Console.WriteLine($"Accumulator range: {accumStart}-{accumEnd}, step {accumStep}");
            Console.WriteLine($"Target diameter: {targetDiameter} px");
            Console.WriteLine($"Using {maxThreads} threads");
            Console.WriteLine($"Weighting: Size={_sizeWeight * 100:F0}%, Center Proximity={_centerWeight * 100:F0}%");
            Console.WriteLine();

            // Counter for progress reporting
            int processedCount = 0;
            int circlesFound = 0;

            // Thread-safe collection for results
            ConcurrentBag<CircleSearchResult> results = new ConcurrentBag<CircleSearchResult>();

            // Create a thread-safe progress reporting mechanism
            object lockObj = new object();

            // Process parameters in parallel
            Parallel.ForEach(
                parameterCombinations,
                new ParallelOptions { MaxDegreeOfParallelism = maxThreads },
                paramCombo =>
                {
                    int canny = paramCombo.Canny;
                    int accum = paramCombo.Accum;

                    // Clone the image for thread-safe processing
                    using (Mat threadImage = image.Clone())
                    {
                        // Try to find a circle with these parameters
                        CircleF[] circles = _houghCircleDetector.DetectCircles(
                            threadImage,
                            minRadius: (int)(targetDiameter / 2 * 0.7),  // Allow some flexibility in size
                            maxRadius: (int)(targetDiameter / 2 * 1.3),
                            cannyThreshold: canny,
                            accumulatorThreshold: accum,
                            minDistBetweenCircles: targetDiameter / 2);  // Avoid overlapping circles

                        // If circles were found, find the best one based on score
                        if (circles.Length > 0)
                        {
                            CircleF bestCircle = circles[0];
                            double bestScore = CalculateCircleScore(circles[0], targetDiameter, imageCenter, imageSize);
                            double bestSizeDiff = Math.Abs(circles[0].Radius * 2 - targetDiameter);

                            // Calculate center distance for best circle
                            double dx = circles[0].Center.X - imageCenter.X;
                            double dy = circles[0].Center.Y - imageCenter.Y;
                            double bestCenterDist = Math.Sqrt(dx * dx + dy * dy);

                            for (int i = 1; i < circles.Length; i++)
                            {
                                double score = CalculateCircleScore(circles[i], targetDiameter, imageCenter, imageSize);
                                if (score < bestScore)
                                {
                                    bestScore = score;
                                    bestCircle = circles[i];
                                    bestSizeDiff = Math.Abs(circles[i].Radius * 2 - targetDiameter);

                                    // Update center distance
                                    dx = circles[i].Center.X - imageCenter.X;
                                    dy = circles[i].Center.Y - imageCenter.Y;
                                    bestCenterDist = Math.Sqrt(dx * dx + dy * dy);
                                }
                            }

                            // If within tolerance (or if no tolerance specified), add to results
                            if (diameterTolerance <= 0 || bestSizeDiff <= diameterTolerance)
                            {
                                results.Add(new CircleSearchResult
                                {
                                    Circle = bestCircle,
                                    Canny = canny,
                                    Accum = accum,
                                    SizeDifference = bestSizeDiff,
                                    CenterDistance = bestCenterDist,
                                    Score = bestScore
                                });

                                // Increment the circles found counter
                                Interlocked.Increment(ref circlesFound);

                                // Save intermediate results if log directory is provided
                                if (logDir != null && bestScore < 0.2) // Only save good matches
                                {
                                    try
                                    {
                                        // Clone the image for drawing (thread-safe)
                                        using (Mat outputImage = threadImage.Clone())
                                        {
                                            // Draw the circle
                                            using (Mat drawnImage = _houghCircleDetector.DrawCircles(
                                                outputImage, new[] { bestCircle }, new MCvScalar(0, 0, 255), 2))
                                            {
                                                // Draw a line from center of image to center of circle
                                                Point imgCenter = new Point((int)imageCenter.X, (int)imageCenter.Y);
                                                Point circleCenter = new Point((int)bestCircle.Center.X, (int)bestCircle.Center.Y);
                                                CvInvoke.Line(drawnImage, imgCenter, circleCenter, new MCvScalar(255, 255, 0), 1);

                                                // Draw cross at image center
                                                int crossSize = 10;
                                                CvInvoke.Line(drawnImage,
                                                    new Point(imgCenter.X - crossSize, imgCenter.Y),
                                                    new Point(imgCenter.X + crossSize, imgCenter.Y),
                                                    new MCvScalar(0, 255, 255), 1);
                                                CvInvoke.Line(drawnImage,
                                                    new Point(imgCenter.X, imgCenter.Y - crossSize),
                                                    new Point(imgCenter.X, imgCenter.Y + crossSize),
                                                    new MCvScalar(0, 255, 255), 1);

                                                // Add text with information
                                                Point textPos = new Point(
                                                    (int)bestCircle.Center.X - (int)bestCircle.Radius,
                                                    (int)bestCircle.Center.Y - (int)bestCircle.Radius - 10);

                                                CvInvoke.PutText(drawnImage,
                                                    $"C{canny} A{accum} D={bestCircle.Radius * 2:F1}px Dist={bestCenterDist:F1}px",
                                                    textPos,
                                                    FontFace.HersheyComplex, 0.5, new MCvScalar(0, 255, 0), 1);

                                                // Use thread-safe file naming
                                                string intermediatePath = System.IO.Path.Combine(
                                                    logDir,
                                                    $"C{canny}_A{accum}_D{bestCircle.Radius * 2:F1}_Dist{bestCenterDist:F0}.png");

                                                CvInvoke.Imwrite(intermediatePath, drawnImage);
                                            }
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        // If there's an error saving the image, just log it and continue
                                        Console.WriteLine($"Error saving image: {ex.Message}");
                                    }
                                }
                            }
                        }
                    }

                    // Update progress in a thread-safe manner
                    lock (lockObj)
                    {
                        processedCount++;

                        // Display progress every 100 combinations or when reaching milestones
                        if (processedCount % 100 == 0 || processedCount == totalCombinations)
                        {
                            Console.WriteLine($"Progress: {processedCount}/{totalCombinations} combinations tested " +
                                              $"({(processedCount * 100.0 / totalCombinations):F1}%)");
                        }
                    }
                });

            // Find the best result based on combined score
            CircleF? bestCircle = null;
            int bestCanny = 0;
            int bestAccum = 0;
            double bestScore = double.MaxValue;
            double bestSizeDiff = 0;
            double bestCenterDist = 0;

            foreach (var result in results)
            {
                if (result.Score < bestScore)
                {
                    bestScore = result.Score;
                    bestCircle = result.Circle;
                    bestCanny = result.Canny;
                    bestAccum = result.Accum;
                    bestSizeDiff = result.SizeDifference;
                    bestCenterDist = result.CenterDistance;
                }
            }

            // If we found a circle and output path is provided, draw and save
            if (bestCircle.HasValue && !string.IsNullOrEmpty(outputPath))
            {
                Console.WriteLine();
                Console.WriteLine($"Parameter search completed!");
                Console.WriteLine($"Tested {processedCount} parameter combinations");
                Console.WriteLine($"Found {circlesFound} circles in total");
                Console.WriteLine($"Best parameters: Canny={bestCanny}, Accum={bestAccum}");
                Console.WriteLine($"Best circle: Center=({bestCircle.Value.Center.X:F1}, {bestCircle.Value.Center.Y:F1}), " +
                                  $"Diameter={bestCircle.Value.Radius * 2:F1}px (Target: {targetDiameter}px)");
                Console.WriteLine($"Diameter difference: {bestSizeDiff:F1}px ({bestSizeDiff / targetDiameter * 100:F1}% of target)");
                Console.WriteLine($"Distance from image center: {bestCenterDist:F1}px");
                Console.WriteLine($"Combined score: {bestScore:F4} (lower is better)");

                // Create a summary file
                if (logDir != null)
                {
                    string summaryPath = System.IO.Path.Combine(logDir, "search_summary.txt");
                    using (System.IO.StreamWriter writer = new System.IO.StreamWriter(summaryPath))
                    {
                        writer.WriteLine($"Parallel Parameter Search Summary");
                        writer.WriteLine($"===============================");
                        writer.WriteLine($"Target diameter: {targetDiameter} px");
                        writer.WriteLine($"Canny range: {cannyStart}-{cannyEnd}, step {cannyStep}");
                        writer.WriteLine($"Accumulator range: {accumStart}-{accumEnd}, step {accumStep}");
                        writer.WriteLine($"Threads used: {maxThreads}");
                        writer.WriteLine($"Size weight: {_sizeWeight:F2}, Center weight: {_centerWeight:F2}");
                        writer.WriteLine($"Combinations tested: {processedCount}");
                        writer.WriteLine($"Circles found: {circlesFound}");
                        writer.WriteLine();
                        writer.WriteLine($"Best Result");
                        writer.WriteLine($"==========");
                        writer.WriteLine($"Canny threshold: {bestCanny}");
                        writer.WriteLine($"Accumulator threshold: {bestAccum}");
                        writer.WriteLine($"Circle center: ({bestCircle.Value.Center.X:F1}, {bestCircle.Value.Center.Y:F1})");
                        writer.WriteLine($"Circle diameter: {bestCircle.Value.Radius * 2:F1}px");
                        writer.WriteLine($"Difference from target: {bestSizeDiff:F1}px ({bestSizeDiff / targetDiameter * 100:F1}% of target)");
                        writer.WriteLine($"Distance from image center: {bestCenterDist:F1}px");
                        writer.WriteLine($"Combined score: {bestScore:F4}");
                    }
                }

                // Save the best result
                using (Mat outputImage = image.Clone())
                {
                    // Draw the best circle
                    using (Mat drawnImage = _houghCircleDetector.DrawCircles(outputImage, new[] { bestCircle.Value }, new MCvScalar(0, 0, 255), 2))
                    {
                        // Draw a line from center of image to center of circle
                        Point imgCenter = new Point((int)imageCenter.X, (int)imageCenter.Y);
                        Point circleCenter = new Point((int)bestCircle.Value.Center.X, (int)bestCircle.Value.Center.Y);
                        CvInvoke.Line(drawnImage, imgCenter, circleCenter, new MCvScalar(255, 255, 0), 1);

                        // Draw cross at image center
                        int crossSize = 10;
                        CvInvoke.Line(drawnImage,
                            new Point(imgCenter.X - crossSize, imgCenter.Y),
                            new Point(imgCenter.X + crossSize, imgCenter.Y),
                            new MCvScalar(0, 255, 255), 1);
                        CvInvoke.Line(drawnImage,
                            new Point(imgCenter.X, imgCenter.Y - crossSize),
                            new Point(imgCenter.X, imgCenter.Y + crossSize),
                            new MCvScalar(0, 255, 255), 1);

                        // Add text with information
                        Point textPos = new Point((int)bestCircle.Value.Center.X - (int)bestCircle.Value.Radius,
                                                 (int)bestCircle.Value.Center.Y - (int)bestCircle.Value.Radius - 10);
                        CvInvoke.PutText(drawnImage,
                            $"BEST: C{bestCanny} A{bestAccum} D={bestCircle.Value.Radius * 2:F1}px",
                            textPos,
                            FontFace.HersheyComplex, 0.7, new MCvScalar(0, 255, 0), 2);

                        // Add distance information
                        Point distPos = new Point((int)bestCircle.Value.Center.X - (int)bestCircle.Value.Radius,
                                                 (int)bestCircle.Value.Center.Y - (int)bestCircle.Value.Radius - 35);
                        CvInvoke.PutText(drawnImage,
                            $"Center dist: {bestCenterDist:F1}px",
                            distPos,
                            FontFace.HersheyComplex, 0.6, new MCvScalar(0, 255, 0), 1);

                        CvInvoke.Imwrite(outputPath, drawnImage);
                    }
                }
            }
            else
            {
                Console.WriteLine();
                Console.WriteLine($"Parameter search completed. No circles found matching the criteria.");
                Console.WriteLine($"Tested {processedCount} parameter combinations");
            }

            // Clean up
            image.Dispose();

            return bestCircle;
        }
    }
}