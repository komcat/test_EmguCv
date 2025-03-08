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
    /// Arguments for progress reporting events
    /// </summary>
    public class ProgressEventArgs : EventArgs
    {
        /// <summary>
        /// Current progress value
        /// </summary>
        public int Current { get; }

        /// <summary>
        /// Total operations to complete
        /// </summary>
        public int Total { get; }

        /// <summary>
        /// Progress percentage (0-100)
        /// </summary>
        public double PercentComplete { get; }

        /// <summary>
        /// Optional status message
        /// </summary>
        public string Message { get; }

        public ProgressEventArgs(int current, int total, string message = null)
        {
            Current = current;
            Total = total;
            PercentComplete = total > 0 ? (current * 100.0 / total) : 0;
            Message = message;
        }
    }

    /// <summary>
    /// Arguments for status message events
    /// </summary>
    public class StatusMessageEventArgs : EventArgs
    {
        /// <summary>
        /// The status message
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Indicates if this is a warning or error message
        /// </summary>
        public bool IsWarningOrError { get; }

        public StatusMessageEventArgs(string message, bool isWarningOrError = false)
        {
            Message = message;
            IsWarningOrError = isWarningOrError;
        }
    }

    /// <summary>
    /// Arguments for circle detection result events
    /// </summary>
    public class CircleDetectionResultEventArgs : EventArgs
    {
        /// <summary>
        /// The detected circle
        /// </summary>
        public CircleF? Circle { get; }

        /// <summary>
        /// Parameters used for detection
        /// </summary>
        public Dictionary<string, object> Parameters { get; }

        /// <summary>
        /// Additional result information
        /// </summary>
        public Dictionary<string, object> ResultInfo { get; }

        public CircleDetectionResultEventArgs(
            CircleF? circle,
            Dictionary<string, object> parameters,
            Dictionary<string, object> resultInfo)
        {
            Circle = circle;
            Parameters = parameters;
            ResultInfo = resultInfo;
        }
    }

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
        public bool SaveFileDuringIteration { get; set; } = false;

        // Events for reporting progress and results
        public event EventHandler<ProgressEventArgs> ProgressChanged;
        public event EventHandler<StatusMessageEventArgs> StatusMessageReported;
        public event EventHandler<CircleDetectionResultEventArgs> CircleDetectionCompleted;

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
        /// Raises the ProgressChanged event
        /// </summary>
        protected virtual void OnProgressChanged(int current, int total, string message = null)
        {
            ProgressChanged?.Invoke(this, new ProgressEventArgs(current, total, message));
        }

        /// <summary>
        /// Raises the StatusMessageReported event
        /// </summary>
        protected virtual void OnStatusMessageReported(string message, bool isWarningOrError = false)
        {
            StatusMessageReported?.Invoke(this, new StatusMessageEventArgs(message, isWarningOrError));
        }

        /// <summary>
        /// Raises the CircleDetectionCompleted event
        /// </summary>
        protected virtual void OnCircleDetectionCompleted(
            CircleF? circle,
            Dictionary<string, object> parameters,
            Dictionary<string, object> resultInfo)
        {
            CircleDetectionCompleted?.Invoke(this,
                new CircleDetectionResultEventArgs(circle, parameters, resultInfo));
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
        /// <param name="saveIntermediateResults">Whether to save intermediate images during parameter search</param>
        /// <param name="maxIntermediateImagesSaved">Maximum number of intermediate images to save</param>
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
            int maxThreads = 0,
            bool saveIntermediateResults = false,
            int maxIntermediateImagesSaved = 20)
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

            // Create log directory for parameter search if outputPath is provided and intermediate saving is enabled
            string logDir = null;
            if (!string.IsNullOrEmpty(outputPath) && saveIntermediateResults && SaveFileDuringIteration)
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

            // Report initial status
            OnStatusMessageReported($"Starting parallel parameter search with {totalCombinations} combinations...");
            OnStatusMessageReported($"Canny range: {cannyStart}-{cannyEnd}, step {cannyStep}");
            OnStatusMessageReported($"Accumulator range: {accumStart}-{accumEnd}, step {accumStep}");
            OnStatusMessageReported($"Target diameter: {targetDiameter} px");
            OnStatusMessageReported($"Using {maxThreads} threads");
            OnStatusMessageReported($"Weighting: Size={_sizeWeight * 100:F0}%, Center Proximity={_centerWeight * 100:F0}%");

            if (saveIntermediateResults && SaveFileDuringIteration)
                OnStatusMessageReported($"Saving up to {maxIntermediateImagesSaved} intermediate results");
            else
                OnStatusMessageReported("Not saving intermediate results (final result only)");

            // Counter for progress reporting
            int processedCount = 0;
            int circlesFound = 0;
            int intermediateImagesSaved = 0;

            // Thread-safe collection for results
            ConcurrentBag<CircleSearchResult> results = new ConcurrentBag<CircleSearchResult>();

            // For tracking best intermediate results to save
            ConcurrentBag<CircleSearchResult> bestResultsToSave = new ConcurrentBag<CircleSearchResult>();

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
                                var result = new CircleSearchResult
                                {
                                    Circle = bestCircle,
                                    Canny = canny,
                                    Accum = accum,
                                    SizeDifference = bestSizeDiff,
                                    CenterDistance = bestCenterDist,
                                    Score = bestScore
                                };

                                results.Add(result);

                                // Increment the circles found counter
                                Interlocked.Increment(ref circlesFound);

                                // If this is a good match and we're saving intermediate results, add to candidates
                                if (saveIntermediateResults && SaveFileDuringIteration && bestScore < 0.15) // Higher threshold for better matches
                                {
                                    bestResultsToSave.Add(result);
                                }
                            }
                        }
                    }

                    // Update progress in a thread-safe manner
                    lock (lockObj)
                    {
                        processedCount++;

                        // Report progress every 100 combinations or when reaching milestones
                        if (processedCount % 100 == 0 || processedCount == totalCombinations)
                        {
                            OnProgressChanged(processedCount, totalCombinations,
                                $"Testing parameter combinations: {processedCount}/{totalCombinations}");
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

            // Save intermediate results if enabled
            if (saveIntermediateResults && SaveFileDuringIteration && logDir != null)
            {
                // Sort results by score (best first) and take only up to maxIntermediateImagesSaved
                var sortedResults = bestResultsToSave.OrderBy(r => r.Score).Take(maxIntermediateImagesSaved);

                int savedCount = 0;
                foreach (var result in sortedResults)
                {
                    try
                    {
                        using (Mat outputImage = image.Clone())
                        {
                            // Draw the circle
                            using (Mat drawnImage = _houghCircleDetector.DrawCircles(
                                outputImage, new[] { result.Circle }, new MCvScalar(0, 0, 255), 2))
                            {
                                // Draw a line from center of image to center of circle
                                Point imgCenter = new Point((int)imageCenter.X, (int)imageCenter.Y);
                                Point circleCenter = new Point((int)result.Circle.Center.X, (int)result.Circle.Center.Y);
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
                                    (int)result.Circle.Center.X - (int)result.Circle.Radius,
                                    (int)result.Circle.Center.Y - (int)result.Circle.Radius - 10);

                                CvInvoke.PutText(drawnImage,
                                    $"C{result.Canny} A{result.Accum} D={result.Circle.Radius * 2:F1}px Dist={result.CenterDistance:F1}px",
                                    textPos,
                                    FontFace.HersheyComplex, 0.5, new MCvScalar(0, 255, 0), 1);

                                // Use thread-safe file naming
                                string intermediatePath = System.IO.Path.Combine(
                                    logDir,
                                    $"Rank{savedCount + 1}_Score{result.Score:F3}_C{result.Canny}_A{result.Accum}.png");

                                CvInvoke.Imwrite(intermediatePath, drawnImage);
                                savedCount++;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        // If there's an error saving the image, just log it and continue
                        OnStatusMessageReported($"Error saving image: {ex.Message}", true);
                    }
                }

                OnStatusMessageReported($"Saved {savedCount} intermediate results out of {bestResultsToSave.Count} candidates");
            }

            // Prepare result information
            Dictionary<string, object> parameters = new Dictionary<string, object>
            {
                { "CannyThreshold", bestCanny },
                { "AccumulatorThreshold", bestAccum },
                { "TargetDiameter", targetDiameter },
                { "SizeWeight", _sizeWeight },
                { "CenterWeight", _centerWeight }
            };

            Dictionary<string, object> resultInfo = new Dictionary<string, object>
            {
                { "ParameterCombinationsTested", processedCount },
                { "CirclesFound", circlesFound },
                { "Score", bestScore },
                { "DiameterDifference", bestSizeDiff },
                { "DiameterDifferencePercent", bestSizeDiff / targetDiameter * 100 },
                { "CenterDistance", bestCenterDist }
            };

            // If we found a circle and output path is provided, draw and save
            if (bestCircle.HasValue && !string.IsNullOrEmpty(outputPath))
            {
                OnStatusMessageReported($"Parameter search completed!");
                OnStatusMessageReported($"Tested {processedCount} parameter combinations");
                OnStatusMessageReported($"Found {circlesFound} circles in total");
                OnStatusMessageReported($"Best parameters: Canny={bestCanny}, Accum={bestAccum}");
                OnStatusMessageReported($"Best circle: Center=({bestCircle.Value.Center.X:F1}, {bestCircle.Value.Center.Y:F1}), " +
                              $"Diameter={bestCircle.Value.Radius * 2:F1}px (Target: {targetDiameter}px)");
                OnStatusMessageReported($"Diameter difference: {bestSizeDiff:F1}px ({bestSizeDiff / targetDiameter * 100:F1}% of target)");
                OnStatusMessageReported($"Distance from image center: {bestCenterDist:F1}px");
                OnStatusMessageReported($"Combined score: {bestScore:F4} (lower is better)");

                // Create a summary file
                if (logDir != null && SaveFileDuringIteration)
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
                OnStatusMessageReported($"Parameter search completed. No circles found matching the criteria.");
                OnStatusMessageReported($"Tested {processedCount} parameter combinations");
            }

            // Notify about the final result
            OnCircleDetectionCompleted(bestCircle, parameters, resultInfo);

            // Clean up
            image.Dispose();

            return bestCircle;
        }
    }
}