using System;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;
using CircleDetectorLib;

namespace CircleDetectorApp
{
    class Program
    {
        static void Main(string[] args)
        {
            string inputFile = "test.png";
            string outputPath = "result.png";
            double targetDiameter = 99; // Target circle diameter in pixels

            // Create the circle finder with custom weights
            var circleFinder = new EmguCvSingleCircleFinderCenter(sizeWeight: 0.7, centerWeight: 0.3);

            // Disable intermediate file saving
            circleFinder.SaveFileDuringIteration = false;

            // Subscribe to events for progress and status updates
            circleFinder.ProgressChanged += OnProgressChanged;
            circleFinder.StatusMessageReported += OnStatusMessageReported;
            circleFinder.CircleDetectionCompleted += OnCircleDetectionCompleted;

            try
            {
                // Find the circle
                var circle = circleFinder.FindSingleCircleParallel(
                    imagePath: inputFile,
                    targetDiameter: targetDiameter,
                    outputPath: outputPath,
                    cannyStart: 50,
                    cannyEnd: 150,
                    accumStart: 20,
                    accumEnd: 80);

                // Process result if needed (additional to event handling)
                if (circle.HasValue)
                {
                    Console.WriteLine($"Circle found and saved to {outputPath}");
                }
                else
                {
                    Console.WriteLine("No circle found matching the criteria.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing image: {ex.Message}");
            }

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        // Event handlers
        private static void OnProgressChanged(object sender, ProgressEventArgs e)
        {
            // Update progress bar or status
            Console.WriteLine($"Progress: {e.Current}/{e.Total} ({e.PercentComplete:F1}%)");

            // You could update a progress bar in a GUI application
            // progressBar.Value = (int)e.PercentComplete;
        }

        private static void OnStatusMessageReported(object sender, StatusMessageEventArgs e)
        {
            // Display status messages
            if (e.IsWarningOrError)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"WARNING: {e.Message}");
                Console.ResetColor();

                // In a GUI app you might log this or show in a different way
                // logTextBox.AppendText($"WARNING: {e.Message}\n");
            }
            else
            {
                Console.WriteLine(e.Message);

                // In a GUI app you might do:
                // statusTextBox.AppendText($"{e.Message}\n");
            }
        }

        private static void OnCircleDetectionCompleted(object sender, CircleDetectionResultEventArgs e)
        {
            if (e.Circle.HasValue)
            {
                CircleF circle = e.Circle.Value;

                Console.WriteLine("\nCIRCLE DETECTION RESULTS:");
                Console.WriteLine($"Circle center: ({circle.Center.X:F1}, {circle.Center.Y:F1})");
                Console.WriteLine($"Circle diameter: {circle.Radius * 2:F1} pixels");

                // Access additional result information
                if (e.ResultInfo.ContainsKey("Score"))
                {
                    Console.WriteLine($"Detection score: {e.ResultInfo["Score"]:F4}");
                }

                if (e.ResultInfo.ContainsKey("DiameterDifferencePercent"))
                {
                    Console.WriteLine($"Diameter difference: {e.ResultInfo["DiameterDifferencePercent"]:F1}%");
                }

                if (e.ResultInfo.ContainsKey("CenterDistance"))
                {
                    Console.WriteLine($"Distance from image center: {e.ResultInfo["CenterDistance"]:F1} pixels");
                }

                // Additional information about parameters used
                if (e.Parameters.ContainsKey("CannyThreshold") && e.Parameters.ContainsKey("AccumulatorThreshold"))
                {
                    Console.WriteLine($"Best parameters: Canny={e.Parameters["CannyThreshold"]}, " +
                                     $"Accumulator={e.Parameters["AccumulatorThreshold"]}");
                }
            }
            else
            {
                Console.WriteLine("\nNo circle was detected that matches the criteria.");

                // You might still want to know how many combinations were tested
                if (e.ResultInfo.ContainsKey("ParameterCombinationsTested"))
                {
                    Console.WriteLine($"Tested {e.ResultInfo["ParameterCombinationsTested"]} parameter combinations");
                }
            }
        }
    }
}