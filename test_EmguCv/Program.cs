using System;
using System.IO;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;
using CircleDetectorLib;

namespace test_EmguCv
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting image processing...");


            string inputFile = "test.png";
            string outputDir = Path.GetDirectoryName(inputFile) ?? ".";
            string baseName = Path.GetFileNameWithoutExtension(inputFile);
            string extension = Path.GetExtension(inputFile);

            // Create instances of our classes
            EmguCvPrepration processor = new EmguCvPrepration();
            EmguCvHough circleDetector = new EmguCvHough();

            // Step 1: Load the image
            Console.WriteLine($"Loading image: {inputFile}");
            Mat originalImage = processor.LoadImageFromFile(inputFile);

            // Create instance of the center-aware circle finder (customize weights if needed)
            EmguCvSingleCircleFinderCenter centerFinder = new EmguCvSingleCircleFinderCenter(
                sizeWeight: 0.7,    // 70% importance on size match
                centerWeight: 0.3); // 30% importance on center proximity

            // Run a parallel parameter search
            string outputPath = Path.Combine(outputDir, $"center_aware_circle_{baseName}{extension}");
            CircleF? bestCircle = centerFinder.FindSingleCircleParallel(
                inputFile,
                targetDiameter: 100,  // Your target diameter in pixels
                outputPath: outputPath,
                // Using step size of 2 for faster iteration
                cannyStep: 2,
                accumStep: 2
            );

            if (bestCircle.HasValue)
            {
                Console.WriteLine($"Found optimal circle with diameter: {bestCircle.Value.Radius * 2:F1}px");
            }
            //EmguCvHough emguCvHough = new EmguCvHough();
            //// Process an image with contour-based circle detection

            //int canny = 10;
            //int accum = 10;

            //int cannymax = 200;
            //int accummax = 100;

            //while (canny < cannymax)
            //{
            //    while (accum < accummax)
            //    {
            //        string contourOutputFile = Path.Combine(outputDir, $"hugh_circle_{canny}_{accum}_{baseName}{extension}");
            //        List<string> circleInfo = emguCvHough.ProcessAndSaveCircleDetection(originalImage,
            //            contourOutputFile,
            //            minRadius: 40,
            //            maxRadius: 60,
            //            cannyThreshold: canny,
            //            accumulatorThreshold: accum,
            //            minDistBetweenCircles: 100);

            //        // Print information about detected circles with parameters
            //        Console.WriteLine($"Parameters: Canny={canny}, Accumulator={accum}");
            //        Console.WriteLine($"Detected {circleInfo.Count} circles:");
            //        foreach (string info in circleInfo)
            //        {
            //            Console.WriteLine(info);
            //        }
            //        Console.WriteLine(); // Add empty line for better readability

            //        accum++;
            //    }
            //    canny++;
            //}



            // Don't forget to dispose
            originalImage.Dispose();


            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }


}