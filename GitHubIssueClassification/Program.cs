using System;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Tranforms;

namespace GitHubIssueClassification
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        static IDataView _trainingDataView;

        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed:0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath,hasHeader: true);
            var pipline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            
            Evaluate();
        }

        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description" outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }

        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            // Create the training algorithm class
            var trainingPipeline = pipline.Append(_mlContext.MulticlassClassification.Trainers.StochasticDualCooredinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            // Train the model
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            _predEngine = _trainModel.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_mlContext);
            // Predicts the area based on training data
            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description = "The WebSockets communication used under the covers by SignalR looks like is going in my development machine.."
            };
            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Results: {prediction.Area} ===============");
            // Saves the model to a zip file
            SaveModelAsFile(_mlContext, _trainedModel);
            // Return the model
            return trainingPipeline;
        }

        public static void Evaluate()
        {
            // Load test dataset
            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath,hasHeader: true);
            // Create the multiclass evaluator
            // Evaluate the model and create metrics
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            // Displays metrics
            Console.WriteLine($"********************************************************************");
            Console.WriteLine($"*     Metrics for Multi-class Classification model - Test Data     *");
            Console.WriteLine($"*------------------------------------------------------------------*");
            Console.WriteLine($"*     MicroAccuracy:    {testMetrics.AccuracyMicro:0.###}          *");
            Console.WriteLine($"*     MacroAccuracy:    {testMetrics.AccuracyMacro:0.###}          *");
            Console.WriteLine($"*     LogLoss:          {testMetrics.LogLoss:#.###}                *");
            Console.WriteLine($"*     LogLossReduction: {testMetrics.LogLossReduction:#.###}       *");
            Console.WriteLine($"********************************************************************");
        }

        public static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            // Saves the model as a zip file
            using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fs);
            Console.WriteLine($"The model is saved to {_modelPath}");
        }
    }
}
