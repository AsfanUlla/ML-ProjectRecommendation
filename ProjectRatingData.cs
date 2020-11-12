using Microsoft.ML.Data;

namespace ProjectRecommender
{
    public class ProjectRating
    {
        [LoadColumn(0)]
        public float userId;
        [LoadColumn(1)]
        public float projectId;
        [LoadColumn(2)]
        public float Label;
    }

    public class ProjectRatingPrediction
    {
        public float Label;
        public float Score;
    }
}
