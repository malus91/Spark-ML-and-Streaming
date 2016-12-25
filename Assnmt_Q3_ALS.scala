import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.ALS
//Databricks specific settings
val AccessKey = ""
val SecretKey = ""
val EncodedSecretKey = SecretKey.replace("/", "%2F")
val AwsBucketName = "assnmtdata"
val MountName = "mydata3"

val ratingsData = sc.textFile(s"s3a://$AccessKey:$EncodedSecretKey@$AwsBucketName/ratings.dat")
val userMovieRatings = ratingsData.map(_.split("::")).map(m => m(0) + "," + m(1)+","+m(2)).
map(_.split(",") match { case Array(user, movie, rating) =>
  Rating(user.toInt, movie.toInt, rating.toDouble)
})

//userMovieRatings.collect().foreach(println)

val rank = 5
val numIter = 15
val splits = userMovieRatings.randomSplit(Array(0.6, 0.4))
val (trainingData, testData) = (splits(0), splits(1))
val ALSmodel = ALS.train(trainingData, rank, numIter, 0.01)

val usersMoviePair = testData.map { case Rating(user, movie, rate) =>
  (user, movie)
}
val predictedRating =
  ALSmodel.predict(usersMoviePair).map { case Rating(user, movie, rate) =>
    ((user, movie), rate)
  }
val RatingJOINPreds = testData.map { case Rating(user, movie, rate) =>
  ((user, movie), rate)
}.join(predictedRating)

val MeanSqError = RatingJOINPreds.map { case ((user, movie), (rating1, rating2)) =>
  val err = (rating1 - rating2)
  err * err
}.mean()
println("Mean Squared Error: "+MeanSqError)
println("Accuracy = " + (1-MeanSqError))
