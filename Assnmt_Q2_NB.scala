import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

//Databricks specific settings
val AccessKey = ""
val SecretKey = ""
val EncodedSecretKey = SecretKey.replace("/", "%2F")
val AwsBucketName = "assnmtdata"
val MountName = "mydata3"
val glassData = sc.textFile(s"s3a://$AccessKey:$EncodedSecretKey@$AwsBucketName/glass.data")

val glassDataMap = glassData.map { line =>
                   val attributes = line.split(',')
                   LabeledPoint(attributes(10).toDouble, Vectors.dense(attributes(0).toDouble,attributes(1).toDouble,attributes(2).toDouble,attributes(3).toDouble,attributes(4).toDouble,attributes(5).toDouble,attributes(6).toDouble,attributes(7).toDouble,attributes(8).toDouble,attributes(9).toDouble))
}

val Array(trainData, testData) =glassDataMap.randomSplit(Array(0.6, 0.4))

val NBmodel = NaiveBayes.train(trainData, lambda = 1.0, modelType = "multinomial")

val LabelPreds = testData.map(p => (NBmodel.predict(p.features), p.label))

val accuracy = 1.0 * LabelPreds.filter(x => x._1 == x._2).count() / test.count()
println("Accuracy of Naive Bayesian Model: " + accuracy)