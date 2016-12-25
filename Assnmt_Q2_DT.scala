import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
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
//glassDataMap.collect().foreach(println)
val Array(train, test) = glassDataMap.randomSplit(Array(0.6, 0.4))
val numLabels = 8
val catFeatureInfo = Map[Int, Int]()
val impurity = "gini"
val maxDepth = 5
val maxBins = 32
val DTmodel = DecisionTree.trainClassifier(train, numLabels, catFeatureInfo,impurity, maxDepth, maxBins)

val LabelPreds = test.map { point =>
     val prediction = DTmodel.predict(point.features)
  (point.label, prediction)
}
val testErr = LabelPreds.filter(reslt => reslt._1 != reslt._2).count().toDouble / test.count()
println(" Accuracy of Decision Tree Model = " + (1-testErr))