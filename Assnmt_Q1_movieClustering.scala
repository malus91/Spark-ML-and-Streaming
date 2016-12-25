import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

//Databricks specific settings
val AccessKey = ""
val SecretKey = ""
val EncodedSecretKey = SecretKey.replace("/", "%2F")
val AwsBucketName = "assnmtdata"
val MountName = "mydata3"
val movieUserMatx = sc.textFile(s"s3a://$AccessKey:$EncodedSecretKey@$AwsBucketName/itemusermat")
val movieDat = sc.textFile(s"s3a://$AccessKey:$EncodedSecretKey@$AwsBucketName/movies.dat")
val k=10; 
val iter =20;
def getRatingsPerMovie(line: String): (Long, Array[String]) = {
        val ratingVals = line.split(" ")
        val movieID = ratingVals(0).toLong
        val mRatings = ratingVals.drop(1)
        return (movieID, mRatings)
  }
val ratingData = movieUserMatx.map(line => Vectors.dense(line.split(" ").drop(1).map(_.toDouble))).cache()
val ratingsPerMovie= movieUserMatx.map(line => getRatingsPerMovie(line)).map(item => (item._1, Vectors.dense(item._2.map(_.toDouble))))
val K_MeansClusterModel = KMeans.train(ratingData, k, iter)        
val mClusters =  ratingsPerMovie.map(item => (item._1, K_MeansClusterModel.predict(item._2)))     

val movies = movieDat.map(line => line.split("::")).map(item => (item(0).toLong, item.mkString(",")))        
val moviesCluster = movies.join(mClusters).map(item => item.swap).map(item => (item._1._2, item._1._1)).reduceByKey(_+"%"+_)

val resultClusters = moviesCluster.map(item => (item._1, item._2.split("%").take(5).mkString("\n\t"))).sortBy(_._1, true).map(item => ("Cluster" + item._1 + "\n\t" + item._2))

resultClusters.collect.foreach(println)