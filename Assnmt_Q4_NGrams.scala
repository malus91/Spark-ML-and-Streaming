/**
  * Created by Malini Bhaskaran on 11/16/2016.
  */
import stemmer.Stemmer
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka.KafkaUtils

object NGrams {

  def main(args: Array[String]) {
    val (zkQuorum, group, topics, numThreads) = ("localhost", "localhost", "NGramsTopic", "20")
    val sparkConf = new SparkConf().setMaster("local[*]").setSparkHome("/usr/local/spark").setAppName("BiGrams")
    val ssc = new StreamingContext(sparkConf, Seconds(10))
    ssc.checkpoint("checkpoint")

    val topicMap = topics.split(",").map((_, numThreads.toInt)).toMap
    val inputTopicDS = KafkaUtils.createStream(ssc, zkQuorum, group, topicMap).map(_._2)

    val ObjStemmer = new Stemmer
    val stopWordsPattern = "(\\ba\\b|\\bable\\b|\\babout\\b|\\bacross\\b|\\bafter\\b|\\ball\\b|\\balmost\\b|\\balso\\b|\\bam\\b|\\bamong\\b|\\ban\\b|\\band\\b|\\bany\\b|\\bare\\b|\\bas\\b|\\bat\\b|\\bbe\\b|\\bbecause\\b|\\bbeen\\b|\\bbut\\b|\\bby\\b|\\bcan\\b|\\bcannot\\b|\\bcould\\b|\\bdear\\b|\\bdid\\b|\\bdo\\b|\\bdoes\\b|\\beither\\b|\\belse\\b|\\bever\\b|\\bevery\\b|\\bfor\\b|\\bfrom\\b|\\bget\\b|\\bgot\\b|\\bhad\\b|\\bhas\\b|\\bhave\\b|\\bhe\\b|\\bher\\b|\\bhers\\b|\\bhim\\b|\\bhis\\b|\\bhow\\b|\\bhowever\\b|\\bi\\b|\\bif\\b|\\bin\\b|\\binto\\b|\\bis\\b|\\bit\\b|\\bits\\b|\\bjust\\b|\\bleast\\b|\\blet\\b|\\blike\\b|\\blikely\\b|\\bmay\\b|\\bme\\b|\\bmight\\b|\\bmost\\b|\\bmust\\b|\\bmy\\b|\\bneither\\b|\\bno\\b|\\bnor\\b|\\bnot\\b|\\bof\\b|\\boff\\b|\\boften\\b|\\bon\\b|\\bonly\\b|\\bor\\b|\\bother\\b|\\bour\\b|\\bown\\b|\\brather\\b|\\bsaid\\b|\\bsay\\b|\\bsays\\b|\\bshe\\b|\\bshould\\b|\\bsince\\b|\\bso\\b|\\bsome\\b|\\bthan\\b|\\bthat\\b|\\bthe\\b|\\btheir\\b|\\bthem\\b|\\bthen\\b|\\bthere\\b|\\bthese\\b|\\bthey\\b|\\bthis\\b|\\btis\\b|\\bto\\b|\\btoo\\b|\\btwas\\b|\\bus\\b|\\bwants\\b|\\bwas\\b|\\bwe\\b|\\bwere\\b|\\bwhat\\b|\\bwhen\\b|\\bwhere\\b|\\bwhich\\b|\\bwhile\\b|\\bwho\\b|\\bwhom\\b|\\bwhy\\b|\\bwill\\b|\\bwith\\b|\\bwould\\b|\\byet\\b|\\byou\\b|\\byour\\b)".r

    val bigramsWithCount = inputTopicDS.map{
      _.split('.').map{ substrings =>
        substrings.trim.split(' ').
          map{_.replaceAll("""\W""", "").toLowerCase()}. map(a=>stopWordsPattern.replaceAllIn(a, "be")).map(v=>{ObjStemmer.add(v.toArray,v.length)
          ObjStemmer.stem()
          ObjStemmer.toString}).
          sliding(2)
      }.flatMap{identity}.map{_.mkString(" ")}.
        groupBy{identity}.mapValues{_.size}
    }. flatMap{identity}.reduceByKey(_+_)

    val finalTop = bigramsWithCount.filter(count=>count._2>=2)
    finalTop.print()
    ssc.start()
    ssc.awaitTermination()
  }
}
