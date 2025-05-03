package spark.streaming.tp22;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.StreamingQueryException;
import org.apache.spark.sql.streaming.Trigger;

import java.util.concurrent.TimeoutException;
import java.util.Arrays;

public class Stream {
    public static void main(String[] args) throws StreamingQueryException, TimeoutException  {
        SparkSession spark = SparkSession
            .builder()
            .appName("NetworkWordCount")
            .master("local[*]")
            .getOrCreate();

        // Create DataFrame representing the stream of input lines from connection to localhost:9999
        Dataset<String> lines = spark
            .readStream()
            .format("socket")
            .option("host", "localhost")
            .option("port", 9999)
            .load()
            .as(Encoders.STRING());

        // Split the lines into words
        Dataset<String> words = lines.flatMap(
            (String x) -> Arrays.asList(x.split(" ")).iterator(),
            Encoders.STRING());

        // Generate running word count
        Dataset<org.apache.spark.sql.Row> wordCounts = words.groupBy("value").count();

        // Start running the query that prints the running counts to the console
        StreamingQuery query = wordCounts.writeStream()
            .outputMode("complete")
            .format("console")
            .trigger(Trigger.ProcessingTime("1 second"))
            .start();

        query.awaitTermination();
    }
}

