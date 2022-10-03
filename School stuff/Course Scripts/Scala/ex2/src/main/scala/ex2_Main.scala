// Mehmet AYDIN

import org.apache.spark.SparkConf

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{sum, min, max, asc, desc, udf}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import com.databricks.spark.xml._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}

import java.lang.Thread

import org.apache.log4j.Logger
import org.apache.log4j.Level

object main extends App {

  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)

	val spark = SparkSession.builder()
                          .appName("ex2")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()

  spark.conf.set("spark.sql.shuffle.partitions", "5")

  // Task 1: File "data/rdu-weather-history.csv" contains weather data in csv format. 
  //         Study the file and read the data into DataFrame weatherDataFrame.
  //         Let Spark infer the schema. Study the schema.
  val weatherDataFrame: DataFrame = spark.read 
                                         .format("csv") 
                                         .option("delimiter", ";") 
                                         .option("header", "true") 
                                         .option("inferSchema", "true") 
                                         .load("data/*.csv")
                                         
  // Study the schema of the DataFrame:                                       
  weatherDataFrame.printSchema()
  
  
  
  // Task 2: print three first elements of the data frame to stdout
 val weatherSample: Array[Row] = weatherDataFrame.take(3)
 weatherSample.foreach(println) 
    
  
  
  // Task 3: Find min and max temperatures from the whole DataFrame
  val minTempArray: Array[Row] = weatherDataFrame.select(min("temperaturemin")).collect() 
  val maxTempArray: Array[Row] = weatherDataFrame.select(max("temperaturemax")).collect()
  minTempArray.foreach(println) 
  maxTempArray.foreach(println) 
  
  
  
  
  // Task 4: Add a new column "year" to the weatherDataFrame. 
  // The type of the column is integer and value is calculated from column "date".
  // You can use function year from org.apache.spark.sql.functions
  // See documentation https://spark.apache.org/docs/2.3.0/api/sql/index.html#year
  import org.apache.spark.sql.functions.year
  val weatherDataFrameWithYear = weatherDataFrame.withColumn("year", year(col("date"))) 
  weatherDataFrameWithYear.printSchema()
  
  
  
  // Task 5: Find min and max for each year
  val aggregatedDF: DataFrame = weatherDataFrameWithYear.groupBy(col("year")) 
                                                        .agg(min("temperaturemin"), 
                                                             max("temperaturemax")) 
                                                        .orderBy(col("year"))
  
  aggregatedDF.printSchema()
  aggregatedDF.collect().foreach(println) 
  
  
  
}