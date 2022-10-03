package assignment21
// Mehmet AYDIN
//additional libraries
import org.apache.spark.sql.functions.{col, lit, when}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.ArrayBuffer
//

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}




import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
// import org.apache.spark.ml.clustering.{KMeans, KMeansSummary}


import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range

object assignment  {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)
                       
  
  val spark = SparkSession.builder()
              	            .appName("assignment")
                          	.config("spark.driver.host", "localhost")
                          	.master("local")
                          	.getOrCreate()

                          	
  // Reading the dataK5D2 data set,
	val dataK5D2 : DataFrame = spark.read.option("delimiter", ",")
                             .option("inferSchema", "true")
                             .option("header", "true")
                             .csv("data/dataK5D2.csv")
  
  // Reading the dataK5D3 data set,  
	val dataK5D3 : DataFrame = spark.read.option("delimiter", ",")
                            	.option("inferSchema", "true")
                            	.option("header", "true")
                            	.csv("data/dataK5D3.csv")
	
  // Extracting the dataK5D3 data set: add a new column with ones and zeros
	// 0 refers to  Fatal labels and 1 refers to Ok labels
	val dataK5D3WithLabels= dataK5D2.withColumn("c", when(col("LABEL").contains("Fatal"),0.toDouble)
					                    .otherwise(1.toDouble))

	
	// TASK1 METHOD				                    
	def task1(df: DataFrame, k: Int): Array[(Double, Double)] = {
    
      // Define the means array of vectors with k elements.
			var means:Array[(Double, Double)] = new Array[(Double, Double)](k)
			val numIterations = 99
			
			// Map the data to an RDD which is to be used by KMeans framework
			val parsedData = df.rdd.map(s => Vectors.dense(s.getDouble(0),s.getDouble(1))).cache()     
			
			// Train the model and get the cluster centers
			val model = KMeans.train(parsedData, k, numIterations)
			
			var ar = model.clusterCenters    

			// Write the values to means array we created and return it
			for(w <- 0 until k){
				means(w) = (ar(w)(0), ar(w)(1))
			}    
      means
	}

  // TASK2 METHOD	
	def task2(df: DataFrame, k: Int): Array[(Double, Double, Double)] = {
			
	    // Define the means array of vectors with k elements.
	    var means:Array[(Double, Double, Double)] = new Array[(Double, Double, Double)](k)
			val numIterations = 99

			var df2 = df.select("a", "b", "c")
			
			// Map the data to an RDD which is to be used by KMeans framework
			val parsedData = df2.rdd.map(s => Vectors.dense(s.getDouble(0),s.getDouble(1),s.getDouble(2))).cache()    
			
			// Train the model and get the cluster centers
			val model = KMeans.train(parsedData, k, numIterations)
			var ar = model.clusterCenters    

			// Write the values to means array we created and return it
			for(w <- 0 until k){
				means(w) = (ar(w)(0), ar(w)(1), ar(w)(2))
			}    
	    means
	}

	// TASK3 METHOD	
	def task3(df: DataFrame, k: Int): Array[(Double, Double)] = {
	  
	    // Define a cluster array of vectors with k elements
			var clusters:Array[(Double, Double)] = new Array[(Double, Double)](k)  

			// Call the previous function that returns 3-dimensional cluster centers.
			var means = task2(df, k)
			val arrayBuf = ArrayBuffer[(Double, Double)]()
			
			// Fill the arrayBuf with the computed means
			for(w <- 0 until k){
				if (means(w)._3 < 0.5){				  
					arrayBuf += ((means(w)._1, means(w)._2))
				}
			}
	    clusters = arrayBuf.toArray  // turn back to an Array and return the clusters array
			clusters
	}


	// TASK4 METHOD	
	def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)] = {
			
	    // First, define a result array of vectors which is to be returned
	    var res:Array[(Int, Double)] = new Array[(Int, Double)](high - low + 1)
			
	    // Map the DataFrame object to an RDD object so that it can be used in KMeans framework
	    val parsedData = df.rdd.map(s => Vectors.dense(s.getDouble(0),s.getDouble(1))).cache()
					
	    // Filling the result array with the computed costs of clustering with several cluster values.
			for ( i <- 0 until (high - low + 1)) {
				val model = KMeans.train(parsedData, low+i, 99)
				val WSSSE = model.computeCost(parsedData)
				var k = (low + i, WSSSE)  // (cluster number, cost with that cluster number)
				res(i) = k
			}
			res  // return the result array
	}
}


