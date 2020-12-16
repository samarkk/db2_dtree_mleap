import org.apache.spark.ml.classification.{DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.bundle.SparkBundleContext
import ml.combust.bundle.serializer.SerializationFormat
import org.apache.spark.ml.{Pipeline, PipelineModel}
import resource.managed

object DecisionTreeMLeap extends App {
  val spark = SparkSession.builder()
    .appName("RFPipelineModelPackaging")
    .master("local").getOrCreate()

  spark.conf.set("driver-memory", "4G")
  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val fileLoc = "file:///home/samar/Downloads/covtype.data"
  val modelLoc = "file:///home/samar/ppmodel"
  //Covtype dataset publicly available dataset provides information on
  //types of forest-covering parcels of land in Colorado, USA
  val dataWithoutHeader = spark.read.
    option("inferSchema", true).
    option("header", false).
    csv(fileLoc)

  // columns 10 to 14 are for wilderness_area and next 40 columns for soil type
  val colNames = Seq(
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points") ++ (
    (0 until 4).map(i => s"Wilderness_Area_$i")) ++ (
    (0 until 40).map(i => s"Soil_Type_$i")) ++ Seq("Cover_Type")

  // lets create the data frame with column names
  // and cast the label that we have to forecast to double
  val data = dataWithoutHeader.toDF(colNames: _*).
    withColumn("Cover_Type", $"Cover_Type".cast("double"))
  val Array(trainData, testData) = data.randomSplit(Array(0.9, 0.1), 100L)
  val savedRFM = PipelineModel.load(modelLoc)

  def unencodeOneHot(data: DataFrame): DataFrame = {
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

    val wildernessAssembler = new VectorAssembler().
      setInputCols(wildernessCols).
      setOutputCol("wilderness")
    // we have a udf here which is going to find the item which is 1.0 in the 4 columns
    // for wilderness and the 40 for the soil
    val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

    //here we employ it to drop the wilderness columns and replace with a single one
    val withWilderness = wildernessAssembler.transform(data).
      drop(wildernessCols: _*).
      withColumn("wilderness", unhotUDF($"wilderness"))

    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

    val soilAssembler = new VectorAssembler().
      setInputCols(soilCols).
      setOutputCol("soil")

    soilAssembler.transform(withWilderness).
      drop(soilCols: _*).
      withColumn("soil", unhotUDF($"soil"))
  }

  val unencData = unencodeOneHot(trainData)

  val sbc = SparkBundleContext().withDataset(savedRFM.transform(unencData))
  //  val sbc = SparkBundleContext().withDataset(savedRFM.transform(pipeline.transform(unencData)))
  for (bf <- managed(BundleFile("jar:file:/home/samar/dtree-spark-pipeline.zip"))) {
    savedRFM.writeBundle.save(bf)(sbc).get
  }
  for (bf <- managed(BundleFile("jar:file:/home/samar/dtree-json-spark-pipeline.zip"))) {
    savedRFM.writeBundle.format(SerializationFormat.Json) save (bf)
  }
}