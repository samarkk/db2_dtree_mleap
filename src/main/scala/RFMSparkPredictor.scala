import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import scala.util.Random

object RFMSparkPredictor extends App {

  val spark = SparkSession.builder()
    .appName("SparkRFMPredictor")
    .master("local[*]")
    .getOrCreate()
  val sc = spark.sparkContext
  sc.setLogLevel("ERROR")

  import spark.implicits._

  val fileLoc = "file:///home/samar/Downloads/covtype20.data"

  val dataWithoutHeader = spark.read.
    option("inferSchema", true).
    option("header", false).
    csv(fileLoc)

  val schemasp = dataWithoutHeader.schema
  val tpoint = "2590,56,2,212,-6,390,220,235,151,6225,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0"
  val spds = sc.parallelize(List(tpoint)).toDS
  val spdf = spark.read.schema(schemasp).csv(spds)
  val colNames = Seq(
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points") ++ (
    (0 until 4).map(i => s"Wilderness_Area_$i")) ++ (
    (0 until 40).map(i => s"Soil_Type_$i")) ++ Seq("Cover_Type")
  val spdata = spdf.toDF(colNames: _*).
    withColumn("Cover_Type", $"Cover_Type".cast("double"))

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

  val mloc = "file:///home/samar/ppmodel"
  val ppmodel = PipelineModel.load(mloc)

  ppmodel.transform(unencodeOneHot(spdata)).show
  ppmodel.transform(unencodeOneHot(spdata)).select("rawprediction", "probability", "prediction").show(false)

}
