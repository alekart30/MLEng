import pyspark
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.ml.feature import Imputer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder.getOrCreate()
seed = 30

def pandas_postprocessing(df, numerical_features_scaled, categorical_features_vectorized):
    """
    Finalizes preprocessing using pandas DF manipulations
    """
    df_copy = df.copy()
    for feature in numerical_features_scaled:
        df_copy[feature] = df_copy[feature].apply(lambda x: x[0])
        
    sub_df_list = []
    for feature in categorical_features_vectorized:
        sub_df = pd.get_dummies(df_copy[feature], drop_first=True)
        sub_df.columns = [feature+"_"+str(idx) for idx in range(len(sub_df.columns))]
        sub_df_list.append(sub_df)
        
    return pd.concat(
        [pd.concat(sub_df_list, axis=1),
        df_copy.loc[:, numerical_features_scaled]],
        axis=1
    )

def run_etl():
    # read source data
    df = spark.read.option("header", True)\
        .csv("data/bank_scoring.csv", inferSchema=True)

    # categorize features
    numerical_features = [feature for (feature, feature_type) in df.dtypes if feature_type in ['int', 'float']]
    categorical_features = [feature for (feature, feature_type) in df.dtypes if feature_type in ['string']]
    numerical_features.remove("default")

    # stratified train/test split
    df = df.withColumn("idx", f.monotonically_increasing_id())
    train_df = df.sampleBy("default", fractions={0: 0.8, 1: 0.8}, seed=seed)
    test_df = df.join(train_df, on='idx', how='leftanti')

    # impute NAs for numerical features
    numerical_features_imputed = [feature+"_imputed" for feature in numerical_features]
    imputer = Imputer(inputCols=numerical_features, outputCols=numerical_features_imputed, strategy='median')

    model = imputer.fit(train_df)
    train_df_num_imputed = model.transform(train_df).select(numerical_features_imputed)
    test_df_num_imputed = model.transform(test_df).select(numerical_features_imputed)

    # impute NAs for categorical features
    categorical_features_imputed = [feature+"_imputed" for feature in categorical_features]
    mapping = dict(zip(categorical_features, categorical_features_imputed))

    train_df_cat_imputed = train_df.select(categorical_features).fillna(value='UNK')
    test_df_cat_imputed = test_df.select(categorical_features).fillna(value='UNK')

    train_df_cat_imputed = train_df_cat_imputed.select([f.col(c).alias(mapping.get(c, c)) for c in train_df_cat_imputed.columns])
    test_df_cat_imputed = test_df_cat_imputed.select([f.col(c).alias(mapping.get(c, c)) for c in test_df_cat_imputed.columns])

    # scale numerical features
    numerical_features_scaled = [feature[:-8] + "_scaled" for feature in numerical_features_imputed]
    assemblers = [VectorAssembler(inputCols=[col], outputCol=col[:-8] + "_vec") for col in numerical_features_imputed]
    scalers = [MinMaxScaler(inputCol=col[:-8] + "_vec", outputCol=col[:-8] + "_scaled") for col in numerical_features_imputed]
    pipeline = Pipeline(stages=assemblers + scalers)

    scalerModel = pipeline.fit(train_df_num_imputed)
    train_df_num_scaled = scalerModel.transform(train_df_num_imputed).select(numerical_features_scaled)
    test_df_num_scaled = scalerModel.transform(test_df_num_imputed).select(numerical_features_scaled)

    # encode categorical features
    categorical_features_vectorized = [feature[:-8] + "_vec" for feature in categorical_features_imputed]
    indexers = [StringIndexer(inputCol=col, outputCol=col[:-8] + "_idx") for col in categorical_features_imputed]
    ohes = [OneHotEncoder(inputCol=col[:-8] + "_idx", outputCol=col[:-8] + "_vec") for col in categorical_features_imputed]
    pipeline = Pipeline(stages=indexers + ohes)

    oheModel = pipeline.fit(train_df_cat_imputed)
    train_df_cat_vectorized = oheModel.transform(train_df_cat_imputed).select(categorical_features_vectorized)
    test_df_cat_vectorized = oheModel.transform(test_df_cat_imputed).select(categorical_features_vectorized)

    # obtain joined df in Pandas
    train_df_cat_vectorized = train_df_cat_vectorized.withColumn("idx", f.monotonically_increasing_id())
    train_df_num_scaled = train_df_num_scaled.withColumn("idx", f.monotonically_increasing_id())

    test_df_cat_vectorized = test_df_cat_vectorized.withColumn("idx", f.monotonically_increasing_id())
    test_df_num_scaled = test_df_num_scaled.withColumn("idx", f.monotonically_increasing_id())

    train_df_preprocessed = train_df_cat_vectorized.join(train_df_num_scaled, on='idx').drop('idx').toPandas()
    test_df_preprocessed = test_df_cat_vectorized.join(train_df_num_scaled, on='idx').drop('idx').toPandas()

    # postprocess
    train_df_final = pandas_postprocessing(train_df_preprocessed, numerical_features_scaled, categorical_features_vectorized)
    test_df_final = pandas_postprocessing(test_df_preprocessed, numerical_features_scaled, categorical_features_vectorized)

    # store porcessed features
    train_df_final.to_csv('data/X_train_transformed.csv')
    test_df_final.to_csv('data/X_test_transformed.csv')

    # store target
    train_df.select('default').toPandas().to_csv('data/y_train.csv')
    test_df.select('default').toPandas().to_csv('data/y_test.csv')
