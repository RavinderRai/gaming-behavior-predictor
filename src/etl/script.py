
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import pandas as pd
from io import StringIO
import boto3

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_BUCKET', 'INPUT_KEY', 'OUTPUT_BUCKET', 'OUTPUT_KEY'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read data from S3
s3_client = boto3.client('s3')
obj = s3_client.get_object(Bucket=args['INPUT_BUCKET'], Key=args['INPUT_KEY'])
df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

# Perform transformations
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['GameDifficulty'] = df['GameDifficulty'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})
df_encoded = pd.get_dummies(df, columns=['Location', 'GameGenre'], drop_first=True)

# Convert the DataFrame back to CSV
csv_buffer = StringIO()
df_encoded.to_csv(csv_buffer, index=False)

# Upload the transformed data to S3
s3_client.put_object(Bucket=args['OUTPUT_BUCKET'], Key=args['OUTPUT_KEY'], Body=csv_buffer.getvalue())

job.commit()
