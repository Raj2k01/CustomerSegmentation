import json
import os
import logging

import boto3
import joblib
import pandas as pd

# -------------------------------------------------
# Logging
# -------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# -------------------------------------------------
# Environment Variable
# -------------------------------------------------
BUCKET_NAME = os.environ["MODEL_BUCKET"]

# -------------------------------------------------
# Local file paths inside Lambda
# -------------------------------------------------
MODEL_PATH = "/tmp/kmeans.pkl"
SCALER_PATH = "/tmp/scaler.pkl"
ENCODER_PATH = "/tmp/gender_encoder.pkl"

# -------------------------------------------------
# S3 Client
# -------------------------------------------------
s3 = boto3.client("s3")


# -------------------------------------------------
# Download model files only once
# -------------------------------------------------
def download_models():

    if not os.path.exists(MODEL_PATH):

        logger.info("Downloading model files from S3...")

        s3.download_file(
            BUCKET_NAME,
            "kmeans.pkl",
            MODEL_PATH
        )

        s3.download_file(
            BUCKET_NAME,
            "scaler.pkl",
            SCALER_PATH
        )

        s3.download_file(
            BUCKET_NAME,
            "gender_encoder.pkl",
            ENCODER_PATH
        )

        logger.info("Download completed.")


download_models()

# -------------------------------------------------
# Load models
# -------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

logger.info("Models loaded successfully.")

# -------------------------------------------------
# Cluster Names
# -------------------------------------------------
SEGMENTS = {
    0: "Budget Customer",
    1: "Regular Customer",
    2: "High Value Customer",
    3: "Premium Customer",
    4: "Luxury Customer"
}


# -------------------------------------------------
# Lambda Handler
# -------------------------------------------------
def lambda_handler(event, context):

    try:

        body = json.loads(event["body"])

        gender = body["Gender"]
        age = body["Age"]
        income = body["Annual Income (k$)"]
        spending = body["Spending Score (1-100)"]

        df = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Annual Income (k$)": income,
            "Spending Score (1-100)": spending
        }])

        df["Gender"] = encoder.transform(df["Gender"])

        scaled = scaler.transform(df)

        scaled_df = pd.DataFrame(
            scaled,
            columns=df.columns
        )

        cluster = int(model.predict(scaled_df)[0])

        logger.info(f"Prediction Successful : Cluster {cluster}")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "cluster": cluster,
                "segment": SEGMENTS[cluster]
            })
        }

    except Exception as e:

        logger.error(str(e))

        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e)
            })
        }