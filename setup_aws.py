
import boto3
import os
import time
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
S3_BUCKET = os.environ.get("CROWDAI_S3_BUCKET", "crowdai-models-unique")
DYNAMODB_TABLE = os.environ.get("CROWDAI_DYNAMO_TABLE", "crowdai-predictions")

def setup_s3():
    print(f"📦 Setting up S3 bucket: {S3_BUCKET}...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"   ✅ Bucket '{S3_BUCKET}' already exists.")
    except:
        try:
            if AWS_REGION == "us-east-1":
                s3.create_bucket(Bucket=S3_BUCKET)
            else:
                s3.create_bucket(
                    Bucket=S3_BUCKET,
                    CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
                )
            print(f"   ✅ Created bucket '{S3_BUCKET}'.")
        except Exception as e:
            print(f"   ❌ Error creating bucket: {e}")

def setup_dynamodb():
    print(f"🗄️  Setting up DynamoDB table: {DYNAMODB_TABLE}...")
    db = boto3.client("dynamodb", region_name=AWS_REGION)
    try:
        db.describe_table(TableName=DYNAMODB_TABLE)
        print(f"   ✅ Table '{DYNAMODB_TABLE}' already exists.")
    except db.exceptions.ResourceNotFoundException:
        try:
            db.create_table(
                TableName=DYNAMODB_TABLE,
                KeySchema=[
                    {"AttributeName": "zone_id", "KeyType": "HASH"},
                    {"AttributeName": "timestamp", "KeyType": "RANGE"}
                ],
                AttributeDefinitions=[
                    {"AttributeName": "zone_id", "AttributeType": "S"},
                    {"AttributeName": "timestamp", "AttributeType": "S"}
                ],
                BillingMode="PAY_PER_REQUEST"
            )
            print(f"   ⏳ Creating table '{DYNAMODB_TABLE}' (this may take a minute)...")
            waiter = db.get_waiter("table_exists")
            waiter.wait(TableName=DYNAMODB_TABLE)
            print(f"   ✅ Table '{DYNAMODB_TABLE}' is now active.")
        except Exception as e:
            print(f"   ❌ Error creating table: {e}")

if __name__ == "__main__":
    print("🚀 Starting AWS Resource Setup...")
    setup_s3()
    setup_dynamodb()
    print("\n✅ AWS Setup Complete! Your dashboard will now have full cloud features.")
