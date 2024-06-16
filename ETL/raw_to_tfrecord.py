import sys
import boto3 # type: ignore
import zipfile
import io
from PIL import Image
import tensorflow as tf
from awsglue.utils import getResolvedOptions # type: ignore
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse job parameters
try:
    args = getResolvedOptions(sys.argv, ['source_bucket', 'zip_key', 'target_bucket', 'output_prefix'])
except Exception as e:
    logger.error("Error parsing arguments: %s", e)
    sys.exit(2)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def image_to_tfrecord(img, bboxes):
    """Convert an image and its bounding boxes to a TFRecord"""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    xmins = [bbox['xmin'] for bbox in bboxes]
    xmaxs = [bbox['xmax'] for bbox in bboxes]
    ymins = [bbox['ymin'] for bbox in bboxes]
    ymaxs = [bbox['ymax'] for bbox in bboxes]
    labels = [bbox['label'] for bbox in bboxes]

    feature = {
        'image': _bytes_feature(img_byte_arr),
        'xmins': _float_feature(xmins),
        'xmaxs': _float_feature(xmaxs),
        'ymins': _float_feature(ymins),
        'ymaxs': _float_feature(ymaxs),
        'labels': _int64_feature(labels)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

def process_images_to_tfrecord(source_bucket, zip_key, target_bucket, output_prefix):
    try:
        s3 = boto3.client('s3')

        # Download the zip file
        zip_obj = s3.get_object(Bucket=source_bucket, Key=zip_key)
        buffer = io.BytesIO(zip_obj["Body"].read())

        tfrecord_filename = '/tmp/data.tfrecord'
        bboxes = {}
        location_txt_found = False

        with zipfile.ZipFile(buffer, 'r') as zip_ref:
            # Check if location.txt exists in the archive, including subdirectories
            for file_info in zip_ref.infolist():
                if file_info.filename.endswith('location.txt'):
                    location_txt_found = True
                    with zip_ref.open(file_info.filename) as label_file:
                        for line in label_file:
                            parts = line.decode('utf-8').strip().split()
                            if len(parts) == 6:
                                image_name, class_id, x, y, width, height = parts
                                if image_name not in bboxes:
                                    bboxes[image_name] = []
                                bboxes[image_name].append({
                                    'xmin': float(x),
                                    'ymin': float(y),
                                    'xmax': float(x) + float(width),
                                    'ymax': float(y) + float(height),
                                    'label': int(class_id)
                                })
                    break

            if not location_txt_found:
                logger.error("location.txt not found in the archive.")
                return

            with tf.io.TFRecordWriter(tfrecord_filename) as writer:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith(('.png', '.jpg', '.jpeg')):
                        with zip_ref.open(file_info.filename) as file:
                            img = Image.open(file)

                            # Get bounding boxes for the image
                            image_bboxes = bboxes.get(file_info.filename.split('/')[-1], [])

                            # Convert image and bounding boxes to TFRecord
                            tfrecord = image_to_tfrecord(img, image_bboxes)
                            writer.write(tfrecord)

        # Upload TFRecord file to S3
        s3.upload_file(tfrecord_filename, target_bucket, f'{output_prefix}/data.tfrecord')
    except Exception as e:
        logger.error("Error processing images: %s", e)
        sys.exit(2)

def main():
    if not all(k in args for k in ('source_bucket', 'zip_key', 'target_bucket', 'output_prefix')):
        logger.error("Missing required arguments: --source_bucket, --zip_key, --target_bucket, --output_prefix")
        sys.exit(2)
        
    source_bucket = args['source_bucket']
    zip_key = args['zip_key']
    target_bucket = args['target_bucket']
    output_prefix = args['output_prefix']
    
    process_images_to_tfrecord(source_bucket, zip_key, target_bucket, output_prefix)

if __name__ == "__main__":
    main()
