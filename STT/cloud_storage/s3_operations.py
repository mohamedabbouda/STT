import os
class S3Sync:

    def sync_folder_to_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync  {aws_bucket_url} {folder} "
        os.system(command)
     
S3Sync().sync_folder_to_s3("artifacts/12_04_2024_09_33_49/model_trainer_artifact","s3://speech-to-text-data-project/data/")
        