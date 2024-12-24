from flask import Flask, request, jsonify
from STT.pipeline.prediction_pipeline import Prediction
from STT.entity.config_entity import PredictionPipelineConfig
from STT.cloud_storage.s3_operations import S3Sync
import os
import wave
from pydub import AudioSegment  # Library to handle audio conversions
import mlflow

app = Flask(__name__)

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  
mlflow.set_experiment("SpeechToTextExperiment")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        config = PredictionPipelineConfig()
        os.makedirs(config.prediction_artifact_dir, exist_ok=True)
        os.makedirs(config.app_artifacts, exist_ok=True)

        s3_op = S3Sync()
        s3_op.sync_folder_from_s3(folder=config.model_download_path, aws_bucket_url=config.s3_model_path)

        if 'audio' not in request.files:
            return jsonify({"error": "Audio file is required"}), 400

        audio_file = request.files['audio']
        input_file_path = os.path.join(config.prediction_artifact_dir, audio_file.filename)
        audio_file.save(input_file_path)

        # Convert audio to WAV format if not already in WAV
        wav_file_path = os.path.splitext(input_file_path)[0] + ".wav"
        if not input_file_path.lower().endswith(".wav"):
            audio = AudioSegment.from_file(input_file_path)
            audio.export(wav_file_path, format="wav")
        else:
            wav_file_path = input_file_path

        # Ensure WAV file is valid
        try:
            with wave.open(wav_file_path, 'rb') as wf:
                wf.readframes(1)  # Read a frame to ensure file validity
        except wave.Error as e:
            return jsonify({"error": "Invalid WAV file: " + str(e)}), 400

        # Perform transcription
        with mlflow.start_run(run_name="PredictionPipelineRun"):
            mlflow.log_param("input_file", wav_file_path)
            pred = Prediction(wav_file_path, config.model_download_path)
            transcript = pred.prediction()

            mlflow.log_param("file_format", "wav")
            mlflow.log_metric("prediction_success", 1)

            return jsonify({"transcript": transcript})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)