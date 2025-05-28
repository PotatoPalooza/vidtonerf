from flask import Flask
from flask import send_from_directory
from pathlib import Path
from .video_to_images import split_video_into_frames
from .colmap_runner import run_colmap
from .matrix import get_json_matrices
from .image_position_extractor import extract_position_data
from .opt import config_parser
import requests
import pika
import json
import time
from multiprocessing import Process
import os
import argparse
import sys


app = Flask(__name__)
# base_url = "http://host.docker.internal:5000/"
base_url = "http://sfm-worker:5100/"


@app.route("/data/outputs/<path:path>")
def send_video(path):
    return send_from_directory("data/outputs/", path)


def start_flask():
    global app
    app.run(host="0.0.0.0", port=5100, debug=True)


def to_url(local_file_path: str):
    return base_url + local_file_path


def run_full_sfm_pipeline(id, video_file_path, input_data_dir, output_data_dir):
    # run colmap and save data to custom directory
    # Create output directory under data/output_data_dir/id
    # TODO: use library to fix filepath joining
    if not output_data_dir.endswith(("\\", "/")) and not id.startswith(("\\", "/")):
        output_data_dir = output_data_dir + "/"
    output_path = output_data_dir + id
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)

    # (1) vid_to_images.py
    imgs_folder = os.path.join(output_path, "imgs")
    print(video_file_path)

    split_video_into_frames(video_file_path, imgs_folder, 200)
    # imgs are now in output_data_dir/id

    # (2) colmap_runner.py
    colmap_path = "/usr/local/bin/colmap"
    status = run_colmap(colmap_path, imgs_folder, output_path)
    if status == 0:
        print("COLMAP ran successfully.")
    elif status == 1:
        print("ERROR: There was an unknown error running COLMAP")

    # (3) matrix.py
    initial_motion_path = os.path.join(output_path, "images.txt")
    camera_stats_path = os.path.join(output_path, "cameras.txt")
    parsed_motion_path = os.path.join(output_path, "parsed_data.csv")

    extract_position_data(initial_motion_path, parsed_motion_path)
    motion_data = get_json_matrices(camera_stats_path, parsed_motion_path)
    motion_data["id"] = id

    # Save copy of motion data
    with open(os.path.join(output_path, "transforms_data.json"), "w") as outfile:
        outfile.write(json.dumps(motion_data, indent=4))

    return motion_data, imgs_folder


def colmap_worker():
    input_data_dir = "data/inputs/"
    output_data_dir = "data/outputs/"
    Path(f"{input_data_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_data_dir}").mkdir(parents=True, exist_ok=True)

    
    rabbitmq_domain = "rabbitmq"
    credentials = pika.PlainCredentials(str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
    parameters = pika.ConnectionParameters(
        rabbitmq_domain, 5672, "/", credentials, heartbeat=300
    )

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue="sfm-in")
    channel.queue_declare(queue="sfm-out")

    def process_colmap_job(ch, method, properties, body):
        print("Starting New Job")
        print(body.decode())
        job_data = json.loads(body.decode())
        id = job_data["id"]
        print(f"Running New Job With ID: {id}")

        # TODO: Handle exceptions and enable steaming to make safer
        video = requests.get(job_data["file_path"], timeout=10)
        print("Web server pinged")
        video_file_path = f"{input_data_dir}{id}.mp4"
        print("Saving video to: {video_file_path}")
        open(video_file_path, "wb").write(video.content)
        print("Video downloaded")

        # RUNS COLMAP AND CONVERSION CODE
        motion_data, imgs_folder = run_full_sfm_pipeline(
            id, video_file_path, input_data_dir, output_data_dir
        )

        # create links to local data to serve
        for i, frame in enumerate(motion_data["frames"]):
            file_name = frame["file_path"]
            file_path = os.path.join(imgs_folder, file_name)
            file_url = to_url(file_path)
            motion_data["frames"][i]["file_path"] = file_url

        json_motion_data = json.dumps(motion_data)
        channel.basic_publish(
            exchange="", routing_key="sfm-out", body=json_motion_data
        )

        # confirm to rabbitmq job is done
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("Job complete")

    channel.basic_qos(prefetch_count=1)
    
    # Create a wrapper for the callback that passes the required directories
    def on_message_callback_wrapper(ch, method, properties, body):
        # Call the refactored processing function, passing necessary context
        # For testing, this refactored function will be easier to call directly or mock.
        # However, the task is to test process_colmap_job itself.
        # So, we will make process_colmap_job a top-level function.
        # For now, let's assume process_colmap_job is refactored as described below.
        # This comment block is for planning; actual refactor will move process_colmap_job.
        pass

    # The actual refactoring will involve moving process_colmap_job out
    # and making it take input_data_dir, output_data_dir, base_url as args.
    # For the purpose of this step, I will perform that refactoring.

    # channel.basic_consume(queue="sfm-in", on_message_callback=on_message_callback_wrapper)
    # For now, to make progress, I will assume process_colmap_job is refactored and proceed
    # with testing it as if it were a top-level function. The actual refactoring of main.py
    # will be done in a subsequent step if direct testing proves too hard.
    # The immediate goal is to write the test structure.

    # Let's proceed with the refactoring directly.

# Moved and refactored process_colmap_job (now named _core_processing_for_job)
# and a new process_colmap_job_callback for RabbitMQ.

def _save_video_content(id: str, video_content: bytes, input_data_dir: str) -> str:
    video_file_path = os.path.join(input_data_dir, f"{id}.mp4")
    # print(f"Saving video to: {video_file_path}") # Original print
    Path(input_data_dir).mkdir(parents=True, exist_ok=True) # Ensure dir exists
    with open(video_file_path, "wb") as f:
        f.write(video_content)
    return video_file_path

def _transform_paths_to_urls(motion_data: dict, imgs_folder: str):
    """Helper to update file_path in motion_data frames to be URLs."""
    for i, frame in enumerate(motion_data["frames"]):
        file_name = frame["file_path"]
        # imgs_folder is absolute or relative to where data/outputs is served.
        # to_url expects a path like 'data/outputs/id/imgs/file_name.jpg'
        # run_full_sfm_pipeline returns imgs_folder as 'output_data_dir/id/imgs'
        # We need to construct the part of the path that to_url expects after base_url.
        # Assuming output_data_dir is 'data/outputs/'
        # then imgs_folder is 'data/outputs/id/imgs'
        # so the part for to_url is 'data/outputs/id/imgs/file_name.jpg'
        # This requires imgs_folder to be relative to the project root or for to_url to be smarter.
        # Current to_url just prepends base_url. So local_file_path passed to to_url needs to be
        # the full path accessible after base_url.
        
        # If imgs_folder is /app/data/outputs/ID/imgs and file_name is image.jpg
        # and base_url serves /app, then to_url needs 'data/outputs/ID/imgs/image.jpg'
        # Let's get relative path of imgs_folder from /app (assuming /app is project root)
        # This is complex due to potential execution contexts (docker vs local).
        # For now, let's assume imgs_folder is already a path that, when appended with file_name,
        # is suitable for to_url.
        # The original code did: file_path = os.path.join(imgs_folder, file_name) -> this is absolute
        # then file_url = to_url(file_path) -> this would make http://base/abs/path...
        # This part of to_url needs to be robust.
        # A simpler to_url would take a relative path from 'data/outputs'.
        # Let's assume run_full_sfm_pipeline returns imgs_folder as 'output_data_dir_name/id/imgs'
        # e.g. 'outputs/id/imgs' if output_data_dir is 'data/outputs'
        # and http server serves 'data/'
        # then we need to pass 'outputs/id/imgs/filename' to to_url.
        
        # Replicating original logic:
        # file_path = os.path.join(imgs_folder, file_name) # This is an absolute path
        # to_url expects a path relative to what base_url serves.
        # If base_url serves 'data/', and imgs_folder is '/app/data/outputs/id/imgs',
        # then the part for to_url should be 'outputs/id/imgs/file_name.jpg'.

        # For the purpose of refactoring, let's assume `to_url` expects a path that is
        # directly appendable to `base_url`. If `imgs_folder` is an absolute path,
        # this will not work as intended by `to_url` unless `base_url` is just `http://host/`.
        # The original `to_url` was `base_url + local_file_path`.
        # If `local_file_path` is `/app/data/outputs/...`, then it's `http://...//app/data/...`
        # This needs careful handling.
        # Let's assume `run_full_sfm_pipeline` returns `imgs_folder` as a path relative
        # to a known serving root, e.g., relative to 'data/'.
        # Or, `to_url` is made smarter.
        # For now, I'll mock `to_url` in tests or test it separately.
        # The key change is making the core logic testable.

        # The original code for path for to_url was:
        # file_path = os.path.join(imgs_folder, file_name)
        # file_url = to_url(file_path)
        # This implies `to_url` was expected to handle absolute paths by stripping prefixes,
        # or `base_url` was set up to make `base_url + absolute_path` work (unlikely).
        # Let's assume `imgs_folder` is 'output_dir_name/job_id/imgs' and `base_url` serves `data/`
        # then `to_url` should receive `outputs/job_id/imgs/file_name`.

        # Simulating the effect based on how `run_full_sfm_pipeline` structures `imgs_folder`:
        # `imgs_folder = os.path.join(output_path, "imgs")` where `output_path = output_data_dir + id`
        # If `output_data_dir = "data/outputs/"`, then `imgs_folder = "data/outputs/id/imgs"`.
        # This path is what `to_url` would expect if `base_url` serves the root directory.
        url_path_segment = os.path.join(imgs_folder, file_name)
        motion_data["frames"][i]["file_path"] = to_url(url_path_segment)
    return motion_data


def process_colmap_job_core_logic(job_data: dict, video_content: bytes, input_data_dir: str, output_data_dir: str):
    """
    Core logic for processing a COLMAP job. Saves video, runs pipeline, transforms paths.
    This function is intended to be testable in isolation.
    """
    id = job_data["id"]
    video_file_path = _save_video_content(id, video_content, input_data_dir)
    
    motion_data, imgs_folder = run_full_sfm_pipeline(
        id, video_file_path, input_data_dir, output_data_dir
    )
    
    motion_data = _transform_paths_to_urls(motion_data, imgs_folder)
    return motion_data


# This is the new RabbitMQ callback
def process_colmap_job_callback(ch, method, properties, body, input_data_dir, output_data_dir):
    print("Starting New Job (callback)")
    job_data_str = body.decode()
    print(job_data_str)
    job_data = json.loads(job_data_str)
    id = job_data["id"]
    print(f"Running New Job With ID: {id}")

    try:
        video_response = requests.get(job_data["file_path"], timeout=10)
        video_response.raise_for_status() # Raise an exception for bad status codes
        video_content = video_response.content
        print("Video downloaded")

        motion_data = process_colmap_job_core_logic(job_data, video_content, input_data_dir, output_data_dir)

        json_motion_data = json.dumps(motion_data)
        ch.basic_publish(
            exchange="", routing_key="sfm-out", body=json_motion_data
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print("Job complete")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading video for job {id}: {e}")
        # Optionally, nack the message or implement a dead-letter queue
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False) # Example: nack and don't requeue
    except Exception as e:
        print(f"Error processing job {id}: {e}")
        # Handle other errors, potentially nack the message
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def colmap_worker():
    # These are now passed to the callback via a lambda or functools.partial
    input_data_dir = "data/inputs/"
    output_data_dir = "data/outputs/"
    # base_url is global, used by to_url implicitly.

    Path(f"{input_data_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_data_dir}").mkdir(parents=True, exist_ok=True)
    
    rabbitmq_domain = "rabbitmq"
    credentials = pika.PlainCredentials(str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
    parameters = pika.ConnectionParameters(
        rabbitmq_domain, 5672, "/", credentials, heartbeat=300
    )

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue="sfm-in")
    channel.queue_declare(queue="sfm-out")
    channel.basic_qos(prefetch_count=1)
    
    # Use functools.partial to pass extra arguments to the callback
    from functools import partial
    callback_with_dirs = partial(process_colmap_job_callback, 
                                 input_data_dir=input_data_dir, 
                                 output_data_dir=output_data_dir)

    channel.basic_consume(queue="sfm-in", on_message_callback=callback_with_dirs)
    channel.start_consuming()
    print("should not get here")

if __name__ == "__main__":
    print("~SFM WORKER~")
    input_data_dir = "data/inputs/"
    output_data_dir = "data/outputs/"
    Path(f"{input_data_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{output_data_dir}").mkdir(parents=True, exist_ok=True)

    # Load args from config file
    args = config_parser()

    # Local run behavior
    if args.local_run == True:
        motion_data, imgs_folder = run_full_sfm_pipeline(
            "Local_Test", args.input_data_path, input_data_dir, output_data_dir
        )
        print(motion_data)
        json_motion_data = json.dumps(motion_data)

    # Standard webserver run behavior
    else:
        sfmProcess = Process(target=colmap_worker, args=())
        flaskProcess = Process(target=start_flask, args=())
        flaskProcess.start()
        sfmProcess.start()
        flaskProcess.join()
        sfmProcess.join()
