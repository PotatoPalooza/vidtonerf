from genericpath import exists
import subprocess
import os
import sys
from pathlib import Path

# new imports
import cv2
from random import sample
#Usage: python video_to_images.py --flags
#Flags: --ffmpeg_exe_path "path" ==> Path to the ffmpeg executeable.
#                                  > Defaults to looking for ffmpeg.exe in the folder this script is in.
#
#       --wanted_frames "uint"   ==> Number of frames we want to use
#                                  > If total frames < wanted_frames, we default to total frames
#                                  > Defaults to 200.
#
#       --name "name"            ==> Name of the folder to be created to store the data for this instance
#                                  > of ffmpeg.
#                                  > Defaults to "ffmpeg_output"
#
#       --output_folder "path"   ==> Directory to where ffmpeg will put its output.
#                                  > Defaults to the folder where this script is
#
#       --video_path "path"      ==> Path to the video to be converted into its composite images.
#                                  > Defaults to looking for "video.mp4" in the folder this script
#                                  > is in.



#split_video_into_frames function:
#
#creates a new folder called instance_name in output_path and fills it with the frames
#    of the video at video_path. Samples wanted_frames amount of frames,
#    or 200 frames by default
#
#returns a status code - 
#    0 = Success
#    1 = Unspecified error
#    2 = FileExistsError; happens when you try to create data in an already existing folder
#    3 = FileNotFoundError; happens when you try to use an output folder that does not exist
#    4 = Video is too blurry (if blur_check is True)

def split_video_into_frames(video_path: str, output_dir_path: str, num_frames_to_select: int = 200, blur_check: bool = False):
  # Standardized parameter names: video_path, output_dir_path, num_frames_to_select
  # Removed max_frames and used num_frames_to_select for clarity, default 200.
  print("Here") # Original print
  ## determines whether image is blurry or not.
  # uses the variance of a laplacian transform to check for edges and returns true
  # if the variance is less than the threshold and the video is determined to be blurry
  # def is_blurry(image, THRESHOLD):
  #   ## Convert image to grayscale
  #   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #   ## run the variance of the laplacian transform to test blurriness
  #   laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
  #   return laplacian_var < THRESHOLD

  ## determines amount of blurriness
  # see IS_BLURRY for more information
  def blurriness(image):
    ## Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ## run the variance of the laplacian transform to test blurriness
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

  # Create output folder
  Path(output_dir_path).mkdir(parents=True, exist_ok=True)

  ## determine video length:
  vidcap = cv2.VideoCapture(video_path)
  if not vidcap.isOpened():
      print(f"Error: Could not open video file: {video_path}")
      return 3 # Using 3 for "file not found / cannot open" type errors as per docstring hints

  frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  if frame_count <= 0:
      print(f"Error: Video file {video_path} has no frames or invalid frame count.")
      vidcap.release()
      return 1 # Unspecified error or could define a new one for "no frames"

  ## sample up to num_frames_to_select
  actual_frames_to_sample = min(frame_count, num_frames_to_select)
  if actual_frames_to_sample <= 0: # Handle cases where num_frames_to_select might be 0 or negative
      actual_frames_to_sample = frame_count # Default to all frames or 1? Let's say all if non-positive.
                                          # Or, more strictly, if num_frames_to_select <= 0, perhaps 0 frames.
                                          # For now, let's ensure it's at least 1 if frame_count > 0.
      if frame_count > 0:
          actual_frames_to_sample = min(frame_count, 1) if num_frames_to_select <=0 else actual_frames_to_sample


  print(f"Frame count: {frame_count}, Target samples: {num_frames_to_select}, Actual samples: {actual_frames_to_sample}")


  success, image = vidcap.read()
  if not success or image is None: # Check if first frame read failed
      print(f"Error: Could not read first frame from video {video_path}.")
      vidcap.release()
      return 1 # Unspecified error
  img_height = image.shape[0]
  img_width = image.shape[1]

  if blur_check:
    ## Rank all images based off bluriness
    blur_list = []
    frame_indices_for_blur_check = []
    current_frame_idx = 0
    ## check blurriness of all images and sort to caluculate threshold
    while success:
      image_blur = blurriness(image)
      blur_list.append(image_blur)
      frame_indices_for_blur_check.append(current_frame_idx)
      current_frame_idx +=1
      success, image = vidcap.read()
    
    # Important: Reset video capture to read frames again for saving
    vidcap.release() 
    vidcap = cv2.VideoCapture(video_path) # Re-open or reset

    if not blur_list: # No frames read during blur check
        print("Error: No frames could be read for blur analysis.")
        vidcap.release()
        return 1

    sorted_list = sorted(blur_list)
    ## we want the remaining best images
    ## e.g, if we want 75 images out of 100, threshold should be 25th image
    # Ensure threshold_idx is valid
    threshold_idx = max(0, len(blur_list) - actual_frames_to_sample) 
    if threshold_idx >= len(sorted_list): # Handles cases where actual_frames_to_sample might be 0 or too small
        THRESHOLD = sorted_list[-1] + 1 # Ensure no frames are selected if not enough valid ones
    else:
        THRESHOLD = sorted_list[threshold_idx]

    # Select frames that meet the threshold
    selected_frame_indices_for_saving = []
    temp_blur_values_and_indices = []

    for i, blur_val in enumerate(blur_list):
        if blur_val >= THRESHOLD:
            temp_blur_values_and_indices.append({'blur': blur_val, 'original_index': frame_indices_for_blur_check[i]})
            
    # Sort by blurriness (desc) to pick the best ones if more than needed
    temp_blur_values_and_indices.sort(key=lambda x: x['blur'], reverse=True)
    
    for item in temp_blur_values_and_indices[:actual_frames_to_sample]:
        selected_frame_indices_for_saving.append(item['original_index'])
    
    # Sort selected_frame_indices_for_saving to write in order if desired, though not strictly necessary
    selected_frame_indices_for_saving.sort()
        

    ## If this threshold is too low, completely reject video 
    avg_threshold = (sorted_list[-1] + THRESHOLD)/2
    if avg_threshold < 100:
      # ERROR: Video is too blurry. Please try again.
      return 4
  

  needs_adjust = False ## determines if we need to adjust
  aspect_ratio = img_height / img_width
  #print (f"aspect ratio: {aspect_ratio}")
  #print (f"img_width: {img_width}")
  #print (f"img_height: {img_height}")
  ## adjust as necessaryx 
  MAX_WIDTH = 400 
  MAX_HEIGHT = 400

  ## for resizing images
  if (img_height > MAX_HEIGHT):
    scaler = MAX_HEIGHT / img_height
    img_height = (int) (img_height * scaler)
    needs_adjust = True

  if (img_width > MAX_WIDTH):
    scaler = MAX_WIDTH / img_width
    img_width = (int) (scaler * img_width)
    needs_adjust = True
  
  ## applying aspect ratio
  if (aspect_ratio > 1):
    img_width = (int) (img_width / aspect_ratio)
  else:
    img_height = (int) (img_height * aspect_ratio)

  #print(f"new img height: {img_height}")
  #print(f"new img width: {img_width}")
  dimensions = (img_width, img_height)

  current_frame_num = 0 # Keep track of the frame number we are reading from video
  saved_frame_count = 0 # Number of frames actually saved

  success, image = vidcap.read() # Start reading frames for saving

  if blur_check:
      # Iterate through frames and save only those selected by blur check
      frames_to_save_set = set(selected_frame_indices_for_saving)
      while success and saved_frame_count < len(selected_frame_indices_for_saving):
          if current_frame_num in frames_to_save_set:
              if needs_adjust:
                  image = cv2.resize(image, dimensions, interpolation=cv2.INTER_LANCZOS4)
              cv2.imwrite(f"{output_dir_path}/img_{current_frame_num:04}.png", image)
              # print(f'Saved image (blur check) {current_frame_num}') # Original print
              saved_frame_count += 1
          success, image = vidcap.read()
          current_frame_num += 1
  else:
      # Simple interval-based sampling
      if actual_frames_to_sample == 0: # Should not happen if actual_frames_to_sample logic is robust
          vidcap.release()
          return 0 # No frames to sample

      step = frame_count // actual_frames_to_sample
      if step == 0 and frame_count > 0 : # Avoid step=0 if actual_frames_to_sample > frame_count (should be handled by min)
          step = 1
      
      img_idx_sequential = 0 # For sequential naming like img_0000.png, img_0001.png
      while success and saved_frame_count < actual_frames_to_sample:
          if current_frame_num % step == 0:
              if needs_adjust:
                  image = cv2.resize(image, dimensions, interpolation=cv2.INTER_LANCZOS4)
              # Using sequential index for non-blur check as per original logic for this branch
              cv2.imwrite(f"{output_dir_path}/img_{img_idx_sequential:04}.png", image)
              # print(f'Saved image (interval) {img_idx_sequential} from original frame {current_frame_num}') # Original print
              img_idx_sequential +=1
              saved_frame_count += 1
          success, image = vidcap.read()
          current_frame_num += 1
  
  vidcap.release()
  return 0


# Test function seems outdated due to signature changes and ffmpeg references
# def test():
#   instance_name = "test"
#   output_path = "test_out"
#   ffmpeg_path = ""
#   video_path = "landscape_video" # change to whatever vid you want
#   wanted_frames = 2001
#   split_video_into_frames(instance_name, output_path, ffmpeg_path, video_path, wanted_frames)

if __name__ == '__main__':
    # This main block is also outdated due to signature changes and ffmpeg removal.
    # It should be updated or removed if not used.
    # For now, commenting out to prevent errors if run directly.
    pass
    # #Default flags
    # instance_name = "ffmpeg_output"
    # output_path = "./"
    # ffmpeg_path = r".\ffmpeg.exe"
    # video_path = r".\video.mp4"
    # wanted_frames = 24

    #Parse flags
    #Flag format up top
    """
    for i in range (len(sys.argv)):
        if i == 0:
            continue
        if sys.argv[i].startswith("--"):
            match sys.argv[i]:
                case "--output_folder":
                    output_path = sys.argv[i+1]
                case "--name":
                    instance_name = sys.argv[i+1]
                case "--ffmpeg_exe_path":
                    ffmpeg_path = sys.argv[i+1]
                case "--video_path":
                    video_path = sys.argv[i+1]
                case "--fps":
                    fps = sys.argv[i+1]
                case _:
                    print("ERROR: Unrecognized flag", sys.argv[i])
                    quit()"""
    
    #Calling split_video_into_frames
    status = split_video_into_frames(instance_name, output_path, ffmpeg_path, video_path, wanted_frames=200)
    if status == 0:
        print("ffmpeg ran successfully.")
    elif status == 1:
        print("ERROR: There was an unknown error running ffmpeg")
    elif status == 2:
        print(f"ERROR: ffmpeg - file {output_path}/{instance_name} already exists.")
    elif status == 3:
        print(f"ERROR: ffmpeg - file {output_path} could not be found.")
    elif status == 4:
        print("ERROR: Video is too blurry. Please try again.")