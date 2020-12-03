import logging
import time
import edgeiq
import pandas as pd
import numpy as np
"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

To change the engine and accelerator, follow this guide:
https://alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html

To install app dependencies in the runtime container, list them in the
requirements.txt file.
"""


def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(engine=edgeiq.Engine.DNN_OPENVINO)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()
            
            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                pdict = dict()
                myList = [None]*15

                # Generate text to display on streamer
                text = ["Model: {}".format(pose_estimator.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration) + 
                        "\nFPS: {:.2f}".format(fps.compute_fps()))
                for ind, pose in enumerate(results.poses):
                    pdict["Person {}".format(ind)] = pose.key_points   
                    df = pd.DataFrame(data=pdict, index=pose.key_points, columns=pdict)
                    print(df)

                streamer.send_data(results.draw_poses(frame), text)
                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
