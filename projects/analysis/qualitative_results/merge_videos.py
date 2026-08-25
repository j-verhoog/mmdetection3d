from moviepy.editor import VideoFileClip, clips_array

video1_path = "/home/jolle/mmdet/mmdetection3d/projects/analysis/qualitative_results/test/e8834785d9ff4783a5950281a4579943/video/sequence_model_a.mp4"
video2_path = "/home/jolle/mmdet/mmdetection3d/projects/analysis/qualitative_results/test/e8834785d9ff4783a5950281a4579943/video/sequence_model_b.mp4"
output_path = "/home/jolle/mmdet/mmdetection3d/projects/analysis/qualitative_results/test/e8834785d9ff4783a5950281a4579943/video/side_by_side.mp4"

v1 = VideoFileClip(video1_path)
v2 = VideoFileClip(video2_path)

combined = clips_array([[v1, v2]])

combined.write_videofile(
    output_path,
    codec="libx264",
    audio_codec="aac",
    fps=v1.fps
)

v1.close()
v2.close()
combined.close()