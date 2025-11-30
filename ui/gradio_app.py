import gradio as gr
from gradio_client import utils as gc_utils  # needed for the patch

# ---- PATCH: fix 'argument of type bool is not iterable' bug in gradio_client ----
# Some gradio versions pass a plain bool into get_type(), which crashes.
# We wrap it so bools are handled safely instead of raising TypeError.

_original_get_type = gc_utils.get_type

def _safe_get_type(schema):
    if isinstance(schema, bool):
        # Treat boolean schema objects as a simple "boolean" type
        return "boolean"
    return _original_get_type(schema)

gc_utils.get_type = _safe_get_type
# ---- END PATCH -------------------------------------------------------------------


from detector.pipeline import (
    process_uploaded_video,
    process_webcam,
    process_phone_ip_cam,
)


def gradio_main(
    source_type,
    video_file,
    webcam_duration,
    phone_ip_url,
    phone_duration,
    conf_thres,
    accident_iou,
    area_growth,
):
    """
    Router function called by Gradio when user clicks 'Run Detection'.
    Chooses which pipeline function to call based on source_type.
    """
    if source_type == "Upload video":
        return process_uploaded_video(video_file, conf_thres, accident_iou, area_growth)

    elif source_type == "Laptop webcam":
        return process_webcam(webcam_duration, conf_thres, accident_iou, area_growth)

    else:  # "Phone IP webcam (URL)"
        return process_phone_ip_cam(
            phone_ip_url, phone_duration, conf_thres, accident_iou, area_growth
        )


def create_app():
    """
    Build and return the Gradio Blocks UI.
    """
    with gr.Blocks(
        title="Vehicle Accident Detection",
        theme=gr.themes.Soft(primary_hue="red", secondary_hue="slate"),
    ) as demo:
        gr.Markdown(
            """
            <div style="text-align:center;">
              <h1>🚦 Smart Vehicle Accident Detector</h1>
              <p>Upload videos or use your laptop / phone camera to detect potential traffic accidents using YOLOv8.</p>
            </div>
            """
        )

        with gr.Tab("Run Detection"):
            with gr.Row():
                # ---------------- LEFT COLUMN: INPUTS ----------------
                with gr.Column(scale=1):
                    source_type = gr.Radio(
                        ["Upload video", "Laptop webcam", "Phone IP webcam (URL)"],
                        value="Upload video",
                        label="Source type",
                    )

                    video_input = gr.Video(
                        label="Video file (if 'Upload video' selected)",
                        interactive=True,
                    )

                    webcam_duration = gr.Slider(
                        minimum=3,
                        maximum=60,
                        value=10,
                        step=1,
                        label="Laptop webcam duration (seconds)",
                    )

                    phone_ip_url = gr.Textbox(
                        label="Phone IP Webcam URL (if 'Phone IP webcam (URL)' selected)",
                        placeholder="e.g. http://192.168.1.5:8080/video",
                    )

                    phone_duration = gr.Slider(
                        minimum=3,
                        maximum=60,
                        value=10,
                        step=1,
                        label="Phone IP webcam capture duration (seconds)",
                    )

                    conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.4,
                        step=0.05,
                        label="YOLO confidence threshold",
                        info="Higher = fewer but more confident detections.",
                    )

                    iou_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.35,
                        step=0.05,
                        label="Accident IoU threshold (overlap)",
                        info="Minimum overlap between vehicles to consider a collision.",
                    )

                    growth_slider = gr.Slider(
                        minimum=1.1,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                        label="Area growth factor",
                        info="How much bigger the bounding box must get between frames to flag an accident.",
                    )

                    run_btn = gr.Button("🚀 Run Detection", variant="primary")

                # ---------------- RIGHT COLUMN: OUTPUTS ----------------
                with gr.Column(scale=1):
                    gallery_output = gr.Gallery(
                        label="Accident Frames (JPG)",
                        type="filepath",
                        columns=3,
                        rows=2,
                        height="auto",
                    )
                    csv_output = gr.File(label="Accident Log CSV")

            # Click handler
            run_btn.click(
                fn=gradio_main,
                inputs=[
                    source_type,
                    video_input,
                    webcam_duration,
                    phone_ip_url,
                    phone_duration,
                    conf_slider,
                    iou_slider,
                    growth_slider,
                ],
                outputs=[gallery_output, csv_output],
            )

        # ---------------- ABOUT TAB ----------------
        with gr.Tab("About"):
            gr.Markdown(
                """
                ### ℹ️ About this project

                - **Model**: YOLOv8n (Ultralytics)  
                - **Inputs supported**:
                    - Uploaded CCTV / dashcam videos  
                    - Laptop webcam capture  
                    - Phone IP webcam (via apps like "IP Webcam" over Wi-Fi)  
                - **Outputs**:
                    - Accident frame snapshots (JPG)  
                    - CSV log with event id, frame, timestamp, and snapshot path  

                💡 Tips:
                - For phone IP webcam, ensure your **phone and laptop are on the same Wi-Fi**.  
                - Test the IP URL (e.g. `http://192.168.1.5:8080/video`) in your browser first to confirm the stream works.
                """
            )

    return demo
