"""Main script: it includes our API initialization and endpoints."""

import tensorflow as tf
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import List
from fastapi import FastAPI, Request,UploadFile,HTTPException,File
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODELS_DIR = Path("models/")
model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Object Recognition System for Visually Impaired People",
    description="This API lets you make predictions for Visually Impaired People",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to IRIS classifier! Please, read the `/docs`!"},
    }
    return response


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
        # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
                ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
                    
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
                
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


@construct_response
@app.post("/predict/{type}")
async def _predict(type: str, file: UploadFile = File(...)):

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        # image=Image.open(nparr).convert('RGB')
        
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        PATH_TO_FROZEN_GRAPH="models//frozen_inference_graph.pb"
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = 'models//mscoco_label_map.pbtxt'

        if image is None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail="Invalid image file"
            )
        else:
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
           
            if image is None:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST, detail="Invalid image file"
                )
            else:
                # image=Image.open(image_path).convert('RGB')
                
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.

                # image_np = load_image_into_numpy_array(image)
                image_np=image.astype(np.uint8)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = run_inference_for_single_image(image_np, detection_graph)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,line_thickness=8)
                
                indexs=[i for i,score in enumerate(output_dict['detection_scores']) if score!=0]
                boxes=[output_dict['detection_boxes'][i] for i in indexs]
                labels=[category_index[output_dict['detection_classes'][i]]['name'] for i in indexs]
                scores=[str(output_dict['detection_scores'][i]) for i in indexs]
                
            return {
                "detected_objects":labels,
                "scores": scores,
                "coordinates": str(boxes)
                # "encoded_img": image_np.decode(),
                
            } 