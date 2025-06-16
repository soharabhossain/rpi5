def get_last_output_name(param_path):
    """
    Parses the NCNN model.param file and returns the output name of the last layer.
    """
    with open(param_path, 'r') as f:
        lines = f.readlines()

    # Skip the first few header lines
    layer_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

    # Search from the end for a layer definition
    for i in range(len(layer_lines) - 1, -1, -1):
        tokens = layer_lines[i].split()
        if len(tokens) >= 4:
            # Format: layer_type layer_name bottom_blob_count top_blob_count
            # e.g., "YoloV5DetectionOutput output 1 1"
            top_blob_count = int(tokens[-1])
            if top_blob_count > 0:
                # Top blob names are usually on the next line
                output_names = layer_lines[i + 1].split()
                if output_names:
                    return output_names[0]

    raise RuntimeError("Could not determine output name from param file.")


param_file_path = "yolov8n_ncnn_model/model.param"
output_name = get_last_output_name(param_file_path)
print(f"Last output name: {output_name}")
