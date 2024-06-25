import json
from pathlib import Path
from PIL import Image, ImageDraw
from semantic_router import Route

def output_parser(output, split_string='', mode='split'):
    if split_string:
        if split_string in output:
            intermediate = output.split(split_string)[1].strip()
        else:
            print(f'LLM output is not in desired format: {output}')
            return output
    
    if mode=='split':
        return intermediate
    elif mode=='json':
        return json.loads(intermediate)
    return output

def get_routes(routes_path='routes'):
    all_routes = Path(routes_path).glob('*.json')
    result = []
    for r in all_routes:
        with open(r, 'r') as f:
            config = json.load(f)
        result.append(Route(**config))
    return result

def draw_bounding_box(image, points, size, color=(255, 0, 0, 32)):
    """
    Draws a bounding box on the image and saves the result.

    :param image_path: Path to the input image
    :param points: List of four tuples representing the coordinates of the bounding box [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    :param output_path: Path to save the output image with the bounding box
    """
    resized_image = image.resize(size).convert("RGBA")
    # Create an overlay image for the highlight
    overlay = Image.new("RGBA", resized_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Ensure points are in the correct format (list of tuples)
    if len(points) != 4 or not all(isinstance(point, tuple) and len(point) == 2 for point in points):
        raise ValueError("Points must be a list of four (x, y) tuples")

    # Draw the transparent colored area
    draw.polygon(points, fill=color)

    # Combine the original image with the overlay
    highlighted_image = Image.alpha_composite(resized_image, overlay)
    return highlighted_image

def merge_bounding_box(bounding_boxes):
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for box in bounding_boxes:
        for (x, y) in box:
            min_x = min(x, min_x)
            min_y = min(y, min_y)
            max_x = max(x, max_x)
            max_y = max(y, max_y)

    return ((min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y))

def merge_elements_metadata(elements):
    all_pages = list(set([ele.metadata.page_number for ele in elements]))
    page = all_pages[0]
    all_bbox = [ele.metadata.coordinates.points for ele in elements]
    final_bbox = merge_bounding_box(all_bbox)
    return page, final_bbox