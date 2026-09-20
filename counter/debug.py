from itertools import cycle

from PIL import ImageDraw, ImageFont


SPECIAL_CLASS_COLORS = {
    "car": "green",
    "free": "blue",
}

CLASS_COLOR_PALETTE = [
    "red",
    "orange",
    "yellow",
    "magenta",
    "cyan",
    "lime",
    "deepskyblue",
    "gold",
    "hotpink",
    "violet",
]


def _build_class_color_map(predictions):
    class_color_map = dict(SPECIAL_CLASS_COLORS)

    available_colors = [
        color for color in CLASS_COLOR_PALETTE if color not in SPECIAL_CLASS_COLORS.values()
    ]
    color_cycle = cycle(available_colors)

    for class_name in sorted({prediction.class_name for prediction in predictions}):
        if class_name in class_color_map:
            continue

        class_color_map[class_name] = next(color_cycle)

    return class_color_map


def draw(predictions, image, output_path):
    draw_image = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype("counter/resources/arial.ttf", 20)
    class_color_map = _build_class_color_map(predictions)

    for prediction in predictions:
        box = prediction.box
        outline_color = class_color_map.get(prediction.class_name, "green")

        draw_image.rectangle([(box.xmin, box.ymin), (box.xmax, box.ymax)], outline=outline_color, width=3)

        label = f"{prediction.class_name} {prediction.score:.2f}"
        draw_image.text((box.xmin, max(0, box.ymin - 25)), label, font=font, fill="black")

    image.save(output_path, "JPEG")