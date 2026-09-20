from unittest.mock import Mock

from counter.debug import draw
from tests.domain.helpers import generate_prediction


class FakeDrawer:

    def __init__(self):
        self.rectangles = []
        self.text_calls = []

    def rectangle(self, box, outline, width):
        self.rectangles.append({"box": box, "outline": outline, "width": width})

    def text(self, position, label, font, fill):
        self.text_calls.append({"position": position, "label": label, "font": font, "fill": fill})


def test_draw_uses_fixed_colors_for_car_and_free(monkeypatch):
    fake_drawer = FakeDrawer()
    image = Mock()

    monkeypatch.setattr("counter.debug.ImageDraw.Draw", lambda image, mode: fake_drawer)
    monkeypatch.setattr("counter.debug.ImageFont.truetype", lambda path, size: Mock())

    predictions = [
        generate_prediction("car", 0.91),
        generate_prediction("free", 0.87),
    ]

    draw(predictions, image, "/tmp/output.jpg")

    assert [rectangle["outline"] for rectangle in fake_drawer.rectangles] == ["green", "blue"]
    image.save.assert_called_once_with("/tmp/output.jpg", "JPEG")


def test_draw_assigns_distinct_colors_to_other_classes(monkeypatch):
    fake_drawer = FakeDrawer()
    image = Mock()

    monkeypatch.setattr("counter.debug.ImageDraw.Draw", lambda image, mode: fake_drawer)
    monkeypatch.setattr("counter.debug.ImageFont.truetype", lambda path, size: Mock())

    predictions = [
        generate_prediction("person", 0.91),
        generate_prediction("bicycle", 0.87),
        generate_prediction("truck", 0.79),
        generate_prediction("person", 0.66),
    ]

    draw(predictions, image, "/tmp/output.jpg")

    outlines = [rectangle["outline"] for rectangle in fake_drawer.rectangles]
    distinct_outlines = {outline for outline in outlines}

    assert len(distinct_outlines) == 3
    assert "green" not in distinct_outlines
    assert "blue" not in distinct_outlines
    assert outlines[0] == outlines[3]
    image.save.assert_called_once_with("/tmp/output.jpg", "JPEG")