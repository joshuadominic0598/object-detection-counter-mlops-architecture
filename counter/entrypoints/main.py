import sys
import uuid

from counter import config

if __name__ == '__main__':
    img_path = sys.argv[1]
    threshold = float(sys.argv[2])
    session_id = sys.argv[3] if len(sys.argv) > 3 else str(uuid.uuid4())

    with open(img_path, 'rb') as img:
        response, result_image_path = config.get_count_action().execute(img, threshold, session_id)

        print(response)
