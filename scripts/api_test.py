import json
from pathlib import Path

import requests


BASE_URL = "http://localhost:5000"
IMAGE_DIR = Path("resources/images")
SESSION_ID = "test-session-001"
THRESHOLD = 0.3


images = sorted(
    [
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
)


if not images:
    print(f"No images found in {IMAGE_DIR}")
    raise SystemExit(1)


print()
print("=" * 70)
print("OBJECT DETECTION - ALL IMAGES")
print("=" * 70)
print()
print(f"Image directory : {IMAGE_DIR}")
print(f"Session ID      : {SESSION_ID}")
print(f"Threshold       : {THRESHOLD}")
print(f"Images found    : {len(images)}")
print(f"Endpoint        : /object-detection")
print(f"Counter         : true")
print()
print("=" * 70)


successful = 0
failed = 0


for image_path in images:

    print()
    print("-" * 70)
    print(f"IMAGE: {image_path.name}")
    print("-" * 70)

    try:

        with image_path.open("rb") as image_file:

            response = requests.post(
                f"{BASE_URL}/object-detection",
                files={
                    "file": (
                        image_path.name,
                        image_file,
                    )
                },
                data={
                    "session_id": SESSION_ID,
                    "threshold": THRESHOLD,
                    "counter": "true",
                },
                timeout=300,
            )


        print(f"Status: {response.status_code}")

        if response.ok:

            successful += 1

            try:
                result = response.json()

                print(
                    json.dumps(
                        result,
                        indent=4,
                    )
                )

            except ValueError:
                print(response.text)

        else:

            failed += 1

            print("FAILED")
            print(response.text)


    except requests.RequestException as ex:

        failed += 1

        print(f"ERROR: {ex}")


print()
print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print()
print(f"Images tested : {len(images)}")
print(f"Successful    : {successful}")
print(f"Failed        : {failed}")
print(f"Session ID    : {SESSION_ID}")
print(f"Threshold     : {THRESHOLD}")
print()
print("=" * 70)