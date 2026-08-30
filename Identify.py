from ultralytics import YOLO

# Load a model
model = YOLO("Animal.pt")  # pretrained YOLO26n model

# Run batched inference on a list of images
results = model(["animal1.jpg", "animalspic.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs# Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk