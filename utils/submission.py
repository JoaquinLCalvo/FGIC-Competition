import json
from config import GROUP_NAME 
from torch.utils.data import DataLoader

def submit__(results, filename="results.json"):
    """
    Saves results to a JSON file with the specified filename.
    """
    res = json.dumps(results, indent=4)
    try:
        with open(filename, "w") as file:
            file.write(res)
        print(f"Results saved to {filename}")
    except IOError as e:
        print(f"ERROR: Unable to write to file {filename}. Exception: {e}")

def test_model_collect_predictions(dataloaders, model, class_names):
    """
    Collects predictions for the test set.
    """
    model.eval()
    preds = {}

    for inputs, image_ids in dataloaders['test']:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        for image_id, pred in zip(image_ids, preds):
            preds[image_id] = class_names[pred.item()]

    return preds

# Example of using the submission function
# class_names = dataloaders['train'].dataset.dataset.classes
# preds = test_model_collect_predictions(dataloaders, model, class_names)
# res = {
#     "images": preds,
#     "groupname": GROUP_NAME  # Set group name from config
# }
# submit__(res)