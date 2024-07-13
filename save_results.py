import json
import os


def save_results(results, filename="system_performance.json"):
    # Check if file exists and is not empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as file:
            data = json.load(file)
            # Update the data with new results, or add a new key if necessary
            data.update(results)
    else:
        data = results

    # Write the updated or new data to the file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
