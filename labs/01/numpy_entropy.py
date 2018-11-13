#!/usr/bin/env python3
import numpy as np
from collections import Counter

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    
    points = list()

    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            points.append(line)

    counts = Counter(points)
    total = len(points)

    sorted_data_probs = {key: value/total for key, value in sorted(counts.items(), key=lambda x: x[0])}

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    
    model_data = list()

    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            
            point, prob = line.split('\t')
            model_data.append((point, float(prob)))

    sorted_model_probs = {point: prob for point, prob in sorted(model_data, key=lambda x: x[0]) if point in sorted_data_probs}
    extras_model = {point: 0.0 for point in sorted_data_probs if point not in sorted_model_probs}
    sorted_model_probs.update(extras_model)
    
    data_dist = np.array([value for _, value in sorted_data_probs.items()])
    model_dist = np.array([value for _, value in sorted_model_probs.items()])

    entropy = -np.sum(data_dist * np.log(data_dist))

    # TODO: Compute and print entropy H(data distribution)
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    
    assert len(data_dist) == len(model_dist)
    
    cross_entropy = -np.sum(data_dist * np.log(model_dist))
    d_kl = np.sum(data_dist * (np.log(data_dist) - np.log(model_dist)))

    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(d_kl))
