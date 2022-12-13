import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from loss import BaselineLoss

if __name__ == "__main__":
    output = torch.rand((2, 3, 128, 128, 128))
    print(output)

    predictions = output.argmax(1, keepdim = True)
    print(predictions)

    ground_truth = output.argmax(1, keepdim = True)
    # create a boolean mask
    probabilities, classes = torch.max(output, keepdim = True, dim = 1)
    print(probabilities)

    # # Mask = probabilities[probabilities < 0.5]
    # # print(Mask)
    #
    # Mask = torch.ByteTensor(2, 1, 128, 128, 128)
    # print(Mask)
    print(ground_truth.shape)
    print(probabilities.shape)
    ground_truth[probabilities > 0.5] = 255
    print(ground_truth)

    loss = BaselineLoss(ignore_label = 255)

    val = loss(output, ground_truth)






