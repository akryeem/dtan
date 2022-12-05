import os
import sys
module_path = os.path.abspath(os.path.join('..\\..'))
module_path1 = os.path.abspath(os.path.join('..\\..\\..'))
module_path2 = os.path.abspath(os.path.join('..\\..\\DTAN'))
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

if module_path not in sys.path:
    sys.path.append(module_path)
if module_path1 not in sys.path:
    sys.path.append(module_path1)
if module_path2 not in sys.path:
    sys.path.append(module_path2)

from models.train_utils import ExperimentsManager, DTAN_args
from DTAN.DTAN_layer import DTAN
from UCR_alignment import argparser
import csv

from tslearn.preprocessing import TimeSeriesResampler
def my_inter(list1, resample_len):
  if (list1.size == resample_len):
      return list1.tolist()
  tmp =  list1.reshape(list1.size)
  resample_list1 = TimeSeriesResampler(sz=resample_len).fit_transform(tmp)
#  print(resample_list1[0])
  #newList = np.array(resample_list1[0])
 # newList = newList.reshape(resample_len)
  #print(list1)
  #print(newList.tolist())
  tmp = resample_list1.reshape(resample_len)
  return tmp.tolist()

def inference(args, dataset_name="ECGFiveDays"):
    """
    Run an example of the full training pipline for DTAN on a UCR dataset.
    After training:
        - The model checkpoint (based on minimal validation loss) at checkpoint dir.
        - Plots alignment, within class, for train and test set.

    Args:
        args: described at argparser. args for training, CPAB basis, etc
        dataset_name: dataset dir name at examples/data

    """

    # Print args
    print(args)

    # Data
    datadir = args.dpath #"data/UCR/UCR_TS_Archive_2015"
    device = 'cpu'
    exp_name = f"{dataset_name}_exp"

    # Init an instance of the experiment class. Holds results
    # and trainning param such as lr, n_epochs etc
    expManager = ExperimentsManager()
    expManager.add_experiment(exp_name, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr, device=device)
    Experiment = expManager[exp_name]

    # DTAN args
    DTANargs1 = DTAN_args(tess_size=args.tess_size,
                          smoothness_prior=args.smoothness_prior,
                          lambda_smooth=args.lambda_smooth,
                          lambda_var=args.lambda_var,
                          n_recurrences=args.n_recurrences,
                          zero_boundary=True,
                          )
    expManager[exp_name].add_DTAN_arg(DTANargs1)
    SIGNAL_LENGTH = 800
    CHANNELS = 1
    DTANargs = Experiment.get_DTAN_args()

    #channels, input_shape = train_loader.dataset[0][0].shape
    channels = CHANNELS
    input_shape = SIGNAL_LENGTH
    loaded_model = DTAN(input_shape, channels, tess=[DTANargs.tess_size,], n_recurrence=DTANargs.n_recurrences,
                    zero_boundary=DTANargs.zero_boundary, device=device).to(device)
    loaded_model.load_state_dict(torch.load('../../checkpoints/ex234_modelstate_dict.pth'))
    loaded_model.eval()

    dataset = dataset_name
    #fdir = os.path.join(datadir, dataset)
    #assert os.path.isdir(datadir), f"{datadir}. {dataset} could not be found in {datadir}"
    # again, for file names
    f_name = os.path.join(datadir, dataset)
    #f_name = r"D:\M.Sc_study\github\thesis\ilan_computerVisionML_AI\mediapipe_eval\data\p7\h15_f2_out.txt"
    noise_amp=[]         #an empty list to store the second column
    with open(f_name+'.txt', 'r') as rf:
        reader = csv.reader(rf, delimiter=',')
        #next(reader) #uncomment if input file has a header line
        for row in reader:
          noise_amp.append(float(row[0])) #change to row[1] if input file has two columns
    X_test = np.array([noise_amp, noise_amp])
    #X_test = np.loadtxt(f_name+'.txt',delimiter=',')


    # get data
    #X_test = data_test_val[:,1:]
    # get labels (numerical, not one-hot encoded)
    #y_test = data_test_val[:,0]

    # add a third channel for univariate data
    if len(X_test.shape) < 3:
        X_test = np.expand_dims(X_test, -1)
    # Switch channel dim ()
    # Torch data format is  [N, C, W] W=timesteps
    X_test = np.swapaxes(X_test, 2, 1)

    to_inference = []
    to_inference.append(my_inter(X_test[0], input_shape))
    to_inference.append(my_inter(X_test[1], input_shape))
    np_to_inference = np.array(to_inference)

    ## add a third channel for univariate data
    if len(np_to_inference.shape) < 3:
        np_to_inference = np.expand_dims(np_to_inference, -1)
    np_to_inference = np.swapaxes(np_to_inference, 2, 1)
    X = torch.Tensor(np_to_inference).to("cpu")

    transformed_input_tensor, thetas = loaded_model(X, return_theta=True)
    transformed_data_numpy = transformed_input_tensor.data.cpu().numpy()
    transformed_data_numpy = transformed_data_numpy.reshape(2, input_shape)
    np.savetxt(f'aligned_{dataset}.txt', transformed_data_numpy[0], delimiter=',',fmt='%d')
    nb = np.arange(1, input_shape+1)
    plt.plot(nb, transformed_data_numpy[0], label="Aligned", color ="green")
    plt.plot(np_to_inference[0][0], label="Orig", color ="blue")
    plt.legend()
    #plt.plot(X_test[0][0], color ="red")
    plt.show()
    list_x = my_inter(X_test[0], input_shape)
    list_xTF = torch.Tensor(np.array([list_x,list_x]).reshape(2,1,800))
    xs = loaded_model.localization(list_xTF)
    xs = xs.view(-1, loaded_model.fc_input_dim)
    theta = loaded_model.fc_loc(xs)
    theta_inv = torch.mul(theta, -1)
    transformed_input_tensor = loaded_model.T.transform_data(list_xTF, theta, outsize=(loaded_model.input_shape,))
    transformed_input_tensor_inv = loaded_model.T.transform_data(transformed_input_tensor, theta_inv, outsize=(loaded_model.input_shape,))

    transformed_data_numpy_inv = transformed_input_tensor_inv.data.cpu().numpy()
    transformed_data_numpy_inv = transformed_data_numpy_inv.reshape(2, input_shape)
    theta_vals = theta.data.cpu().numpy()


    plt.plot(nb, transformed_data_numpy_inv[0], label="Aligned", color ="green")
    plt.plot(np_to_inference[0][0], label="Orig", color ="blue")
    plt.legend()
    #plt.plot(X_test[0][0], color ="red")
    plt.show()
    print("done")

if __name__ == "__main__":
    args = argparser()
    inference(args, dataset_name=args.dataset)

