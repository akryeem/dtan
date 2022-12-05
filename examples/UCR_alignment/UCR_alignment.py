"""
Created on Oct  2019
Modified for torch Aug 2020

author: ronsha

End-to-end alignment of datasets belonging to the UCR archive.
If you call 'run_UCR_alignment(...)' from another scripts, make sure to construct and pass an args class.
You can see "UCR_NCC.py" for example.

Plotting:
By default, the script will produce figures for each class in a similar fashion to figure 1. from [1].
You disable it via 'plot_signals_flag'.
In addition, you can:
1. Plot the output of RDTAN at each recurrence
2. Create an animation of RDTAN at each recurrence.

This is possible by simply uncommenting the relevant lines.
"""
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

# From helper
from helper.plotting_torch import plot_signals
from helper.UCR_loader import get_UCR_data

# from models
from models.train_model import train

from models.train_utils import ExperimentsManager, DTAN_args
from DTAN.DTAN_layer import DTAN
from helper.UCR_loader import processed_UCR_data, load_txt_file

def argparser():
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--dataset', type=str, default='ECGFiveDays')
    parser.add_argument('--tess_size', type=int, default=16,
                        help="CPA velocity field partition")
    parser.add_argument('--smoothness_prior', default=True,
                        help="smoothness prior flag", action='store_true')
    parser.add_argument('--no_smoothness_prior', dest='smoothness_prior', default=True,
                        help="no smoothness prior flag", action='store_false')
    parser.add_argument('--lambda_smooth', type=float, default=1,
                        help="lambda_smooth, larger values -> smoother warps")
    parser.add_argument('--lambda_var', type=float, default=0.1,
                        help="lambda_var, larger values -> larger warps")
    parser.add_argument('--n_recurrences', type=int, default=1,
                        help="number of recurrences of R-DTAN")
    parser.add_argument('--zero_boundary', type=bool, default=True,
                        help="zero boundary constrain")
    parser.add_argument('--n_epochs', type=int, default=500,
                        help="number of epochs")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--dpath', type=str, default="", help="dataset dir path, default examples/data")
    args = parser.parse_args()
    return args
from tslearn.preprocessing import TimeSeriesResampler
def my_inter(list1, resample_len):
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
    loaded_model.load_state_dict(torch.load('../../checkpoints/identity_modelstate_dict.pth'))
    loaded_model.eval()

    dataset = dataset_name
    #fdir = os.path.join(datadir, dataset)
    #assert os.path.isdir(datadir), f"{datadir}. {dataset} could not be found in {datadir}"
    # again, for file names
    f_name = os.path.join(datadir, dataset)

    X_test = np.loadtxt(f_name+'.txt',delimiter=',')


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

    nb = np.arange(1, input_shape+1)
    plt.plot(nb, transformed_data_numpy[0], label="Aligned", color ="green")
    plt.plot(np_to_inference[0][0], label="Orig", color ="blue")
    plt.legend()
    #plt.plot(X_test[0][0], color ="red")
    plt.show()

def run_UCR_alignment(args, dataset_name="ECGFiveDays"):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"{dataset_name}_exp"
    # Plotting flag
    plot_signals_flag = True

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
    train_loader, validation_loader, test_loader = get_UCR_data(dataset_name=dataset_name,
                                                                datadir=datadir,
                                                                batch_size=Experiment.batch_size)
    X_train, X_test, y_train, y_test = load_txt_file(datadir, dataset_name)
    X_train, X_test, y_train, y_test = processed_UCR_data(X_train, X_test, y_train, y_test)

    ## Train model
    model = train(train_loader, validation_loader, DTANargs, Experiment, print_model=True)
    torch.save(model.state_dict(), f'../../checkpoints/ex234_modelstate_dict.pth')
    #torch.save(model, f'../../checkpoints/identity_model.pth')
    # Plot aligned signals
    if plot_signals_flag:
        # Plot test data
        plot_signals(model, device, datadir, dataset_name)



if __name__ == "__main__":
    args = argparser()
    #inference(args, dataset_name=args.dataset)
    run_UCR_alignment(args, dataset_name=args.dataset)


#Ttosavealign = X_aligned_within_class[:, channel,:].T
#
#Ttosavealign1 = Ttosavealign.reshape(10,800)
#
#np.savetxt('C:/tmp/Talined.txt', Ttosavealign1, delimiter=',')

# References:
# [1] - Diffeomorphic Temporal Alignment Nets (NeurIPS 2019)

