import dataloader
import torch
import numpy as np
import math
from model import LSTM
from CNN import CNN1D
from matplotlib import pyplot as plt
from subprocess import call
import sys

def main():

    """"

    Select a model
    Run a full experiment
    -Import data
    -Train
    -Validate model against test set
    -Save model and plot results

    """

    indim = 3
    batch_size=64
    cuda=True
    device = torch.device('cuda:0' if cuda else 'cpu')
    net = "CNN"
    # net = "RNN"

    if net == "RNN":
        model = LSTM(indim, hidden_dim=32, batch_size=batch_size, dropout=0.2, wantscuda=cuda, num_layers=2)
    elif net == "CNN":
        model = CNN1D(indim, batch_size, dropout=0.5, wantscuda=cuda)

    model = model.float().to(device)
    
    data = dataloader.split_data()  

    train_hists = training(model, data, 15, batch_size, cuda)
    
    show_res(train_hists)

    evaluate(model, data['Test'], batch_size, cuda, validation=False)

def pre_trained():

    # Load a pre-trained model to visualize test results
    data = dataloader.split_data()

    indim = 3
    batch_size=64
    cuda=True
    device = torch.device('cuda:0' if cuda else 'cpu')
    net = "CNN"

    model = CNN1D(indim, batch_size, dropout=0.5, wantscuda=cuda)

    model = model.to(device)

    evaluate(model, data['Test'], batch_size, cuda, validation=False)

def training(_model, _data, num_epochs, batch_size, wantscuda):

    learning_rate=0.001
    
    device = torch.device('cuda:0' if wantscuda else 'cpu')

    optimiser = torch.optim.Adam(_model.parameters(), lr=learning_rate)

    loss_fn = torch.nn.BCELoss().to(device)
    
    val_acc_hist = []
    hist = []

    rltime_hist = []

    no_batches = len(_data['Training'][0])//batch_size
    
    #####################
    # Train model
    #####################

    train_data = _data['Training']

    for t in range(1,num_epochs+1):
        test_cuda()
        
        _model.train()

        train_acc_counter = 0

        for k in range(no_batches):

            print("Epoch: " + str(t) + ", Batch:" + str(k) + "/" + str(no_batches))

            # Input format: tensor[Batch Size, Seq Len, Channels]
            x_batch = torch.stack(train_data[0][k*batch_size:(k+1)*batch_size]).to(device)

            y_batch = torch.tensor(train_data[1][k*batch_size:(k+1)*batch_size]).to(device)

            # Clear stored gradient
            optimiser.zero_grad()
            
            # Forward pass
            y_pred = _model(x_batch)

            acc = binary_acc(y_pred, y_batch)

            train_acc_counter += acc

            print("Batch Accuracy: {:.3f}%".format(acc*100/batch_size))
        
            loss = loss_fn(y_pred, y_batch.float())
            
            # Backward pass
            loss.backward()
        
            # Update parameters
            optimiser.step()

            rltime_hist.append(loss.item())

        print("Epoch ", t, "Loss: ", loss.item())
        hist.append(loss.item())

        train_acc = train_acc_counter/ (no_batches*batch_size)

        print("Training Accuracy: {:.3f}%".format(train_acc*100))

        val_acc = evaluate(_model, _data['Validation'], batch_size, wantscuda)

        print("Validation Accuracy: {:.3f}%".format(val_acc*100))

        val_acc_hist.append(val_acc)

        if t > 30 and earlystopper(val_acc_hist):
            print('Early stopping at:' + str(t))
            break

    torch.save(_model.state_dict(), './state_dict.pt')

    return (np.array(val_acc_hist), np.array(rltime_hist))

def earlystopper(epoch_hist, patience=10):

    """
    Early stop based on validation accuracy
    Find 2 max accuracy values in the patience window.  
    If improvement is insignificant, halt training
    """

    last = epoch_hist[-1]
    thresh = 0.0001

    history_seq = epoch_hist[-1:-patience:-1]

    max1 = max(history_seq)

    ind = history_seq.index(max1)

    hist2 = history_seq[:ind]
    hist2.extend(history_seq[ind:])

    max2 = max(history_seq)

    if max1-max2 < thresh:
        return True
    else:
        return False

def test_cuda():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()
    
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', torch.cuda.memory_allocated(0)/10**9, 'GB')
        print('Cached:   ', torch.cuda.memory_cached(0)/10**9, 'GB')

def evaluate(_model, test, batch_size, wantscuda, validation=True):
    testtype = 'Validation'

    if not validation:
        _model.load_state_dict(torch.load('./state_dict.pt'))
        testtype = 'Testing'

    device = torch.device('cuda:0' if wantscuda else 'cpu')

    x_test = test[0]
    labels = test[1]

    num_correct = 0
    
    no_batches = len(test[0])//batch_size

    _model.eval()
    
    confusion = np.zeros((3,3))

    for k in range(no_batches):

        x_batch = torch.stack(x_test[k*batch_size:(k+1)*batch_size]).to(device)

        y_batch = torch.tensor(labels[k*batch_size:(k+1)*batch_size]).to(device)

        outputs = _model(x_batch)
        
        acc = binary_acc(outputs, y_batch)

        if not validation:
            confusion += measure_acc(outputs,y_batch)

        num_correct += acc

    acc = num_correct/(no_batches*batch_size)

    if not validation:
        plot_confusion(confusion, no_batches*batch_size)

    print('Test: ' + testtype)
    print("Accuracy: {:.3f}%".format(acc*100))
    
    return acc

def binary_acc(y_hat, labels):

    pred = torch.round(y_hat)
    correct_tensor = pred.eq(labels)
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct = np.sum(correct)

    return num_correct

def show_res(hists):

    valdata = hists[0]
    batches = hists[1]

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(valdata)
    plt.ylabel('Validation Accuracy')

    plt.subplot(2,1,2)
    plt.plot(batches)
    plt.ylabel('Loss by Batch')
    plt.show()

def measure_acc(y_hat, y_pred):

    pred = torch.round(y_hat)
    pred = pred.type_as(y_pred)

    pairs = torch.stack((pred,y_pred)).transpose(1,0)

    confusion = np.zeros((3,3))

    for pair in pairs:
        confusion[pair[0]][pair[1]] += 1

        confusion[pair[0]][2] += 1
        confusion[2][pair[1]] += 1

        confusion[2][2] += 1

    return confusion

def plot_confusion(confusion, sample_size):

        plt.subplot(2,1,1)
    
        rows = ['0', '1', 'total']
        columns = ['0', '1', 'total']
        table = plt.table(cellText=confusion,
                  rowLabels=rows,
                  colLabels=columns, 
                  bbox=[0.15,0.2,0.7,0.5])
        plt.title('Test Set Confusion Matrix')
        plt.axis('off')
        plt.text(0.5,0.1, 'Targets', horizontalalignment='center')
        plt.text(0,0.3, 'Predictions', rotation=90)

        # table indeces include labels
        table[(1,1)].set_facecolor('#EF3030')
        table[(2,0)].set_facecolor('#EF3030')

        table[(1,0)].set_facecolor('#98DC95')
        table[(2,1)].set_facecolor('#98DC95')

        table[(3,2)].set_facecolor('#5D6D7E')

        plt.subplot(2,1,2)

        
        rows = ['0', '1', 'total']
        columns = ['0', '1', 'total']
        table = plt.table(cellText=np.around(confusion/sample_size,3),
                  rowLabels=rows,
                  colLabels=columns, 
                  bbox=[0.15,0.2,0.7,0.5])
        plt.title('Test Set Confusion Matrix - Probabilities')
        plt.axis('off')
        plt.text(0.5,0.1, 'Targets', horizontalalignment='center')
        plt.text(0,0.3, 'Predictions', rotation=90)

        # table indeces include labels
        table[(1,1)].set_facecolor('#EF3030')
        table[(2,0)].set_facecolor('#EF3030')

        table[(1,0)].set_facecolor('#98DC95')
        table[(2,1)].set_facecolor('#98DC95')

        table[(3,2)].set_facecolor('#5D6D7E')

        plt.show()



if __name__ == "__main__":

    np.random.seed(7)

    # main()

    pre_trained()
    pass
