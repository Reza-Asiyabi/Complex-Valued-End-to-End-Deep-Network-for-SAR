import torch
from torch.nn import Module
import torch.nn.functional as F

from Src.CV_Operators import complex_MSE_loss, complex_CrossEntropy_loss, complex_avg_pool1d, complex_avg_pool2d, complex_Upsample
from Src.Coherence import complex_correlation


class ComplexMSEloss(Module):
    '''
    Computs the loss by subtracting real part and imaginary part.
    '''

    def __init__(self):
        super(ComplexMSEloss, self).__init__()

    def forward(self, predicted, target):

        loss = complex_MSE_loss()
        loss_r = loss(predicted.real, target.real)
        loss_i = loss(predicted.imag, target.imag)

        return (loss_r + loss_i)/2

class ComplexCrossEntropyloss(Module):
    '''
    Computs the cross entropy loss.
    '''

    def __init__(self):
        super(ComplexCrossEntropyloss, self).__init__()

    def forward(self, predicted, target):

        loss_madule = complex_CrossEntropy_loss()
        loss = loss_madule(predicted, target)

        return loss

class Complex2foldloss(Module):
    '''
    Computs the MSE loss for the reconstruction and the classification and average them with the regularization term (alpha).
    '''

    def __init__(self, alpha=0.5):
        super(Complex2foldloss, self).__init__()
        self.alpha = alpha

    def forward(self, reconstroction_predicted, reconstroction_target, classification_predicted, classification_target):

        reconstroction_loss_madule = complex_MSE_loss()
        reconstroction_loss_r = reconstroction_loss_madule(reconstroction_predicted.real, reconstroction_target.real)
        reconstroction_loss_i = reconstroction_loss_madule(reconstroction_predicted.imag, reconstroction_target.imag)

        # classification_loss_madule = complex_CrossEntropy_loss()
        # classification_loss_r = classification_loss_madule(classification_predicted.real, classification_target.real.long())
        # classification_loss_i = classification_loss_madule(classification_predicted.imag, classification_target.imag.long())

        classification_loss_madule = torch.nn.CrossEntropyLoss()
        classification_loss = classification_loss_madule(classification_predicted, classification_target.long())

        reconstroction_loss = (reconstroction_loss_r + reconstroction_loss_i)/2
        # classification_loss = (classification_loss_r + classification_loss_i)/2

        loss = self.alpha*reconstroction_loss + (1-self.alpha)*classification_loss
        # print("reconstroction_loss", reconstroction_loss)
        # print("classification_loss", classification_loss)
        # print("loss", loss)

        return loss, reconstroction_loss, classification_loss

class Complex2foldloss_Coh(Module):
    '''
    Computs the Coherency loss for the reconstruction and the classification and average them with the regularization term (alpha).
    '''

    def __init__(self, alpha=0.5):
        super(Complex2foldloss_Coh, self).__init__()
        self.alpha = alpha

    def forward(self, reconstroction_predicted, reconstroction_target, classification_predicted, classification_target):

        reconstroction_loss = 1 - torch.mean(complex_correlation(inputdata1=reconstroction_predicted, inputdata2=reconstroction_target, k=2))

        classification_loss_madule = torch.nn.CrossEntropyLoss()
        classification_loss = classification_loss_madule(classification_predicted, classification_target.long())

        loss = self.alpha*reconstroction_loss + (1-self.alpha)*classification_loss
        # print("reconstroction_loss", reconstroction_loss)
        # print("classification_loss", classification_loss)
        # print("loss:", loss)

        return loss, reconstroction_loss, classification_loss

class ComplexCoherencyloss_1d(Module):
    '''
    Computs the loss by averaging the coherency of amplithud and phase between the input and output 1D vectors.
    '''

    def __init__(self):
        super(ComplexCoherencyloss_1d, self).__init__()

    def forward(self, predicted, target, filter_size=5):


        filter = torch.ones(predicted.size(1), predicted.size(1), filter_size).cuda()

        c = (F.conv1d(torch.real(predicted * torch.conj(target)), filter, padding="valid") + 1j *
             F.conv1d(torch.imag(predicted * torch.conj(target)), filter, padding="valid")) / \
            (torch.sqrt(F.conv1d(torch.abs(predicted) ** 2, filter, padding="valid") *
                        F.conv1d(torch.abs(target) ** 2, filter, padding="valid")))

        coh = torch.mean(abs(c))

        return (1 - coh)


class ComplexCoherencyloss(Module):
    '''
    Computs the loss by averaging the coherency of amplithud and phase between the input and output images.
    '''

    def __init__(self):
        super(ComplexCoherencyloss, self).__init__()

    def forward(self, predicted, target, filter_size=5):


        filter = torch.ones(predicted.size(1), predicted.size(1), filter_size, filter_size).cuda()

        c = (F.conv2d(torch.real(predicted * torch.conj(target)), filter, padding="valid") + 1j *
             F.conv2d(torch.imag(predicted * torch.conj(target)), filter, padding="valid")) / \
            (torch.sqrt(F.conv2d(torch.abs(predicted) ** 2, filter, padding="valid") *
                        F.conv2d(torch.abs(target) ** 2, filter, padding="valid")))

        coh = torch.mean(abs(c))
        # print("predicted", predicted.size())
        # print("target", target.size())
        # print("c", c.size())
        # print("coh", coh)
        return (1 - coh)




def get_loss_function(loss_function_name):

    if loss_function_name == 'ComplexMSEloss':
        loss = ComplexMSEloss()
    elif loss_function_name == 'ComplexCrossEntropyloss':
        loss = ComplexCrossEntropyloss()
    else:
        print('Loss Function is not defined')

    return loss