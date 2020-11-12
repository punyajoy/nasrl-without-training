import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self, actions):
        super(Net, self).__init__()

        if actions is None or len(actions) == 0:
            raise Exception('You must specify at least one action!')

        layers = []
        features_prev = 2

        activation_functions = {
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'ReLU': nn.ReLU(),
            'LeakyReLU': nn.LeakyReLU()
        }

        for action in actions:
            if isinstance(action, int):
                layers.append(nn.Linear(
                    in_features=features_prev,
                    out_features=action))
                features_prev = action
            elif action == 'EOS':
                break
            else:
                layers.append(
                    activation_functions[action]
                )

        layers.append(nn.Linear(
            in_features=features_prev,
            out_features=2
        ))

        self.layers = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)


    def forward(self, x):
        return self.layers(x)


    def fit(self, dl_train, dl_dev):
        num_epochs = 100
        losses_train = []
        max_acc = 0
        patience = 10
        es_counter = 0

        for epoch in range(num_epochs):

            epoch_loss = 0

            for i, data in enumerate(iter(dl_train), 0):
                X, y = data
                yhat = self(X)
                loss = F.cross_entropy(yhat, y.type(torch.LongTensor))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.data.numpy()

            losses_train.append(epoch_loss / (i+1))
            epoch_accuracy = 0

            for i, data in enumerate(iter(dl_dev), 0):
                X, y = data
                yhat = self(X)
                yhat = torch.argmax(F.softmax(yhat, dim=-1), dim=-1)
                accuracy = torch.mean(torch.eq(yhat, y.type(torch.LongTensor)).type(torch.FloatTensor))

                epoch_accuracy += accuracy.data.numpy()

            epoch_accuracy /= (i+1)

            if epoch_accuracy > max_acc:
                max_acc = epoch_accuracy
                es_counter = 0
            else:
                es_counter += 1
                if es_counter == patience:
                    break

        return epoch_accuracy

    def get_score_wt(self, dl_train, dl_dev):
        list_jacob=[]
        
        for i, data in enumerate(iter(dl_train), 0):
            X, y = data
            X.requires_grad_(True)
            yhat = self(X)
            yhat.backward(torch.ones_like(yhat))
            jacob = X.grad.cpu().detach().numpy()
            list_jacob.append(jacob)
        for i, data in enumerate(iter(dl_dev), 0):
            X, y = data
            X.requires_grad_(True)
            yhat = self(X)
            yhat.backward(torch.ones_like(yhat))
            jacob = X.grad.detach()
            list_jacob.append(jacob)
        jacob=np.concatenate(list_jacob).ravel()
        final_score=eval_score(jacob)

        return final_score


    def eval_score(jacob, labels=None):
        corrs = np.corrcoef(jacob)
        v, _  = np.linalg.eig(corrs)
        k = 1e-5
        return -np.sum(np.log(v + k) + 1./(v + k))