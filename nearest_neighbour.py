import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.neighbors import NearestNeighbors

class NearestNeighbour(object):
    def __init__(self, device, train_shape):
        self.device = device
        self.train_size = train_shape[0]
        self.latent_dim = train_shape[1]
        self.representations_queue = torch.zeros(size=train_shape, dtype=torch.double, device=device)
        self.distance = 'L1'
        self.bernoulli_p = 0.6 # Chances of getting 1 and looking for nearest neighbour

    def update_queue(self, batch_idx, batch_representation):
        # print(f'batch shape: {batch_representation.shape}')
        self.representations_queue[batch_idx[:]] = batch_representation.detach()
       

    def nearest_neighbour(self, batch_idx, batch_representation):
        alpha = torch.bernoulli(torch.tensor(self.bernoulli_p)) 
        if alpha:
            # Find nearest neighbour
            temp_representations = self.representations_queue.clone()

            # Replace samples from the batch with high values so indicies don't find rep of themselves
            temp_representations[batch_idx[:]] = 1000 * torch.ones(
                                                            size=(len(batch_idx), self.latent_dim),
                                                            dtype=torch.double,
                                                            device=self.device
                                                            )

            if self.distance == 'L1':
                idx_nn = self.NN_predict(batch_representation.detach(), temp_representations, distance='L1')
            elif self.distance == 'L2':
                idx_nn = self.NN_predict(batch_representation.detach(), temp_representations, distance='L2')
            else:
                raise ValueError(f'NearestNeighbour(): distance {self.distance} is not valid!')
            nearest_neighbours = \
                torch.index_select(self.representations_queue, dim=0, index=idx_nn)

            return nearest_neighbours, idx_nn
        else:
            # Return the batch as it is
            return batch_representation, batch_idx


    def NN_predict(self, batch_representation, representations_queue, distance):
        '''A little slower way
        batch_size = batch_representation.shape[0]
        indicies = np.zeros(batch_size)

        for i in range(batch_size):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            if distance == 'L1':
                distances = torch.sum(torch.abs(representations_queue - batch_representation[i,:]), dim=1)
            # using the L2 distance (sum of absolute value differences)
            if distance == 'L2':
                distances = torch.sqrt(
                                torch.sum(
                                    torch.square(representations_queue - batch_representation[i,:]),
                                dim=1)
                            )
            min_index = torch.argmin(distances) # get the index with smallest distance
            indicies[i] = min_index
            #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
        print(f'torch:\n{distance}_idx: {indicies}')
        '''
        if distance == 'L1':
            # cist needs (B,P,M) and (B,R,M) shaped tensors to compute pairwise and
            # generate distances of (B,P,R) shape. So, we first convert (R,M) to (B,P,M)
            distances_cdist = torch.cdist(torch.unsqueeze(batch_representation, dim=0),
                                        torch.unsqueeze(representations_queue, dim=0), p=1.0)
        if distance == 'L2':
            distances_cdist = torch.cdist(torch.unsqueeze(batch_representation, dim=0),
                                        torch.unsqueeze(representations_queue, dim=0), p=2.0)

        # Now we convert the distances of shape (B,P,R) to (P,R).
        distances_cdist = torch.squeeze(distances_cdist, dim=0)

        # For each sample in the batch, find a sample with the smallest distance.
        indicies_cdist = torch.argmin(distances_cdist, dim=1)

        print(f'cdist:\n{distance}_idx: {indicies_cdist}')
        return indicies_cdist
