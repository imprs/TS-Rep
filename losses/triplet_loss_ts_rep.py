# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import torch
import torch.nn.functional as F
import numpy
from nearest_neighbour import NearestNeighbour

torch.set_printoptions(linewidth=120)

class TripletLoss(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.

    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.

    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                device, latent_shape):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.latent_shape = latent_shape
        self.use_nn = True
        self.n_neighbour = NearestNeighbour(device=device, train_shape=latent_shape)
        self.nn_epoch = 30
        self.sampling_type_str = "nonoverlapping"
        self.neg_loss_type_str = "batch"

    def forward(self, batch_idx, batch, encoder, train, epochs, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        length = min(self.compared_length, train.size(2))

        random_length, beginning_batches, length_pos_neg, beginning_positive = \
            pos_sampling(self.sampling_type_str, length, batch_size, train_size)

        representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length
            ] for j in range(batch_size)]
        ))  # Anchors representations

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_positive[j]: beginning_positive[j] + length_pos_neg
            ] for j in range(batch_size)]
        ))  # Positive samples representations

        loss = self.loss_mm(epochs, batch_size, batch, batch_idx, representation, positive_representation, encoder, length_pos_neg, save_memory, length, train)
        return loss

    def loss_mm(self, epochs, batch_size, batch, batch_idx, representation, positive_representation, encoder, length_pos_neg, save_memory, length, train):
        size_representation = representation.size(1)
        if self.use_nn:
            # Positive representations in the queue
            self.n_neighbour.update_queue(batch_idx, positive_representation)

            if epochs >= self.nn_epoch:
                positive_representation_nn, nn_idx = self.n_neighbour.nearest_neighbour(
                    batch_idx, positive_representation) # nearest neighbour of positive

                # Positive loss: -logsigmoid of dot product between anchor and positive
                # representations
                loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    positive_representation_nn.view(batch_size, size_representation, 1)
                )))
            else:
                # Positive loss: -logsigmoid of dot product between anchor and positive
                # representations
                loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    positive_representation.view(batch_size, size_representation, 1)
                )))
        else:
            # Positive loss: -logsigmoid of dot product between anchor and positive
            # representations
            loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                representation.view(batch_size, 1, size_representation),
                positive_representation.view(batch_size, size_representation, 1)
            )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            if self.use_nn and epochs >= self.nn_epoch:
                del positive_representation_nn
            else:
                del positive_representation
            torch.cuda.empty_cache()

        neg_loss = self.negative_loss(self.neg_loss_type_str, encoder, batch, length,
                                    length_pos_neg, representation, train, save_memory)

        loss += neg_loss
        print(f'total loss fn: {loss}')
        return loss

    def negative_loss(self, neg_loss_type_str, encoder, batch, length,
                                    length_pos_neg, representation, train, save_memory):
        if neg_loss_type_str == "train":
            neg_loss = self.train_negatives_loss(encoder, batch, length,
                                    length_pos_neg, representation, train, save_memory)
        elif neg_loss_type_str == "batch":
            neg_loss = self.batch_negatives_loss(encoder, batch, length,
                                    length_pos_neg, representation, train, save_memory)
        else:
            raise ValueError(f'neg_loss_type_str = {neg_loss_type_str} is invalid!')
        return neg_loss


    def train_negatives_loss(self, encoder, batch, length, length_pos_neg, representation, train, save_memory):
        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        batch_size = batch.size(0)
        train_size = train.size(0)
        loss = 0

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(self.nb_random_samples, batch_size)
        )

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat([train[samples[i, j]: samples[i, j] + 1][
                    :, :,
                    beginning_samples_neg[i, j]:
                    beginning_samples_neg[i, j] + length_pos_neg
                ] for j in range(batch_size)])
            )
            size_representation = representation.size(1)
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()
        return loss


    def batch_negatives_loss(self, encoder, batch, length, length_pos_neg, representation, train, save_memory):
        batch_size = batch.size(0)

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(batch_size)
        ) # Shape: (B,)

        end_samples_neg = beginning_samples_neg + length_pos_neg

        # Negative loss: -logsigmoid of minus the dot product between
        # anchor and negative representations
        negative_representation = encoder(torch.cat(
            [batch[j: j + 1, :,
                beginning_samples_neg[j]:
                end_samples_neg[j]
            ] for j in range(batch_size)]
        )) # Shape: (B,output_dim)

        sim_neg = torch.mm(representation, negative_representation.transpose(1,0))

        """ Without mask multiply
        sim_neg = torch.from_numpy(sim_neg.numpy()[~numpy.eye(batch_size,dtype=bool)
                        ].reshape(batch_size,-1)) # Remove main diagonal # Shape: (B,B-1)
        total_output = (torch.nn.functional.logsigmoid(-1 * sim_neg))
        loss = -self.negative_penalty * torch.mean(total_output)
        """
        self_mask = torch.DoubleTensor(numpy.logical_not(numpy.eye(batch_size)).astype(numpy.double)).to(batch.device)
        total_output = (torch.nn.functional.logsigmoid(-1 * sim_neg)) * self_mask
        loss = -self.negative_penalty * torch.mean(torch.sum(total_output, 1)/ (batch_size-1))
        return loss


class TripletLossVaryingLength(torch.nn.modules.loss._Loss):
    """
    Triplet loss for representations of time series. Optimized for training
    sets where all time series have the same length.

    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing
    the training set, where `B` is the batch size, `C` is the number of
    channels and `L` is the length of the time series, as well as a boolean
    which, if True, enables to save GPU memory by propagating gradients after
    each loss term, instead of doing it after computing the whole loss.

    The triplets are chosen in the following manner. First the size of the
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.

    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                device, latent_shape):
        super(TripletLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.latent_shape = latent_shape
        self.use_nn = True
        self.n_neighbour = NearestNeighbour(device=device, train_shape=latent_shape)
        self.nn_epoch = 30
        self.sampling_type_str = "nonoverlapping" # overlapping is not implemented for varying.
        self.neg_loss_type_str = "batch"

    def forward(self, batch_idx, batch, encoder, train, epochs, save_memory=False):
        batch_size = batch.size(0)
        train_size = train.size(0)
        max_length = train.size(2)

        # length (batch_size) of the samples in the batch.
        length = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
                ).data.cpu().numpy()

        random_length, beginning_batches, length_pos_neg, beginning_positive = \
            pos_sampling(self.sampling_type_str, length, batch_size, train_size)

        # We have to run forward loop on anchor one at a time due to different lengths.
        # Notice how we pass it through encoder first and concat later opposed to positve
        # representation below.
        representation = torch.cat(
            [encoder(batch[
                j: j + 1, :,
                beginning_batches[j]: beginning_batches[j] + random_length[j]
            ]) for j in range(batch_size)]
        )  # Anchors representations # Shape: (B,output_dim)

        positive_representation = encoder(torch.cat(
            [batch[
                j: j + 1, :,
                beginning_positive[j]: beginning_positive[j] + length_pos_neg
            ] for j in range(batch_size)]
        ))  # Positive samples representations # Shape: (B,output_dim)

        loss = self.loss_mm(epochs, batch_size, batch, batch_idx, representation, positive_representation, encoder, length_pos_neg, save_memory, length, train)
        return loss

    def loss_mm(self, epochs, batch_size, batch, batch_idx, representation, positive_representation, encoder, length_pos_neg, save_memory, length, train):
        size_representation = representation.size(1)
        if self.use_nn:
            # Positive representations in the queue
            self.n_neighbour.update_queue(batch_idx, positive_representation)

            if epochs >= self.nn_epoch:
                positive_representation_nn, nn_idx = self.n_neighbour.nearest_neighbour(
                    batch_idx, positive_representation) # nearest neighbour of positive

                # Positive loss: -logsigmoid of dot product between anchor and positive
                # representations
                loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    positive_representation_nn.view(batch_size, size_representation, 1)
                )))
            else:
                # Positive loss: -logsigmoid of dot product between anchor and positive
                # representations
                loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    positive_representation.view(batch_size, size_representation, 1)
                )))
        else:
            # Positive loss: -logsigmoid of dot product between anchor and positive
            # representations
            loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
                representation.view(batch_size, 1, size_representation),
                positive_representation.view(batch_size, size_representation, 1)
            )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            if self.use_nn and epochs >= self.nn_epoch:
                del positive_representation_nn
            else:
                del positive_representation
            torch.cuda.empty_cache()

        neg_loss = self.negative_loss(self.neg_loss_type_str, encoder, batch, length,
                                    length_pos_neg, representation, train, save_memory)

        loss += neg_loss
        print(f'total loss fn: {loss}')
        return loss

    def negative_loss(self, neg_loss_type_str, encoder, batch, length,
                                    length_pos_neg, representation, train, save_memory):
        if neg_loss_type_str == "batch":
            neg_loss = self.batch_negatives_loss(encoder, batch, length,
                                    length_pos_neg, representation, train, save_memory)
        else:
            raise ValueError(f'neg_loss_type_str = {neg_loss_type_str} is invalid!')
        return neg_loss


    def batch_negatives_loss(self, encoder, batch, length, length_pos_neg, representation, train, save_memory):
        batch_size = batch.size(0)

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1,
            size=(batch_size)
        ) # Shape: (B,)

        end_samples_neg = beginning_samples_neg + length_pos_neg

        # Negative loss: -logsigmoid of minus the dot product between
        # anchor and negative representations
        negative_representation = encoder(torch.cat(
            [batch[j: j + 1, :,
                beginning_samples_neg[j]:
                end_samples_neg[j]
            ] for j in range(batch_size)]
        )) # Shape: (B,output_dim)

        sim_neg = torch.mm(representation, negative_representation.transpose(1,0))

        """ Without mask multiply
        sim_neg = torch.from_numpy(sim_neg.numpy()[~numpy.eye(batch_size,dtype=bool)
                        ].reshape(batch_size,-1)) # Remove main diagonal # Shape: (B,B-1)
        total_output = (torch.nn.functional.logsigmoid(-1 * sim_neg))
        loss = -self.negative_penalty * torch.mean(total_output)
        """
        self_mask = torch.DoubleTensor(numpy.logical_not(numpy.eye(batch_size)).astype(numpy.double)).to(batch.device)
        total_output = (torch.nn.functional.logsigmoid(-1 * sim_neg)) * self_mask
        loss = -self.negative_penalty * torch.mean(torch.sum(total_output, 1)/ (batch_size-1))
        return loss


def pos_sampling(sampling_type_str, length, batch_size, train_size):
    if sampling_type_str == "overlapping":
        samples = overlapping_sampling(length, batch_size, train_size)
    elif sampling_type_str == "nonoverlapping":
        samples = nonoverlapping_sampling(length, batch_size, train_size)
    else:
        raise ValueError(f'sampling_type_str = {sampling_type_str} is invalid!')
    return samples


def overlapping_sampling(length, batch_size, train_size):
    # Choice of length of positive and negative samples
    length_pos_neg = numpy.random.randint(1, high=length + 1)

    # We choose for each batch example a random interval in the time
    # series, which is the 'anchor'
    random_length = numpy.random.randint(
        length_pos_neg, high=length + 1
    )  # Length of anchors
    beginning_batches = numpy.random.randint(
        0, high=length - random_length + 1, size=batch_size
    )  # Start of anchors

    # The positive samples are chosen at random in the chosen anchors
    beginning_samples_pos = numpy.random.randint(
        0, high=random_length - length_pos_neg + 1, size=batch_size
    )  # Start of positive samples in the anchors
    # Start of positive samples in the batch examples
    beginning_positive = beginning_batches + beginning_samples_pos
    # End of positive samples in the batch examples
    end_positive = beginning_positive + length_pos_neg

    return random_length, beginning_batches, length_pos_neg, beginning_positive


def nonoverlapping_sampling(length, batch_size, train_size):
    if numpy.isscalar(length): # Type: int for fixed length
        min_length = length
    else: # Type: array for varying length
        min_length = min(length)

    # Choice of length of positive and negative samples
    # length_pos_neg = numpy.random.randint(1, high=length + 1)
    length_pos_neg = numpy.random.randint(int(0.3*min_length), high=int(0.7*min_length) + 1) # Type: int

    random_length = length - length_pos_neg # Length of anchors  # Type: int (fixed length)/ array (varying)

    # Sample an array of size (batch_size) on whether to
    # start anchor at 0 or length_pos_neg.
    beginning_batches = numpy.random.choice(
        [0, length_pos_neg], size=batch_size) # Start of anchors # Shape: (B,)

    # Do opposite of beginning_batches i.e., if an anchor
    # start at 0 then start positive from random_length (anchor
    # length), meaning where anchor finishes (random_length-1).
    # If an anchor start at length_pos_neg  then start positive
    # at 0, and this positive will finish at (length_pos_neg-1).

    # Start of positive samples in the batch examples
    # beginning_positive = beginning_batches + beginning_samples_pos
    beginning_positive = (beginning_batches==0) *  (random_length) # Shape: (B,)

    # End of positive samples in the batch examples
    end_positive = beginning_positive + length_pos_neg # Shape: (B,)

    return random_length, beginning_batches, length_pos_neg, beginning_positive
