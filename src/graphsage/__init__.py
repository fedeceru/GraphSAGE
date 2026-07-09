"""
graphsage: reference implementation of GraphSAGE
(Hamilton, Ying, Leskovec -- "Inductive Representation Learning on Large
Graphs", NeurIPS 2017).

Package layout, roughly bottom-up:

- ``inits`` / ``layers``            -- small TensorFlow building blocks
                                        (variable initializers, base Layer
                                        class, Dense layer) shared by
                                        everything else.
- ``aggregators``                   -- the AGGREGATE_k functions from
                                        Algorithm 1 of the paper: mean, GCN,
                                        pooling and LSTM variants.
- ``neigh_samplers``                -- uniform neighborhood sampling used to
                                        keep the per-batch cost fixed
                                        (Section 3.1 of the paper).
- ``prediction``                    -- the unsupervised skip-gram-style link
                                        prediction loss (Eq. 1 of the paper).
- ``models``                        -- ``SampleAndAggregate``, the class that
                                        implements Algorithm 1/2 (the actual
                                        "sample and aggregate" forward pass)
                                        plus the unsupervised training
                                        objective; also a Node2Vec/DeepWalk
                                        baseline model.
- ``supervised_models``             -- ``SupervisedGraphsage``, which reuses
                                        the same sample/aggregate machinery
                                        but attaches a supervised
                                        classification loss instead.
- ``minibatch``                     -- minibatch iterators that turn a
                                        networkx graph into the batches of
                                        node/edge ids fed to the models above.
- ``metrics`` / ``utils``           -- loss helpers and dataset loading.
- ``supervised_train`` /
  ``unsupervised_train``            -- command-line entry points that wire
                                        everything together and run the
                                        training loop.
"""
from __future__ import print_function
from __future__ import division
