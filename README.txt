Playing ATARI games using a convolutional autoencoder and an evolutionary algorithm.

Usually, ATARI games are solved using DQN network [1]:
1. Convolutional layers
2. Fully-connected layers
3. Input: raw image, output: Q(s,a)
4. Training: gradient updates via Bellman equations.

We use another approach for training fully connected layers: genetic algorithms.

- Explanation of choice - 
In the game Skiing reward is given only at the end. Therefore, Bellman updates are useless for 99.9% frames.
There are some techniques to overcome such obstacle (use advanced experience replay) [2], but, as the article shows, improvement is insignificant.

Therefore, we use older approach to Atari games: neuroevolution [3], [4]. Specifically, we use NEAT algorithm [5].
It uses a specific representation of the fully connected part of the network, "genome". The algorithm changes genomes the following way:
1. Create random set of NN's
2. Evaluate their fitness (i.e. sum of rewards)
3. Choose the best ones
4. Crossover and mutate them (possibly adding new neurons)
5. Repeat stage 2 for the result.

We train the convolutional part of the network in advance of running neuroevolution.
Specifically, we use convolutional autoencoder:

inp -> conv -> encoded -> deconv -> out

1. Sample 10000 frames from the environment using random actions: action_space.sample()
2. Train the autoencoder in supervised way
3. Remove deconv and out parts of the autoencoder
4. Use 'encoded' features as the description of 'inp'

Code:
1. collect/ -- Autoencoder training & weights:
2. neat_python -- Neuroevolution using python-neat library

Additionally, autoencoder receives not the raw observation, but a frame which roughly follows the idea of "Motion vectors" in video estimation [6]:

  alpha = 0.6
  diff = zeros

  o = env.step()
  diff = (1 - alpha) * diff + alpha * (o - prev_o)
  prev_o = o

'diff' is used as the input for autoencoder

[1] https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
[2] https://arxiv.org/pdf/1511.05952.pdf
[3] http://people.idsia.ch/~koutnik/papers/koutnik2014gecco.pdf
[4] http://www.cs.utexas.edu/users/pstone/Papers/bib2html-links/TCIAIG13-mhauskn.pdf
[5] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
[6] https://en.wikipedia.org/wiki/Motion_vector
