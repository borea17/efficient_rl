# Efficient Reinforcement Learning [![Build Status](https://travis-ci.com/borea17/efficient_rl.svg?token=rFpzsqEK7NXyNhFzhbms&branch=master)](https://travis-ci.com/borea17/efficient_rl)

## About this repository

This is a Python *reimplementation* of 

* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/Thesis.pdf) (C. Diuks Dissertation)
* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/OORL.pdf) (Paper by C. Diuk et al.)

If you are only interested in the results, you can directly jump to the
[Results](https://github.com/borea17/efficient_rl#results). If you
haven't read the dissertation or paper, I can definetly recommend
them. However, I will give a short motivation and summary in [TL;
DR](https://github.com/borea17/efficient_rl#tldr). If you want to use
this repository to reproduce the results or adapt it to for your own
research, there is a short introduction in [How to use this
repository](https://github.com/borea17/efficient_rl#how-to-use-this-repository). 

### TL;DR

#### Motivation

It is a well known empirical fact in reinforcement learning that
model-based approaches (e.g., <i>R</i><sub><font
size="4">max</font></sub>) are more sample-efficient than model-free
algorithms (e.g., <i>Q-learning</i>). One of the main reasons may be
that model-based learning tackles the exploration-exploitation dilemma
in a smarter way by using the accumulated experience to build an
approximate model of the environment. Furthermore, it has also been
shown that rich state representations such as in a *factored MDP* can
make model-based learning even more sample-efficient. *Factored MDP*s
enable an effective parametrization of transition and reward dynamics
by using *dynamic Bayesian networks* (DBNs) to represent partial
dependency relations between state variables, thereby the environment dynamics
can be learned with less samples. A major downside of these approaches
is that the DBNs need to be provided as prior knowledge which might be
impossible sometimes.

Motivated by human intelligence, Diuk et al. introduce a new
framework *propositional object-oriented MDPs* (OO-MDPs) to model
environments and their dynamics. As it turns out, humans are way more
sample efficient than state-of-the-art algorithms when playing games 
such as taxi (Diuk actually performed an experiment). Diuk argues that 
humans must use some prior knowledge when playing this game, he
further speculates that this knowledge might come in form of object
represetations, e.g., identifying horizontal lines as *walls* when 
observing that the taxi cannot move through them. 

Diuk et al. also provide a learning algorithm for deterministic
OO-MDPs (<i>DOOR</i><sub><font size="4">max</font></sub>) which
outperforms *factored* <i>R</i><sub><font size="4">max</font></sub>.
As prior knowledge <i>DOOR</i><sub><font size="4">max</font></sub>
needs the objects and relations to consider, which seems more natural
as these may also be used throughout different games. Furthermore,
this approach may also be used to inherit human biases. 

#### Summary


- adaptations to state-of-the-art Rmax (model based) -> provably efficient algorithm to surpass exploration-exploitation dilemma 
- role of state representations (factored Rmax)

### How to use this repository

In order to use this repositoy, clone it and run the following command in the directoy of the repository
```python
python3 setup.py install
```
To reproduce the results go into `efficient_rl` folder and run 
```python
python3 dissertation_script.py
```
or
```python
python3 paper_script.py
```
Defaultly, each agent runs only once. To increase the number of repetitions change `n_repetitions` in the scripts. 

If you want to use this repository to play a different game, you may want to look at (). Contributions are welcome and if needed, I will provide a more detailed documentation.

## Results

The experimental setup is described on p.31 of Diuks Dissertation. It
consists of testing against six probe states and reporting the number
of steps the agent had to take until the optimal policy for these 6
start states was reached. Since there is some randomness in the
trials, each algorithm runs 100 (`n_repetitions`) times and the
results are then averaged. 

It should be noted that Diuk mainly focused on learning the transition model:
> I will focus on learning dynamics and assume the reward function is available as a black box function (p.61 Diss)

In this reimplementation also the reward function is learned.

### Dissertation (see p.49)

<table>
  <tr>
    <th>Domain knowledge</th>
    <th>Algorithm</th>
    <th colspan="2">Diuks Results<br></th>
    <th colspan="3">Reimplementation Results</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td><u><i># Steps</i></u></td>
    <td><u><i>Time/step</i></u></td>
    <td><u><i># Steps</i></u></td>
    <td><u><i>Time/step</i></u></td>
    <td><u><i>Total Time</i></u></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|<br></td>
    <td>Q-learning</td>
    <td align="center"><b>106859</b></td>
    <td align="center">&lt; 1ms</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td>Q-learning - optimistic <br>initialization</td>
    <td align="center"><b>29350</b></td>
    <td align="center">&lt;1ms</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td><i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>4151</b></td>
    <td align="center">74ms</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td><i>R</i><sub><font size="4">max</font></sub>, DBN structure</td>
    <td>Factored <i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>1676</b></td>
    <td align="center">97.7ms</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>Objects, relations to consider,<br><i>R</i><sub><font size="4">max</font></sub></td>
    <td>DOO<i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>529</b></td>
    <td align="center">48.2ms</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>|<i>A</i>|, visualization of game</td>
    <td>Humans (non-<br>videogamers)<br></td>
    <td align="center"><b>101</b></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
  </tr>
  <tr>
    <td>|<i>A</i>|, visualization of game</td>
    <td>Humans (videogamers)</td>
    <td align="center"><b>48.8</b></td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
    <td align="center">NA</td>
  </tr>
</table>

### Paper (see p.7)

<table>
  <tr>
    <th></th>
    <th colspan="3">Diuks Result</th>
    <th colspan="3">Reimplementation Results</th>
  </tr>
  <tr>
    <td></td>
    <td><i>Taxi 5x5 </i></td>
    <td><i>Taxi 10x10</i></td>
    <td><i>Ratio</i></td>
    <td><i>Taxi 5x5</i><br></td>
    <td><i>Taxi 10x10</i></td>
    <td><i>Ratio</i></td>
  </tr>
  <tr>
    <td>Number of states</td>
    <td align="center">500</td>
    <td align="center" >7200</td>
    <td align="center">14.40</td>
    <td align="center">500</td>
    <td align="center">7200</td>
    <td align="center">14.40</td>
  </tr>
  <tr>
    <td>Q-learning<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    <td align="center">
      &nbsp;<br>NA<br>&nbsp;
    </td>
    <td align="center">
      &nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center"></td>
    <td align="center"></td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>Q-learning - optimistic initialization<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><i>R</i><sub><font size="4">max</font></sub><br> 
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
     </td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td align="center">&nbsp;<br>NA<br>&nbsp;</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Factored <i>R</i><sub><font size="4">max</font></sub><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>1676<br>43.59ms</td>
    <td align="center">&nbsp;<br>19866<br>306.71ms</td>
    <td align="center">&nbsp;<br>11.85<br>7.03</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>DOO<i>R</i><sub><font size="4">max</font></sub><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>529<br>13.88ms</td>
    <td align="center">&nbsp;<br>821<br>293.72ms</td>
    <td align="center">&nbsp;<br>1.55<br>21.16</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>
