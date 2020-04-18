# Efficient Reinforcement Learning
[![Build Status](https://travis-ci.com/borea17/efficient_rl.svg?token=rFpzsqEK7NXyNhFzhbms&branch=master)](https://travis-ci.com/borea17/efficient_rl) [![PyPI version](https://badge.fury.io/py/efficient-rl.svg)](https://badge.fury.io/py/efficient-rl)

**[Motivation](https://github.com/borea17/efficient_rl#motivation)** | **[Summary](https://github.com/borea17/efficient_rl#summary)** | **[Results](https://github.com/borea17/efficient_rl#results)** | **[How to use this repository](https://github.com/borea17/efficient_rl#how-to-use-this-repository)**

-------------------------------------------------------------------------------------

This is a Python *reimplementation* for the Taxi domain of 

* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/Thesis.pdf) (C. Diuk's Dissertation)
* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/OORL.pdf) (Paper by C. Diuk et al.)

<table>
<tbody>
<tr>
  <td>In the <i>Taxi domain</i> the goal is to navigate the <i>taxi</i> (initially yellow box) towards<br>
    the <i>passenger</i> (blue letter), take a <i>Pickup</i> action and then deliver the <i>taxi with<br>passenger inside</i> (green box) towards the <i>destination</i> (magenta letter) and perform <br> a <i>Dropoff</i> action. A reward of -1 is obtained for every time step it takes until delivery. <br>Successful <i>Dropoff</i> results in +20 reward, while non-successful <i>Dropoff</i> or <i>Pickup</i> is<br> penalized with -10. 
    This task was introduced by <a href="https://arxiv.org/abs/cs/9905014">Dietterich</a>.
 </td>
  <td><img src='gifs/example.gif' width='120' height='185.25'></td>
</tr>
</tbody>
</table>

-------------------------------------------------------------------------------------
### Motivation

It is a well known empirical fact in reinforcement learning that
model-based approaches (e.g., <i>R</i><sub><font
size="4">max</font></sub>) are more sample-efficient than model-free
algorithms (e.g., <i>Q-learning</i>). One of the main reasons may be
that model-based learning tackles the exploration-exploitation dilemma
in a smarter way by using the accumulated experience to build an
approximate model of the environment. Furthermore, it has been 
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
sample-efficient than state-of-the-art algorithms when playing games 
such as Taxi (Diuk actually performed an experiment). Diuk argues that 
humans must use some prior knowledge when playing this game, he
further speculates that this knowledge might come in form of object
representations, e.g., identifying horizontal lines as *walls* when 
observing that the taxi cannot move through them. 

Diuk et al. provide a learning algorithm for deterministic
OO-MDPs (<i>DOOR</i><sub><font size="4">max</font></sub>) which
outperforms *factored* <i>R</i><sub><font size="4">max</font></sub>.
As prior knowledge <i>DOOR</i><sub><font size="4">max</font></sub>
needs the objects and relations to consider, which seems more natural
as these may also be used throughout different games. Furthermore,
this approach may also be used to inherit human biases. 

-------------------------------------------------------------------------------------

### Summary

This part shall give an overview about the different reimplemented
algorithms. These can be divided into *model-free* and *model-based* approaches.

#### Model-free Approaches

In model-free algorithms the agent learns the optimal action-value
function (or value function or policy) directly from experience
without having an actual model of the environment. Probably the most
famous model-free algorithm is *Q-learning* which also builds the
basis for the (perhaps even more famous) [DQN paper](https://arxiv.org/abs/1312.5602).

##### Q-learning

Q-learning aims to approximate the optimal action-value function 
from which the optimal policy can be inferred. In the simplest case, 
a table (*Q-table*) is used as a function approximator. 

The basic idea is to start with a random action-value function and
then iteratively update this function towards the optimal action-value
function. The update comes after each action *a* with the observed
reward *r* and new state *s<sup>'</sup>*, the update rule is very
simple and is derived from Bellman's optimality equation:

|![\displaystyle Q(s,a)\leftarrow (1-\alpha) Q(s,a) + \alpha\left\[r + \gamma \max_{a^{'}} Q(s^{'}, a^{'})\right\]](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle%20Q(s%2Ca)%5Cleftarrow%20(1-%5Calpha)%20Q(s%2Ca)%20%2B%20%5Calpha%5Cleft%5Br%20%2B%20%5Cgamma%20%5Cmax_%7Ba%5E%7B'%7D%7D%20Q(s%5E%7B'%7D%2C%20a%5E%7B'%7D)%5Cright%5D)|
|:---:|
  
where &alpha; is the learning rate. To allow for exploration,
Q-learning commonly uses *&epsi;-greedy exploration* or the *Greedy
in the Limit with Infinite Exploration* approach (see [David Silver,
p.13
ff](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)).

Diuk uses two variants of Q-learning:
* **Q-learning**: standard Q-learning approach with &epsi;-greedy
  exploration where parameters &alpha;=0.1 and &epsi;=0.6 have been
  found via parameter search.
* **Q-learning with optimistic initialization**: instead of some
  random initialization of the Q-table a smart initialization to an
  optimistic value (maximum possible value of any state action pair 
  <img
  src="https://render.githubusercontent.com/render/math?math=v_{max}">)
  is used. Thereby unvisited state-action pairs become more likely to be
  visited. Here, &alpha; was is to 1 (deterministic environment) and
  &epsi; to 0 (exploration ensured via initialization).

#### Model-based Approaches 

In model-based approaches the agent learns a model of the environment
by accumulating experience. Then, an optimal action-value
function (or value function or policy) is obtained through *planning*. Planning
can be done exactly or approximately. In the experiments, Diuk et al.
use exact planning, more precisely *value iteration*. The difference
between the following three algorithms lies in the way they learn the
environment dynamics.

##### R<sub><font size="4">max</font></sub>

R<sub><font size="4">max</font></sub> is a provably efficient
state-of-the-art algorithm to surpass the exploration-exploitation
dilemma through an intuitive approach: R<sub><font
size="4">max</font></sub> divides state-action pairs into *known*
(state-action pairs which have been visited often enough to build an
accurate transition/reward function) and *unknown*. Whenever a state
is *known*, the algorithm uses the empirical transition and reward
function for planning. In case a state is *unknown*,  R<sub><font
size="4">max</font></sub> assumes a transition to a fictious state
from which maximum reward can be obtained consistently (hence the name) and it uses
that for planning. Therefore, actions which have not been tried out
(often enough) in the actual state will be preferred unless the
*known* action also leads to maximal return. The parameter *M* defines the number of
observations the agent has to see until it considers a
transition/reward to be known, in a deterministic case such as the
Taxi domain, it can be set to 1. R<sub><font size="4">max</font></sub>
is guaranteed to find a near-optimal action-value function in
polynomial time.

###### Learning transition and reward dynamics

The 5x5 Taxi domain has 500 different states:
 - 5 *x* positions for taxi
 - 5 *y* positions for taxi
 - 5 passenger locations (4 designated locations plus *in-taxi*)
 - 4 destinations

In the standard R<sub><font size="4">max</font></sub> approach without
any domain knowledge (except for the maximum possible reward
*R*<sub><font size="4">max</font></sub>, the
number of states *|S|*, the number of actions *|A|*), the states are
simply enumerated and the agent will
not be able to transfer knowledge throughout the domain. E.g.,
assume the agent performs an action *North* at some location on the
grid and learns the state transition (more precisely it would learn
something like *picking action 1 at state 200 results in ending up in
state 220*). Being at the same location but with a different *passenger location* or *destination
location* the agent will not be able to predict the outcome of action
*North*. It will take the agent at least 3000 (*|S| &middot; |A|*)
steps until it has fully learned the 5x5 Taxi transition dynamics.
Furthermore, the learned transition and reward dynamics are rather
difficult to interpret.
To address this shortcoming, the agent needs a different
representation and some prior knowledge.

##### Factored R<sub><font size="4">max</font></sub>

Factored R<sub><font size="4">max</font></sub> is a R<sub><font
size="4">max</font></sub> adaptation that builds on a *factored MDP*
environment representation. In a *factored MDP* a state is represented
as tuple (hence *factored state*), e.g., in the Taxi domain the state
can be represented as the 4-tuple 
> [taxi x location, taxi y location, passenger location, passenger destination]

(Note that *passenger location* actually enumerates the different *(x,
y)* start passenger locations plus whether the passenger is *in
taxi*.) This representation allows to represent partial dependency
relations for the environment dynamics between variables using
*Dynamic Bayesian Networks (DBNs)*. E.g., for action *North* we know
that each state variable at time *t+1* only depends on its own value at time *t*, i.e., the *x
location* at time *t+1* under action *North* is independent of the *y
location*, *passenger location* and *passenger destination* at time
*t*. This knowledge is encoded in a *DBN* (each action may have a
different DBN) and it enables Factored R<sub><font
size="4">max</font></sub> to much more sample-efficient learning. 
The downside of this approach is that this kind of prior knowledge may not
be available and that it lacks some generalization, e.g., although
Factored R<sub><font size="4">max</font></sub> knows that the *x location* is independent of all other state
variables, Factored R<sub><font size="4">max</font></sub> still needs
to perfom action *North* at each *x location* to learn the outcome.


##### DOOR<sub><font size="4">max</font></sub>

DOOR<sub><font size="4">max</font></sub> is a R<sub><font
size="4">max</font></sub> adaptation that builds on a *deterministic (propositional)
object-oriented MDP (OO MDP)* environment representation. This representation
is based on objects and their interactions, a state is presented as 
the union of all (object) attribute values. Additionally, each state
has an attributed boolean vector describing which *relations* are
enabled and which are not in that state. During a transition each
attribute of the state may exert some kind of *effect* which results in
an attribute change. There are some limitations to the *effects* that can
occur which are well explained in Diuk's dissertation. The basic idea
of DOOR<sub><font size="4">max</font></sub> is to recover the
deterministic OO MDP using *condition-effect learners* (in these
learners *conditions* are basically the relations that need to hold in
order for an effect to occur).
The paper results show that in DOOR<sub><font size="4">max</font></sub>
knowledge can much better transfer throughout the domain compared to the
other algorithms indicating that DOOR<sub><font size="4">max</font></sub>
offers better generalization. Another feature is that the learned transition
dynamics is easy to interpret, e.g., DOOR<sub><font
size="4">max</font></sub> will learn that action *North* has the
effect ot incrementing *taxi.y* by 1 when the relation
*touch_north(taxi, wall)* outputs *False* and there wont be any change
in *taxi.y* if *touch_north(taxi, wall)* outputs *True*.

-------------------------------------------------------------------------------------

### Results

#### Experimental Setup 

The experimental setup is described on p.31 of Diuk's Dissertation or
p.7 of the paper. It consists of testing against six probe states and reporting the number
of steps the agent had to take until the optimal policy for these 6
start states was reached. Since there is some randomness in the
trials, each algorithm runs 100 times and the results are then averaged. 

#### Differences between Reimplementation and Diuk

There are some differences between this reimplementation and Diuk's
approach which are listed below:

1) For educational purposes, the reward function is also learned in
   this reimplementation (always in the simplest possible way). Note that Diuk mainly focused on learning the
   transition model:
   > I will focus on learning dynamics and assume the reward function is available as a black box function.
2) It is unknwon whether in Diuk's setting during training the passenger start location
   and destination could be the same. The original definition by
   [Diettetrich](https://arxiv.org/abs/cs/9905014) states:
   
   > To keep things uniform, the taxi must pick up and drop off the passenger even if he/she is already at the destination.  
   
   Therefore, in this reimplementation this was also possible during
   training. While the results for R<sub><font
   size="4">max</font></sub> an its adaptations indicate that Diuk used
   the same setting, there is a discrepancy for Q-learning. When the
   setting was changed such that passenger start and destination could
   not be the same (these are the results in brackets), similar results to Diuk could be obtained.
3) Some implementation details are different such as the update procedure
   of the empirical transition and reward functions or the
   condition-effect-learners which were not well enough documented or
   which did not fit into the reimplementation structure.

#### Dissertation Results (p.49)

The dissertation results align with the reimplementation results. Clearly, DOOR<sub><font size="4">max</font></sub> outperfroms the other algorithms in terms of sample-efficiency.

For the differences in *Q-Learning* and the values in brackets, refer to
2) of [Differences between Reimplementation and Diuk](https://github.com/borea17/efficient_rl/#differences-between-reimplementation-and-diuk).

The results were obtained on a lenovo thinkpad yoga 260 (i7-6500 CPU
@ 2.50 GHz x 4).

<table>
  <tr>
    <th>Domain knowledge</th>
    <th>Algorithm</th>
    <th colspan="2">Diuk's Results<br></th>
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
    <td align="center"><b>117716</b><br>(118329)</td>
    <td align="center">&lt;1ms<br>(&lt;1ms)</td>
    <td align="center">4.3s<br>(4.5s)</td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td>Q-learning - optimistic <br>initialization</td>
    <td align="center"><b>29350</b></td>
    <td align="center">&lt;1ms</td>
    <td align="center"><b>75219</b><br>(28154)</td>
    <td align="center">&lt;1ms<br>(&lt;1ms)</td>
    <td align="center">3.7s<br>(1.5s)</td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td><i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>4151</b></td>
    <td align="center">74ms</td>
    <td align="center"><b>4087</b></td>
    <td align="center">2.9ms</td>
    <td align="center">11.8s</td>
  </tr>
  <tr>
    <td><i>R</i><sub><font size="4">max</font></sub>, DBN structure</td>
    <td>Factored <i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>1676</b></td>
    <td align="center">97.7ms</td>
    <td align="center"><b>1718</b></td>
    <td align="center">30ms</td>
    <td align="center">51.6s</td>
  </tr>
  <tr>
    <td>Objects, relations to consider,<br><i>R</i><sub><font size="4">max</font></sub></td>
    <td>DOO<i>R</i><sub><font size="4">max</font></sub></td>
    <td align="center"><b>529</b></td>
    <td align="center">48.2ms</td>
    <td align="center"><b>483</b></td>
    <td align="center">34.3ms</td>
    <td align="center">16.5s</td>
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

#### Paper Results (p.7)

The paper results align with the reimplementation results. These results show that DOOR<sub><font size="4">max</font></sub>  not only outperforms Factored R<sub><font size="4">max</font></sub> in terms of sample-efficiency, but also scales much better to larger problems. Note that the number of states increases by a factor of more than 14 times. 

The results were obtained on a cluster from which I do not know the CPU specifics (this is not too important since the focus lies on the comparison). Note that Diuk et al. used a more powerful machine for the paper result: the average step times are notably smaller compared to the dissertation results. 

<table>
  <tr>
    <th></th>
    <th colspan="3">Diuk's Result</th>
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
    <td>Factored <i>R</i><sub><font size="4">max</font></sub><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>1676<br>43.59ms</td>
    <td align="center">&nbsp;<br>19866<br>306.71ms</td>
    <td align="center">&nbsp;<br>11.85<br>7.03</td>
    <td align="center">&nbsp;<br>1589<br>23.7ms</td>
    <td align="center">&nbsp;<br>16274<br>442.7ms</td>
    <td align="center"></td>
  </tr>
  <tr>
    <td>DOO<i>R</i><sub><font size="4">max</font></sub><br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# steps<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Time per step</td>
    </td>
    <td align="center">&nbsp;<br>529<br>13.88ms</td>
    <td align="center">&nbsp;<br>821<br>293.72ms</td>
    <td align="center">&nbsp;<br><b>1.55</b><br>21.16</td>
    <td align="center">&nbsp;<br>452<br>27.45ms</td>
    <td align="center">&nbsp;<br>949<br>320.6ms</td>
    <td align="center"></td>
  </tr>
</table>




### How to use this repository

#### Installation

##### Building from Source

```bash
git clone --depth 1 https://github.com/borea17/efficient_rl/
cd efficient_rl
python setup.py install
```

##### Via Pip

```bash
pip install efficient_rl
```

#### Reproduce results

After successful installation, download `dissertation_script.py` and `paper_script.py` (which are in folder [efficient_rl](https://github.com/borea17/efficient_rl/tree/master/efficient_rl)), then run

```bash
python dissertation_script.py 
python paper_script.py
```

Defaultly, each agent runs only once. To increase the number of repetitions change `n_repetitions` in the scripts. 

WARNING: It is not recommended to run `paper_script.py` on a standard computer as it may take
several hours.

#### Contributions

If you want to use this repository for a different environment, you
may want to have a look at `efficient_rl/environment` folder. There is
a self written environment called `TaxiEnvironmentClass.py` and there
are extensions to the `gym` Taxi environment in the corresponding folders. 

Contributions are welcome and if needed, I will provide a more detailed documentation.
