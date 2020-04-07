# Efficient Reinforcement Learning

This is a Python *reimplementation* of 

* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/Thesis.pdf) (C. Diuks Dissertation)
* [An Object-Oriented Representation for Efficient Reinforcement Learning](http://carlosdiuk.github.io/papers/OORL.pdf) (Paper by C. Diuk et al.)

## TL;DR



## Results

### Dissertation (see p.49)

<table>
  <tr>
    <th>Domain knowledge</th>
    <th>Algorithm</th>
    <th colspan="2">Result Diuk<br></th>
    <th colspan="3">Reimplementation Results</th>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td># Steps</td>
    <td>Time/step</td>
    <td># Steps</td>
    <td>Time/step</td>
    <td>Total Time</td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|<br></td>
    <td>Q-learning</td>
    <td>106859</td>
    <td>&lt; 1ms</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td>Q-learning - optimistic <br>initialization</td>
    <td>29350</td>
    <td>&lt;1ms</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>|<i>S</i>|, |<i>A</i>|, <i>R</i><sub><font size="4">max</font></sub></td>
    <td><i>R</i><sub><font size="4">max</font></sub></td>
    <td>4151</td>
    <td>74ms</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><i>R</i><sub><font size="4">max</font></sub>, DBN structure</td>
    <td>Factored <i>R</i><sub><font size="4">max</font></sub></td>
    <td>1676</td>
    <td>97.7ms</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td><br>Objects, relations to consider,<br><i>R</i><sub><font size="4">max</font></sub><br></td>
    <td>DOO<i>R</i><sub><font size="4">max</font></sub></td>
    <td>329</td>
    <td>48.2ms</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>|<i>A</i>|, visualization of game</td>
    <td>Humans (non-<br>videogamers)<br></td>
    <td>101</td>
    <td>NA</td>
    <td>NA</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
  <tr>
    <td>|<i>A</i>|, visualization of game</td>
    <td>Humans (videogamers)</td>
    <td>48.8</td>
    <td>NA</td>
    <td>NA</td>
    <td>NA</td>
    <td>NA</td>
  </tr>
</table>

### Paper
