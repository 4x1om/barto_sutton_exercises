I wasted like 3 hours here trying to debug a weird dip in height at v[20, 20]. I rewrote pretty much every part of the code, logged every state, asked GPT o1, to no avail. Turned out I was overwriting the state value as I'm computing it! And of course, the state (20, 20) transitions to itself most often, so it was affected the most.

Problematic:

```python
v[c1, c2] = rewards[(c1, c2), a]
for c1_prime, c2_prime in states:
    prob = p1[m1, c1_prime] * p2[m2, c2_prime]
    v[c1, c2] += GAMMA * prob * v[c1_prime, c2_prime]
```

Corrected:

```python
v_new = rewards[(c1, c2), a]
for c1_prime, c2_prime in states:
    prob = p1[m1, c1_prime] * p2[m2, c2_prime]
    v_new += GAMMA * prob * v[c1_prime, c2_prime]
v[c1, c2] = v_new
```
