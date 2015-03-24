text-clustering
===============

An implementation of textual clustering, using k-means for clustering, and
[cosine similarity](https://github.com/sergeio/text_comparer) as the distance
metric.


Wait, What?
-----------

Basically, if you have a bunch of documents of text, and you want to group them
by similarity into n groups, you're in luck.


Example
-------

To test this out, we can look in `test_clustering.py`:

```python
from vectorizer import cluster_paragraphs
from random import shuffle

text1 = """Type theory is closely related to (and in some cases overlaps with) type systems, which are a programming language feature used to reduce bugs. The types of type theory were created to avoid paradoxes in a variety of formal logics and rewrite systems and sometimes "type theory" is used to refer to this broader application."""
text2 = """The types of type theory were invented by Bertrand Russell in response to his discovery that Gottlob Frege's version of naive set theory was afflicted with Russell's paradox. This theory of types features prominently in Whitehead and Russell's Principia Mathematica. It avoids Russell's paradox by first creating a hierarchy of types, then assigning each mathematical (and possibly other) entity to a type. Objects of a given type are built exclusively from objects of preceding types (those lower in the hierarchy), thus preventing loops."""
text3 = """The common usage of "type theory" is when those types are used with a term rewrite system. The most famous early example is Alonzo Church's lambda calculus. Church's Theory of Types[1] helped the formal system avoid the Kleene-Rosser paradox that afflicted the original untyped lambda calculus. Church demonstrated that it could serve as a foundation of mathematics and it was referred to as a higher-order logic."""
text4 = """A pulsar (short for pulsating radio star) is a highly magnetized, rotating neutron star that emits a beam of electromagnetic radiation. This radiation can only be observed when the beam of emission is pointing toward the Earth, much the way a lighthouse can only be seen when the light is pointed in the direction of an observer, and is responsible for the pulsed appearance of emission. Neutron stars are very dense, and have short, regular rotational periods. This produces a very precise interval between pulses that range from roughly milliseconds to seconds for an individual pulsar."""
text5 = """The precise periods of pulsars make them useful tools. Observations of a pulsar in a binary neutron star system were used to indirectly confirm the existence of gravitational radiation."""
texts = [text1, text2, text3, text4, text5]
shuffle(texts)

cluster_paragraphs(texts, num_clusters=2)
clusters = cluster_paragraphs(texts, num_clusters=2)

print
print 'Group 1:'
print '========\n'
print '\n-----\n'.join(t for t in clusters[0])
print
print 'Group 2:'
print '========\n'
print '\n-----\n'.join(t for t in clusters[1])
print
```

Here, we take some text from two Wikipedia pages, shuffle them into a random
order, and try to pull them back out into two groups automatically.

```bash
$ python2 test_clustering.py

Group 1:
========

The precise periods of pulsars make them useful tools. Observations of a pulsar
in a binary neutron star system were used to indirectly confirm the existence
of gravitational radiation.
-----
A pulsar (short for pulsating radio star) is a highly magnetized, rotating
neutron star that emits a beam of electromagnetic radiation. This radiation can
only be observed when the beam of emission is pointing toward the Earth, much
the way a lighthouse can only be seen when the light is pointed in the
direction of an observer, and is responsible for the pulsed appearance of
emission. Neutron stars are very dense, and have short, regular rotational
periods. This produces a very precise interval between pulses that range from
roughly milliseconds to seconds for an individual pulsar.

Group 2:
========

Type theory is closely related to (and in some cases overlaps with) type
systems, which are a programming language feature used to reduce bugs. The
types of type theory were created to avoid paradoxes in a variety of formal
logics and rewrite systems and sometimes "type theory" is used to refer to this
broader application.
-----
The types of type theory were invented by Bertrand Russell in response to his
discovery that Gottlob Frege's version of naive set theory was afflicted with
Russell's paradox. This theory of types features prominently in Whitehead and
Russell's Principia Mathematica. It avoids Russell's paradox by first creating
a hierarchy of types, then assigning each mathematical (and possibly other)
entity to a type. Objects of a given type are built exclusively from objects of
preceding types (those lower in the hierarchy), thus preventing loops.
-----
The common usage of "type theory" is when those types are used with a term
rewrite system. The most famous early example is Alonzo Church's lambda
calculus. Church's Theory of Types[1] helped the formal system avoid the
Kleene-Rosser paradox that afflicted the original untyped lambda calculus.
Church demonstrated that it could serve as a foundation of mathematics and it
was referred to as a higher-order logic.
```

**Success!**  Group 1 contains only paragraphs about pulsars, and group 2 is
all about type theory!


Warning
-------

Cosine similarity alone is not a sufficiently good comparison function for
good text clustering.  And K-means clustering is not guaranteed to give the
same answer every time.

`test_clustering_probability.py` has some code to test the success rate of
this algorithm with the example data above.  It gives a perfect answer only
60% of the time.

That's not great, but it is not nothing.  Random guessing, even if you knew
how many elements were supposed to be in each cluster, only has a 10% success
rate.
